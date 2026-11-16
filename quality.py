"""Per-job quality metrics for the SAM enhancement worker.

Privacy contract
----------------
Everything returned is a flat dict of rounded scalars (bool, int, float with 3
decimals, or None) plus one constant version string. No image, mask, crop,
embedding, coordinate list or free text is ever returned, stored or logged by
this module. Nothing here prints.

Safety contract
---------------
- Called only after the output PNG has been encoded, on read-only views, so a
  bug here cannot change the image (a write raises, and is swallowed).
- `compute_with_budget` returns None on any exception or when the budget is
  exceeded. A metrics failure never fails a job.

Each metric is documented in METRICS.md. Bump METRICS_VERSION whenever a
definition changes, so historical rows stay comparable.
"""

import math
import threading
import time

import cv2
import numpy as np

WORKER_VERSION = "sam-v5-q1"
METRICS_VERSION = 1
BUDGET_S = 2.0

# Photo-quality metrics are measured on the box crop resized so its longest side
# is this many pixels, so they do not depend on the phone's resolution.
_PHOTO_SIDE = 512
# Up to this many mask pixels feed the PCA; larger masks are sampled on a grid.
_PCA_MAX_POINTS = 250_000
# Pixels this close to the pasted region are excluded from outside_change.
_OUTSIDE_MARGIN_PX = 5

# The complete, ordered schema. Anything not listed here can never be returned.
KEYS = (
    "worker_version", "metrics_version",
    # SAM cutout
    "mask_score", "mask_score_margin", "mask_box_fill", "mask_outside_box",
    "mask_touches_box_edge", "mask_components", "mask_largest_frac",
    "mask_angle_deg", "mask_elongation", "center_on_mask",
    # Box (user error or app box mapping)
    "box_area_frac", "box_aspect", "box_out_of_image",
    # Photo (user error)
    "box_luma_mean", "box_luma_std", "box_sharpness",
    # Orientation (app EXIF handling)
    "exif_orientation", "exif_rotated", "image_w", "image_h",
    # Stretch and paste (algorithm limits)
    "clipped_bottom", "clipped_side", "seam_score", "outside_change",
)


class _BudgetExceeded(Exception):
    pass


def _num(value):
    """Native, finite, rounded scalar or None. JSON never sees NaN or numpy types."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 3) if math.isfinite(value) else None
    raise TypeError("metric is not a scalar")


def _readonly(array):
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


def _sanitise(raw):
    out = {}
    for key in KEYS:
        value = raw.get(key)
        out[key] = value if key == "worker_version" else _num(value)
    if out["worker_version"] != WORKER_VERSION:
        raise ValueError("unexpected version")
    return out


def compute(original, enhanced, masks, scores, best_idx, box, paste_info, exif_orientation, deadline=None):
    """Returns the sanitised metrics dict. Raises on any problem (callers catch)."""

    def check():
        if deadline is not None and time.perf_counter() > deadline:
            raise _BudgetExceeded()

    original = _readonly(original)
    enhanced = _readonly(enhanced)
    mask = _readonly(np.asarray(masks[best_idx]).astype(bool, copy=False))
    scores = np.asarray(scores, dtype=np.float64).ravel()
    h, w = original.shape[:2]
    m = {"worker_version": WORKER_VERSION, "metrics_version": METRICS_VERSION,
         "image_w": w, "image_h": h}

    # --- Orientation --------------------------------------------------------
    orientation = exif_orientation if isinstance(exif_orientation, int) and 1 <= exif_orientation <= 8 else None
    m["exif_orientation"] = orientation
    m["exif_rotated"] = orientation is not None and orientation != 1

    # --- SAM scores ---------------------------------------------------------
    ordered = np.sort(scores)[::-1]
    m["mask_score"] = float(scores[best_idx])
    m["mask_score_margin"] = float(ordered[0] - ordered[1]) if ordered.size >= 2 else None

    # --- Box ----------------------------------------------------------------
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, dtype=np.float64).ravel()[:4])
    raw_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    cx1, cx2 = min(max(x1, 0.0), w), min(max(x2, 0.0), w)
    cy1, cy2 = min(max(y1, 0.0), h), min(max(y2, 0.0), h)
    clamped_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    m["box_area_frac"] = clamped_area / (w * h)
    m["box_aspect"] = (y2 - y1) / (x2 - x1) if x2 > x1 else None
    m["box_out_of_image"] = 1.0 - clamped_area / raw_area if raw_area > 0 else None
    bx1, bx2 = int(math.floor(cx1)), int(math.ceil(cx2))
    by1, by2 = int(math.floor(cy1)), int(math.ceil(cy2))
    box_ok = bx2 > bx1 and by2 > by1

    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    ix, iy = int(center_x), int(center_y)
    m["center_on_mask"] = bool(mask[iy, ix]) if 0 <= ix < w and 0 <= iy < h else False
    check()

    # --- Mask shape ---------------------------------------------------------
    rows = np.flatnonzero(mask.any(axis=1))
    mask_area = 0
    if rows.size:
        cols = np.flatnonzero(mask.any(axis=0))
        my1, my2, mx1, mx2 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        crop = mask[my1:my2, mx1:mx2]
        mask_area = int(np.count_nonzero(crop))
        n, _, stats, _ = cv2.connectedComponentsWithStats(crop.astype(np.uint8), connectivity=8)
        m["mask_components"] = int(n - 1)
        m["mask_largest_frac"] = float(stats[1:, cv2.CC_STAT_AREA].max()) / mask_area
        check()

        stride = max(1, int(math.ceil(math.sqrt(mask_area / _PCA_MAX_POINTS))))
        ys, xs = np.nonzero(crop[::stride, ::stride])
        if ys.size >= 3:
            cov = np.cov(np.stack([xs.astype(np.float64), ys.astype(np.float64)]))
            evals, evecs = np.linalg.eigh(cov)
            vx, vy = evecs[:, 1]
            m["mask_angle_deg"] = math.degrees(math.atan2(abs(vx), abs(vy)))
            m["mask_elongation"] = math.sqrt(evals[1] / evals[0]) if evals[0] > 1e-9 else None
    else:
        m["mask_components"] = 0
    check()

    if box_ok and raw_area > 0:
        m["mask_box_fill"] = mask_area / raw_area
        inside = int(np.count_nonzero(mask[by1:by2, bx1:bx2]))
        m["mask_outside_box"] = (1.0 - inside / mask_area) if mask_area else None
        edge = [mask[by1, bx1:bx2], mask[by2 - 1, bx1:bx2],
                mask[by1 + 1:by2 - 1, bx1], mask[by1 + 1:by2 - 1, bx2 - 1]]
        total = sum(e.size for e in edge)
        m["mask_touches_box_edge"] = sum(int(np.count_nonzero(e)) for e in edge) / total if total else None

        # --- Photo, inside the box ------------------------------------------
        photo = original[by1:by2, bx1:bx2]
        scale = _PHOTO_SIDE / max(photo.shape[:2])
        if scale < 1:
            photo = cv2.resize(photo, (max(1, round(photo.shape[1] * scale)), max(1, round(photo.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(np.ascontiguousarray(photo), cv2.COLOR_RGB2GRAY)
        m["box_luma_mean"] = float(gray.mean())
        m["box_luma_std"] = float(gray.std())
        m["box_sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    check()

    # --- Stretch and paste --------------------------------------------------
    if "clipped_bottom" in paste_info:
        m["clipped_bottom"] = paste_info["clipped_bottom"]
        m["clipped_side"] = paste_info["clipped_side"]

    region = np.zeros((h, w), dtype=np.uint8)
    paste = paste_info.get("paste")
    if paste is not None:
        py1, py2, px1, px2, feather, alpha = paste
        pasted = (np.asarray(alpha) > 0).astype(np.uint8)
        region[py1:py2, px1:px2] = pasted

        kernel = np.ones((3, 3), np.uint8)
        e1 = cv2.erode(pasted, kernel, iterations=1)
        e3 = cv2.erode(e1, kernel, iterations=2)
        e6 = cv2.erode(e3, kernel, iterations=3)
        border = (pasted > 0) & (e1 == 0)
        border[:feather] = False  # the top rows are feathered; only the hard edge is a seam
        band = (e3 > 0) & (e6 == 0)
        # Gradient of the output around the pasted region (1 px context each side).
        gy1, gy2, gx1, gx2 = max(py1 - 1, 0), min(py2 + 1, h), max(px1 - 1, 0), min(px2 + 1, w)
        gray = cv2.cvtColor(np.ascontiguousarray(enhanced[gy1:gy2, gx1:gx2]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        grad = grad[py1 - gy1:py1 - gy1 + (py2 - py1), px1 - gx1:px1 - gx1 + (px2 - px1)]
        if border.any() and band.any():
            inner = float(grad[band].mean())
            m["seam_score"] = float(grad[border].mean()) / inner if inner > 1e-6 else None
        check()

        pad = _OUTSIDE_MARGIN_PX
        ry1, ry2, rx1, rx2 = max(py1 - pad, 0), min(py2 + pad, h), max(px1 - pad, 0), min(px2 + pad, w)
        region[ry1:ry2, rx1:rx2] = cv2.dilate(region[ry1:ry2, rx1:rx2],
                                              np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    outside = (region == 0).astype(np.uint8)
    if outside.any():
        diff = cv2.absdiff(np.ascontiguousarray(original), np.ascontiguousarray(enhanced))
        m["outside_change"] = float(np.mean(cv2.mean(diff, mask=outside)[:3]))
    check()

    return _sanitise(m)


def compute_with_budget(budget_s=BUDGET_S, **kwargs):
    """compute() in a worker thread with a hard wall-clock budget. Never raises."""
    deadline = time.perf_counter() + budget_s
    result = {}

    def run():
        try:
            result["value"] = compute(deadline=deadline, **kwargs)
        except BaseException:
            result["value"] = None

    thread = threading.Thread(target=run, name="quality-metrics", daemon=True)
    thread.start()
    thread.join(budget_s)
    if thread.is_alive():
        return None  # it stops itself at its next check
    return result.get("value")
