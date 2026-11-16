# Quality metrics (metrics_version 1, worker_version `sam-v5-q1`)

Every successful job returns `output.quality` next to `mask_score`: a flat
object of rounded scalars. The image is unchanged (proved by
`tests/test_golden.py` against an unmodified copy of the v5 handler).

- **Privacy:** only the scalars below. No image, mask, crop, embedding, coordinate
  list or free text is returned, stored or logged. `quality.py` contains no
  print or logging calls (a test enforces this).
- **Safety:** metrics run after the PNG is encoded, on read-only arrays, in a
  thread with a 2 second budget. Any error or timeout gives `quality: null` and
  the job succeeds as normal.
- **Format:** floats are rounded to 3 decimals. A value that does not apply, or
  is not finite, is `null`. Keys are always present and always in this order.
- **Versioning:** if any definition below changes, bump `METRICS_VERSION` and
  `WORKER_VERSION` in `quality.py`. Compare rows only within one `metrics_version`.

Terms: *box* is the app's `boundingBox` `[x1, y1, x2, y2]` in pixels of the image
**after** EXIF rotation. *Mask* is SAM's best mask (highest score). *Paste* is the
stretched patch `enhance_penis` blends back into the image.

## What each metric points to

| Key | Type | Definition | Points to |
| --- | --- | --- | --- |
| `worker_version` | string | Worker build constant | (bookkeeping) |
| `metrics_version` | int | Version of these definitions | (bookkeeping) |
| `mask_score` | float 0-1 | SAM's predicted IoU for the chosen mask | SAM cutout |
| `mask_score_margin` | float | Best score minus second best. Small means SAM was unsure which object was meant | SAM cutout, or an ambiguous box |
| `mask_box_fill` | float | Mask area / box area. Low means the box is far bigger than the object | User error (loose box) |
| `mask_outside_box` | float 0-1 | Fraction of the mask outside the box. High means SAM leaked into surroundings | SAM cutout |
| `mask_touches_box_edge` | float 0-1 | Fraction of box perimeter pixels covered by the mask. High means the box cuts the object off | User error (tight box) |
| `mask_components` | int | Connected pieces in the mask (8-connected) | SAM cutout (fragmented) |
| `mask_largest_frac` | float 0-1 | Largest piece / mask area | SAM cutout |
| `mask_angle_deg` | float 0-90 | Principal-axis angle of the mask by PCA. 0 = vertical, 90 = horizontal | Algorithm limit: the stretch is always vertical |
| `mask_elongation` | float >= 1 | sqrt(major / minor PCA eigenvalue). Near 1 means a blob, not an elongated shape | SAM cutout, or a wrong object |
| `center_on_mask` | bool | Whether the box centre (SAM's positive point prompt) lies on the mask | User error or box mapping |
| `box_area_frac` | float 0-1 | Box area inside the image / image area | User error (subject too small or too large) |
| `box_aspect` | float | Box height / width | Box mapping (a transposed box looks wide) |
| `box_out_of_image` | float 0-1 | Fraction of the box lying outside the image. Should be 0 | **App bug** (box mapping) |
| `box_luma_mean` | float 0-255 | Mean luminance in the box | User error (dark or blown-out photo) |
| `box_luma_std` | float | Luminance standard deviation in the box | User error (flat, low-contrast photo) |
| `box_sharpness` | float | Variance of the Laplacian in the box, measured at 512 px on the long side | User error (blur) |
| `exif_orientation` | int 1-8 or null | EXIF orientation tag seen by the worker | App bug (orientation) |
| `exif_rotated` | bool | Whether `exif_transpose` changed the orientation (tag 2-8) | App bug (orientation): the app's box must be in rotated coordinates |
| `image_w`, `image_h` | int | Image size after EXIF rotation | App bug (orientation, resizing) |
| `clipped_bottom` | bool | Stretched patch ran past the bottom edge and was cut off | Algorithm limit (clipping), or framing |
| `clipped_side` | bool | Stretched patch ran past the left or right edge and was cut off | Algorithm limit (clipping), or framing |
| `seam_score` | float | Mean gradient magnitude on the paste's hard (unfeathered) border / mean gradient in a band 3-6 px inside it. Around 1 is invisible; much higher is a visible edge | Algorithm limit (seams) |
| `outside_change` | float 0-255 | Mean absolute pixel change outside the paste region dilated by 5 px. Must be 0 | Worker bug if not 0 |

Notes:
- `seam_score` is relative: natural object edges also raise it. Compare it
  across jobs, not against a fixed threshold, until it is calibrated.
- `mask_box_fill` can exceed 1 if the mask extends beyond the box; read it with
  `mask_outside_box`.
- `box_out_of_image` and `center_on_mask` are the main box-mapping signals.
  The `simulate` function rejects coordinates over 20000, but not boxes larger
  than the image, so a mapping bug shows up here.

## Starting triage rules (to calibrate, not ground truth)

These are first-cut thresholds for sorting low ratings. Fit them against the
clinician-rated set before acting on them.

| Bucket | Signal |
| --- | --- |
| App bug: box mapping | `box_out_of_image > 0.02`, or `center_on_mask = false` with `mask_score > 0.9` |
| App bug: orientation | low ratings concentrated in `exif_rotated = true` compared with `false` |
| User error: box | `mask_box_fill < 0.25` (box too loose) or `mask_touches_box_edge > 0.3` (box too tight) |
| User error: photo | `box_luma_mean < 40` or `> 225`, `box_luma_std < 12`, or low `box_sharpness` (bottom decile) |
| Algorithm: SAM cutout | `mask_score < 0.85`, `mask_score_margin < 0.03`, `mask_components > 1` with `mask_largest_frac < 0.9`, or `mask_outside_box > 0.2` |
| Algorithm: fixed vertical stretch | `mask_angle_deg > 25` |
| Algorithm: clipping | `clipped_bottom` or `clipped_side` |
| Algorithm: seams | `seam_score` in the top decile |
| Worker bug | `outside_change > 0` |
