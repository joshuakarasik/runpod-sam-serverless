"""The returned PNG must be byte-for-byte identical to the v5 worker at 514906e
(tests/baseline/handler_v5_514906e.py, an unmodified copy of handler.py)."""
import base64
import hashlib

import pytest

from conftest import FakePredictor, ellipse_mask, encode, run_job, synthetic_rgb

CASES = {
    # name: (h, w, encoder kwargs, mask_fn, box)
    "centred_vertical_png": (480, 640, {}, lambda h, w: ellipse_mask(h, w, (320, 240), (40, 110)), (270, 120, 370, 360)),
    "exif_rotated_jpeg": (900, 1200, {"orientation": 6},  # becomes 1200 x 900 after exif_transpose
                          lambda h, w: ellipse_mask(h, w, (450, 600), (60, 180)), (380, 400, 520, 800)),
    "reaches_bottom_edge": (600, 800, {}, lambda h, w: ellipse_mask(h, w, (400, 520), (50, 120)), (340, 390, 460, 600)),
    "near_left_edge": (700, 700, {}, lambda h, w: ellipse_mask(h, w, (40, 350), (45, 150)), (0, 190, 90, 510)),
    "tilted_two_blobs": (750, 1000, {}, lambda h, w: ellipse_mask(h, w, (500, 380), (50, 160), 30)
                         | ellipse_mask(h, w, (700, 200), (20, 20)), (380, 180, 740, 560)),
}


def _sha(result):
    return hashlib.sha256(base64.b64decode(result["enhancedImageData"])).hexdigest()


@pytest.mark.parametrize("name", sorted(CASES))
def test_png_identical_to_baseline(name, handler_old, handler_new):
    h, w, enc, mask_fn, box = CASES[name]
    image = encode(synthetic_rgb(h, w, seed=len(name)), **enc)
    old = run_job(handler_old, image, box, FakePredictor(mask_fn))
    new = run_job(handler_new, image, box, FakePredictor(mask_fn))
    assert old["status"] == new["status"] == "success"
    assert _sha(new) == _sha(old)
    assert new["enhancedImageData"] == old["enhancedImageData"]
    assert new["mask_score"] == old["mask_score"]
    assert set(new) - set(old) == {"quality"}, "the only addition is the quality field"
    assert isinstance(new["quality"], dict)


def test_empty_mask_identical_to_baseline(handler_old, handler_new):
    image = encode(synthetic_rgb(300, 400, seed=9))
    empty = lambda h, w: ellipse_mask(h, w, (0, 0), (0, 0)) & False
    old = run_job(handler_old, image, (10, 10, 100, 100), FakePredictor(empty))
    new = run_job(handler_new, image, (10, 10, 100, 100), FakePredictor(empty))
    assert _sha(new) == _sha(old)
    assert new["quality"]["clipped_bottom"] is None and new["quality"]["outside_change"] == 0.0
