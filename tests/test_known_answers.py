import json

import pytest

import quality
from conftest import FakePredictor, ellipse_mask, encode, run_job, synthetic_rgb


def q(handler_new, h, w, mask_fn, box, **enc):
    result = run_job(handler_new, encode(synthetic_rgb(h, w, seed=3), **enc), box, FakePredictor(mask_fn))
    assert result["status"] == "success"
    assert result["quality"] is not None
    return result["quality"]


def test_vertical_shape_angle_near_zero(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (400, 300), (40, 150)), (350, 140, 450, 460))
    assert m["mask_angle_deg"] < 2
    assert m["mask_elongation"] > 3


def test_rotated_90_angle_near_ninety(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (400, 300), (40, 150), 90), (240, 250, 560, 350))
    assert m["mask_angle_deg"] > 88


def test_tilted_45(handler_new):
    m = q(handler_new, 800, 800, lambda h, w: ellipse_mask(h, w, (400, 400), (40, 200), 45), (250, 250, 550, 550))
    assert 43 < m["mask_angle_deg"] < 47


def test_shape_reaching_bottom_edge_is_clipped(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (400, 520), (50, 120)), (340, 390, 460, 600))
    assert m["clipped_bottom"] is True


def test_centred_shape_is_not_clipped(handler_new):
    m = q(handler_new, 800, 800, lambda h, w: ellipse_mask(h, w, (400, 300), (40, 100)), (350, 190, 450, 410))
    assert m["clipped_bottom"] is False and m["clipped_side"] is False


def test_shape_at_side_edge_is_clipped_sideways(handler_new):
    m = q(handler_new, 700, 700, lambda h, w: ellipse_mask(h, w, (40, 300), (45, 150)), (0, 140, 90, 460))
    assert m["clipped_side"] is True


def test_box_twice_too_big_has_low_fill(handler_new):
    mask_fn = lambda h, w: ellipse_mask(h, w, (400, 300), (40, 120))
    snug = q(handler_new, 600, 800, mask_fn, (360, 180, 440, 420))
    big = q(handler_new, 600, 800, mask_fn, (320, 60, 480, 540))  # twice the width and height
    assert snug["mask_box_fill"] > 0.7
    assert big["mask_box_fill"] < 0.5
    assert snug["mask_touches_box_edge"] > big["mask_touches_box_edge"]
    assert big["box_area_frac"] == pytest.approx(4 * snug["box_area_frac"], rel=1e-3)


def test_scores_margin_components_and_center(handler_new):
    two = lambda h, w: ellipse_mask(h, w, (400, 300), (40, 120)) | ellipse_mask(h, w, (100, 100), (15, 15))
    m = q(handler_new, 600, 800, two, (360, 180, 440, 420))
    assert m["mask_score"] == pytest.approx(0.95, abs=1e-3)
    assert m["mask_score_margin"] == pytest.approx(0.14, abs=1e-3)
    assert m["mask_components"] == 2
    assert 0.9 < m["mask_largest_frac"] < 1
    assert m["center_on_mask"] is True
    assert m["mask_outside_box"] > 0


def test_center_off_mask(handler_new):
    ring = lambda h, w: ellipse_mask(h, w, (400, 300), (80, 120)) & ~ellipse_mask(h, w, (400, 300), (50, 90))
    m = q(handler_new, 600, 800, ring, (320, 180, 480, 420))
    assert m["center_on_mask"] is False


def test_exif_rotation_reported(handler_new):
    mask_fn = lambda h, w: ellipse_mask(h, w, (450, 600), (60, 180))
    m = q(handler_new, 900, 1200, mask_fn, (380, 400, 520, 800), orientation=6)
    assert m["exif_rotated"] is True and m["exif_orientation"] == 6
    assert (m["image_w"], m["image_h"]) == (900, 1200)
    plain = q(handler_new, 900, 1200, mask_fn, (380, 400, 520, 800), orientation=1)
    assert plain["exif_rotated"] is False and (plain["image_w"], plain["image_h"]) == (1200, 900)
    png = q(handler_new, 900, 1200, mask_fn, (380, 400, 520, 800))
    assert png["exif_rotated"] is False and png["exif_orientation"] is None


def test_box_outside_image_is_measured(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (700, 300), (40, 120)), (650, 180, 950, 420))
    assert m["box_out_of_image"] == pytest.approx(0.5, abs=1e-3)


def test_outside_change_is_zero_and_seam_finite(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (400, 260), (40, 120)), (360, 140, 440, 380))
    assert m["outside_change"] == 0.0
    assert m["seam_score"] is not None and m["seam_score"] > 0


def test_photo_metrics_separate_dark_and_blurred(handler_new):
    import cv2
    import numpy as np
    from conftest import FakePredictor as FP
    mask_fn = lambda h, w: ellipse_mask(h, w, (400, 300), (40, 120))
    box = (360, 180, 440, 420)
    base = synthetic_rgb(600, 800, seed=3)
    sharp = run_job(handler_new, encode(base), box, FP(mask_fn))["quality"]
    dark = run_job(handler_new, encode((base // 8).astype(np.uint8)), box, FP(mask_fn))["quality"]
    blur = run_job(handler_new, encode(cv2.GaussianBlur(base, (0, 0), 6)), box, FP(mask_fn))["quality"]
    assert dark["box_luma_mean"] < sharp["box_luma_mean"] / 4
    assert blur["box_sharpness"] < sharp["box_sharpness"] / 10


def test_schema_is_flat_scalar_and_json_safe(handler_new):
    m = q(handler_new, 600, 800, lambda h, w: ellipse_mask(h, w, (400, 300), (40, 120)), (360, 180, 440, 420))
    assert tuple(m) == quality.KEYS
    assert m["worker_version"] == quality.WORKER_VERSION and m["metrics_version"] == quality.METRICS_VERSION
    for key, value in m.items():
        if key == "worker_version":
            assert isinstance(value, str) and len(value) <= 32
        else:
            assert value is None or type(value) in (bool, int, float), key
            if type(value) is float:
                assert value == round(value, 3), key
    assert len(json.dumps(m, allow_nan=False)) < 2048


def test_non_finite_values_become_null():
    assert quality._num(float("nan")) is None and quality._num(float("inf")) is None
    assert quality._num(1.23456) == 1.235
    with pytest.raises(TypeError):
        quality._num([1, 2])
