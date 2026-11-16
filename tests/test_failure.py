"""A metrics failure or timeout returns the image unchanged with quality = null."""
import time

import cv2
import numpy as np

import quality
from conftest import FakePredictor, ellipse_mask, encode, run_job, synthetic_rgb

MASK = lambda h, w: ellipse_mask(h, w, (320, 240), (40, 110))
BOX = (270, 120, 370, 360)


def _image():
    return encode(synthetic_rgb(480, 640, seed=1))


def _baseline(handler_old):
    return run_job(handler_old, _image(), BOX, FakePredictor(MASK))["enhancedImageData"]


def test_metric_raising_gives_null_quality(handler_old, handler_new, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced")
    monkeypatch.setattr(quality.cv2, "connectedComponentsWithStats", boom)
    result = run_job(handler_new, _image(), BOX, FakePredictor(MASK))
    assert result["status"] == "success" and result["quality"] is None
    assert result["enhancedImageData"] == _baseline(handler_old)


def test_metric_writing_to_image_is_blocked(handler_old, handler_new, monkeypatch):
    real = quality._readonly
    def evil(array):
        view = real(array)
        view[...] = 0  # would corrupt the image if the view were writable
        return view
    monkeypatch.setattr(quality, "_readonly", evil)
    result = run_job(handler_new, _image(), BOX, FakePredictor(MASK))
    assert result["quality"] is None
    assert result["enhancedImageData"] == _baseline(handler_old)


def test_slow_metrics_time_out(handler_old, handler_new, monkeypatch):
    monkeypatch.setattr(quality, "BUDGET_S", 0.2)
    monkeypatch.setattr(quality, "compute", lambda **k: time.sleep(1.5) or {"x": 1})
    start = time.perf_counter()
    result = run_job(handler_new, _image(), BOX, FakePredictor(MASK))
    assert time.perf_counter() - start < 1.0
    assert result["quality"] is None
    assert result["enhancedImageData"] == _baseline(handler_old)


def test_budget_wrapper_itself_failing(handler_new, monkeypatch):
    def broken(**k):
        raise MemoryError()
    monkeypatch.setattr(quality, "compute_with_budget", broken)
    result = run_job(handler_new, _image(), BOX, FakePredictor(MASK))
    assert result["status"] == "success" and result["quality"] is None


def test_missing_module_gives_null_quality(handler_old, handler_new, monkeypatch):
    monkeypatch.setattr(handler_new, "_quality", None)
    result = run_job(handler_new, _image(), BOX, FakePredictor(MASK))
    assert result["quality"] is None
    assert result["enhancedImageData"] == _baseline(handler_old)


def test_quality_module_never_prints_or_logs():
    source = open(quality.__file__).read()
    for banned in ("print(", "logging", "logger", "sys.stdout", "sys.stderr", "open("):
        assert banned not in source, banned
