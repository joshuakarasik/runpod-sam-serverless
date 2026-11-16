"""Stub torch, runpod and segment_anything so handler.py imports on CPU, and
provide a fake SAM that returns a fixed synthetic mask. No real photo is used."""
import base64
import contextlib
import importlib.util
import io
import pathlib
import sys
import types

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.no_grad = contextlib.nullcontext
    torch.__version__ = "stub"
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules.setdefault("torch", torch)

    runpod = types.ModuleType("runpod")
    runpod.__version__ = "stub"
    runpod.serverless = types.SimpleNamespace(start=lambda config: None)
    sys.modules.setdefault("runpod", runpod)

    sam = types.ModuleType("segment_anything")
    sam.sam_model_registry = {}
    sam.SamPredictor = object
    sys.modules.setdefault("segment_anything", sam)


_install_stubs()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def handler_new():
    return load_module("handler_new", ROOT / "handler.py")


@pytest.fixture(scope="session")
def handler_old():
    return load_module("handler_baseline", ROOT / "tests" / "baseline" / "handler_v5_514906e.py")


def ellipse_mask(h, w, center, axes, angle=0.0):
    """Filled ellipse. axes = (half-width, half-height) before rotation; angle in degrees."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, 1, -1)
    return m.astype(bool)


class FakePredictor:
    """Stands in for SamPredictor. mask_fn(h, w) -> bool mask for the image it is given."""

    def __init__(self, mask_fn, scores=(0.62, 0.95, 0.81)):
        self.mask_fn = mask_fn
        self.scores = np.array(scores, dtype=np.float32)
        self.shape = None

    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, point_coords, point_labels, box, multimask_output):
        h, w = self.shape
        best = self.mask_fn(h, w)
        others = [np.roll(best, 7, axis=1), np.roll(best, -5, axis=0)]
        order = [others[0], best, others[1]]  # best sits at argmax(scores) == 1
        return np.stack(order), self.scores, None


def synthetic_rgb(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx + yy) % 256)], -1)
    noise = rng.integers(0, 40, size=(h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def encode(array, fmt="PNG", orientation=None):
    img = Image.fromarray(array)
    buf = io.BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        img.save(buf, format="JPEG", quality=92, exif=exif.tobytes())
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def run_job(module, image_b64, box, predictor):
    module.load_sam_model = lambda: predictor
    return module.handler({"id": "test", "input": {"imageData": image_b64, "boundingBox": list(box)}})
