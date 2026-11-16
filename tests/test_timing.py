import time

import numpy as np

import quality
from conftest import ellipse_mask


def test_metrics_under_200ms_on_4000x3000(handler_new):
    h, w = 3000, 4000
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = ellipse_mask(h, w, (2000, 1400), (250, 700))
    box = np.array([1700, 650, 2300, 2150])
    info = {}
    enhanced = handler_new.enhance_penis(image, mask, info=info)
    masks, scores = np.stack([mask, mask, mask]), np.array([0.6, 0.95, 0.8])
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        out = quality.compute_with_budget(original=image, enhanced=enhanced, masks=masks, scores=scores, best_idx=1,
                                          box=box, paste_info=info, exif_orientation=1)
        runs.append(time.perf_counter() - start)
        assert out is not None
    assert max(runs) < 0.2, runs
    assert out["outside_change"] == 0.0
