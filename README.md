# RunPod SAM Image Enhancement

Serverless endpoint for image enhancement using Segment Anything Model (SAM).

## Current outage (Sept 2026)

Rolling out new workers with the old image does **not** fix this.

RunPod is assigning **NVIDIA RTX PRO 6000 Blackwell** GPUs (`sm_120`). The previous worker (`runpod/pytorch:2.1.0` + CUDA 11.8) only had kernels for `sm_50`–`sm_90`, so every job failed with:

```
CUDA error: no kernel image is available for execution on the device
```

The worker is now built on **PyTorch 2.8 + CUDA 12.8**, which includes Blackwell kernels.

After GitHub rebuilds the image:

1. Confirm logs say `RUNPOD DIRECT HANDLER v5 — BLACKWELL/CUDA12.8`.
2. In the endpoint, finish/purge the stuck rollout so old CUDA 11.8 workers are gone.
3. Optional while waiting for the rebuild: in **Manage → GPU configuration**, uncheck RTX PRO 6000 Blackwell and keep older cards (A40, L40, RTX 4090, A100).

## Files

- `handler.py` - RunPod serverless handler function
- `quality.py` - per-job quality metrics (scalars only; see `METRICS.md`)
- `Dockerfile` - CUDA 12.8 / PyTorch 2.8 container
- `requirements.txt` - Python dependencies (does **not** pin an old torch)
- `sam_vit_h_4b8939.pth` - SAM model weights (downloaded at image build)

## Deployment

This worker is deployed to RunPod Serverless via GitHub integration.

## Usage

Send POST requests to your RunPod endpoint:

```json
{
  "input": {
    "imageData": "base64_encoded_image",
    "boundingBox": [x1, y1, x2, y2]
  }
}
```

## Quality metrics

Successful jobs also return `quality`, a flat object of rounded scalar metrics
(no image data). It is `null` if metrics fail or exceed their 2 second budget;
the image is never affected. Definitions and versioning: `METRICS.md`.

## Tests (CPU only, no GPU or SAM weights needed)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest -q tests
```

`tests/baseline/handler_v5_514906e.py` is an unmodified copy of the v5 handler.
The golden tests require the returned PNG to be byte-identical to it.
