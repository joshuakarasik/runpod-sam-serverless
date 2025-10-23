# RunPod SAM Image Enhancement

Serverless endpoint for image enhancement using Segment Anything Model (SAM).

## Files

- `handler.py` - RunPod serverless handler function
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `sam_vit_h_4b8939.pth` - SAM model weights

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
