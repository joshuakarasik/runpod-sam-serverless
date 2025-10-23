import runpod
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import torch
import base64
import os
from segment_anything import sam_model_registry, SamPredictor

# --- Constants ---
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sam_{MODEL_TYPE}_4b8939.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize SAM ---
predictor = None
try:
    print(f"Loading SAM model ({MODEL_TYPE}) from {CHECKPOINT_PATH}")
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam_model.to(device=DEVICE).eval()
    predictor = SamPredictor(sam_model)
    print("SAM model loaded successfully.")
except Exception as e:
    print(f"Error loading SAM model: {e}")
    import traceback
    traceback.print_exc()

# --- Enhancement Parameters ---
LENGTH_SCALE = 1.3
WIDTH_SCALE = 1.35
HF_BLEND = 0.15
SMOOTH_BLEND = 0.85
LIGHT_DIR = np.array([0.0, -0.3, 0.95])
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
FEATHER_RATIO = 0.3
FEATHER_EXP = 0.6

# --- Enhancement Function ---
def enhance_penis(img, mask, length_scale=LENGTH_SCALE, width_scale=WIDTH_SCALE):
    mask = mask.astype(np.uint8) * 255
    y_indices, x_indices = np.where(mask > 0)
    if y_indices.size == 0 or x_indices.size == 0:
        print("Warning: Empty mask generated. Returning original image.")
        return img.copy()
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    margin = int(0.1 * max(y_max - y_min, x_max - x_min))
    y_min = max(0, y_min - margin)
    y_max = min(img.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(img.shape[1], x_max + margin)
    
    patch = img[y_min:y_max, x_min:x_max].copy()
    mask_patch = mask[y_min:y_max, x_min:x_max].copy()
    
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        print("Warning: Patch dimensions are zero. Returning original image.")
        return img.copy()
    
    h, w = patch.shape[:2]
    new_h = int(h * length_scale)
    new_w = int(w * width_scale)
    
    patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    offset_x = (new_w - w) // 2
    new_x_min = x_min - offset_x
    patch_x_start = 0
    
    if new_x_min < 0:
        patch_x_start = -new_x_min
        new_x_min = 0
    
    y_end = min(y_min + new_h, img.shape[0])
    x_end = min(new_x_min + new_w - patch_x_start, img.shape[1])
    paste_h = y_end - y_min
    paste_w = x_end - new_x_min
    
    if paste_h <= 0 or paste_w <= 0:
        print("Warning: Paste region invalid. Returning original image.")
        return img.copy()
    
    patch_to_paste = patch_resized[:paste_h, patch_x_start:patch_x_start + paste_w]
    mask_to_paste = mask_resized[:paste_h, patch_x_start:patch_x_start + paste_w]
    
    base_feather = int(FEATHER_RATIO * paste_h)
    for i in range(base_feather):
        alpha = 0.5 * (1 + np.cos(np.pi * (1 - i/base_feather)))
        mask_to_paste[i] = mask_to_paste[i] * alpha
    
    output = img.copy()
    if mask_to_paste.shape == patch_to_paste.shape[:2]:
        output[y_min:y_end, new_x_min:x_end] = \
            output[y_min:y_end, new_x_min:x_end] * (1 - mask_to_paste[..., None]/255.0) + \
            patch_to_paste * (mask_to_paste[..., None]/255.0)
    else:
        print(f"Warning: Dimension mismatch for blending. Returning original image.")
        return img.copy()
        
    return output

# --- RunPod Handler Function ---
def handler(job):
    """RunPod serverless handler function"""
    try:
        # Get input from job
        job_input = job["input"]
        image_data = job_input.get("imageData")
        bounding_box = job_input.get("boundingBox")
        
        if not image_data or not bounding_box:
            return {"error": "Missing imageData or boundingBox"}
        
        if predictor is None:
            return {"error": "SAM model not loaded. Server is not ready."}
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_pil = ImageOps.exif_transpose(image_pil)
        image_np = np.array(image_pil, dtype=np.uint8)
        
        print(f"image_np shape: {image_np.shape}, dtype: {image_np.dtype}, is_contiguous: {image_np.flags['C_CONTIGUOUS']}")
        
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image_np.shape}. Expected (H, W, 3)")
        if not image_np.flags['C_CONTIGUOUS']:
            image_np = np.ascontiguousarray(image_np)
        
        input_box_np = np.array(bounding_box)
        
        x1, y1, x2, y2 = input_box_np
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])
        
        with torch.no_grad():
            predictor.set_image(image_np)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=input_box_np,
                multimask_output=True,
            )
        
        if len(masks) == 0:
            return {"error": "SAM did not generate a mask. Try adjusting the box."}
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        enhanced_image_np = enhance_penis(image_np, mask)
        enhanced_image_pil = Image.fromarray(enhanced_image_np)
        buf = io.BytesIO()
        enhanced_image_pil.save(buf, format="PNG")
        encoded_enhanced_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {
            'status': 'success',
            'enhancedImageData': encoded_enhanced_image_data,
            'mask_score': float(scores[best_idx])
        }
        
    except Exception as e:
        print(f"Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Internal server error: {str(e)}"}
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

