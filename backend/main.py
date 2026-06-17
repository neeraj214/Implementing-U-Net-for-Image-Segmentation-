import os
import sys
import json
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add root project dir to python path to import utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.utils import dice_coef, dice_loss, iou

app = FastAPI(title="U-Net Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Pet', 'Background', 'Border']
# Colors in RGB: Red, Green, Blue
CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'unet_best.h5')
METRICS_DIR = os.path.join(BASE_DIR, 'outputs', 'metrics')

model = None

@app.on_event("startup")
async def load_model():
    global model
    custom_objs = {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou': iou
    }
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "unet_best.h5"}

@app.get("/metrics")
async def get_metrics():
    eval_path = os.path.join(METRICS_DIR, 'eval_results.json')
    train_meta_path = os.path.join(METRICS_DIR, 'train_meta.json')
    
    result = {}
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            result['eval_results'] = json.load(f)
            
    if os.path.exists(train_meta_path):
        with open(train_meta_path, 'r') as f:
            result['train_meta'] = json.load(f)
            
    return result

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size # (width, height)
        
        # Resize to 128x128
        image_resized = image.resize((128, 128))
        img_array = np.array(image_resized) / 255.0
        
        # Predict
        input_tensor = np.expand_dims(img_array, axis=0) # (1, 128, 128, 3)
        preds = model.predict(input_tensor, verbose=0)
        pred_mask = np.argmax(preds[0], axis=-1) # (128, 128)
        
        # Create colored mask
        colored_mask = np.zeros((128, 128, 3), dtype=np.uint8)
        class_distribution = {}
        total_pixels = 128 * 128
        
        for i in range(len(CLASS_NAMES)):
            class_mask = (pred_mask == i)
            pixel_count = np.sum(class_mask)
            # percentage of pixels
            class_distribution[CLASS_NAMES[i]] = float((pixel_count / total_pixels) * 100.0)
            
            colored_mask[class_mask] = CLASS_COLORS[i]
            
        # Convert colored mask to base64
        mask_image = Image.fromarray(colored_mask)
        # Resize the mask back to the original image dimensions using Nearest Neighbor to preserve crisp classes
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        
        buffered = io.BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        dominant_class = max(class_distribution, key=class_distribution.get)
        
        return {
            "original_size": [original_size[0], original_size[1]],
            "mask_base64": mask_base64,
            "class_distribution": class_distribution,
            "dominant_class": dominant_class
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
