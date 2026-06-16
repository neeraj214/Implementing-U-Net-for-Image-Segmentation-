import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse
from setup.train_config import IMAGE_SIZE, MODELS_DIR
from models.unet import build_unet

def predict_mask(image_path, model_weights_path=None):
    # Load and preprocess image
    original_img = Image.open(image_path).convert("RGB")
    img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img) / 255.0
    img_input = np.expand_dims(img_np, axis=0)
    
    # Initialize model
    model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), n_classes=3)
    
    if model_weights_path and os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        print("Warning: No weights provided or found. Using random initialized weights.")
        
    # Predict
    prediction = model.predict(img_input, verbose=0)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    
    return predicted_mask

def create_mask_image(predicted_mask):
    # Map classes to colors
    # 0: Pet (Red), 1: Background (Black), 2: Border (White)
    colors = np.array([
        [255, 0, 0],   # Pet
        [0, 0, 0],     # Background
        [255, 255, 255] # Border
    ])
    
    color_mask = colors[predicted_mask]
    return Image.fromarray(color_mask.astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Segmentation Mask")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default=os.path.join(MODELS_DIR, 'unet_oxford_pets.h5'), help="Path to model weights")
    parser.add_argument("--output", default="output_mask.png", help="Path to save output mask")
    
    args = parser.parse_args()
    
    mask = predict_mask(args.image, args.weights)
    mask_img = create_mask_image(mask)
    mask_img.save(args.output)
    print(f"Mask saved to {args.output}")
