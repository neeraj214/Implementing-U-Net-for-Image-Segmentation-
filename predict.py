import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

# Ensure project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from setup.train_config import IMAGE_SIZE, MODELS_DIR
from models.unet import build_unet

# Default model path — matches what src/train.py saves as 'unet_best.h5'
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, 'unet_best.h5')


def predict_mask(image_path, model_weights_path=None):
    """
    Load an image, run U-Net inference, and return the predicted class mask.

    Args:
        image_path: Path to the input image (any format supported by PIL).
        model_weights_path: Path to a saved .h5 weights file. If not provided
                            or not found, random weights are used.

    Returns:
        predicted_mask: numpy array of shape (H, W) with integer class labels.
    """
    # Load and preprocess image
    original_img = Image.open(image_path).convert("RGB")
    img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img) / 255.0
    img_input = np.expand_dims(img_np, axis=0).astype(np.float32)

    # Initialize model (3 classes: Pet, Background, Border)
    model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), n_classes=3)

    if model_weights_path and os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
        print(f"Loaded weights from: {model_weights_path}")
    else:
        print("Warning: No weights provided or found. Using randomly initialized weights.")

    # Predict
    prediction = model.predict(img_input, verbose=0)
    predicted_mask = np.argmax(prediction, axis=-1)[0]

    return predicted_mask


def create_mask_image(predicted_mask):
    """
    Convert an integer class mask to a color PIL Image.

    Class colors:
        0 — Pet       → Red   (255, 0, 0)
        1 — Background→ Green (0, 255, 0)
        2 — Border    → Blue  (0, 0, 255)
    """
    colors = np.array([
        [255,   0,   0],   # Pet      — red
        [  0, 255,   0],   # Background — green
        [  0,   0, 255],   # Border   — blue
    ], dtype=np.uint8)

    color_mask = colors[predicted_mask]
    return Image.fromarray(color_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Segmentation Mask with U-Net")
    parser.add_argument("--image",   required=True,              help="Path to input image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,    help="Path to model weights (.h5)")
    parser.add_argument("--output",  default="output_mask.png",  help="Path to save the output mask PNG")

    args = parser.parse_args()

    mask = predict_mask(args.image, args.weights)
    mask_img = create_mask_image(mask)
    mask_img.save(args.output)
    print(f"Mask saved to {args.output}")
