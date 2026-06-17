import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.switch_backend('agg')

try:
    from setup.train_config import (
        IMAGE_SIZE, NUM_CLASSES, CHANNELS, UNET_BATCH_SIZE, MODELS_DIR, PREDICTIONS_DIR
    )
except ImportError:
    # Fallback if config can't be imported
    IMAGE_SIZE = 128
    NUM_CLASSES = 3
    CHANNELS = 3
    UNET_BATCH_SIZE = 16
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    PREDICTIONS_DIR = os.path.join(BASE_DIR, 'outputs', 'predictions')

from utils.utils import dice_coef, dice_loss, iou

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (IMAGE_SIZE, IMAGE_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def create_mask_colormap():
    # 0=red pet, 1=green background, 2=blue border
    colors = ['red', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def calculate_iou(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    ious = []
    for cls in range(NUM_CLASSES):
        true_cls = (y_true_f == cls)
        pred_cls = (y_pred_f == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

def visualize_predictions():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    print("Loading Oxford-IIIT Pet Test Dataset...")
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    test_images = dataset['test'].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_images.batch(6) # We need 6 samples
    
    model_path = os.path.join(MODELS_DIR, 'unet_best.h5')
    print(f"Loading model from: {model_path}")
    
    custom_objs = {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou': iou
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
    
    # Get exactly 1 batch of 6 images
    for images, masks in test_dataset.take(1):
        batch_images = images.numpy()
        batch_masks = masks.numpy()
        break
        
    print("Predicting masks...")
    preds = model.predict(batch_images, verbose=0)
    preds_classes = np.argmax(preds, axis=-1)
    true_classes = np.squeeze(batch_masks, axis=-1)
    
    cmap, norm = create_mask_colormap()
    
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    total_iou = 0.0
    
    for i in range(6):
        img = batch_images[i]
        true_mask = true_classes[i]
        pred_mask = preds_classes[i]
        
        sample_iou = calculate_iou(true_mask, pred_mask)
        total_iou += sample_iou
        
        # Panel 1: Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i} Original", fontsize=16)
        axes[i, 0].axis('off')
        
        # Panel 2: True Mask
        axes[i, 1].imshow(true_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[i, 1].set_title("Ground Truth Mask", fontsize=16)
        axes[i, 1].axis('off')
        
        # Panel 3: Predicted Mask
        axes[i, 2].imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[i, 2].set_title(f"Predicted Mask (IoU: {sample_iou:.3f})", fontsize=16)
        axes[i, 2].axis('off')
        
        # Save individual files
        # 1. Original Image
        plt.imsave(os.path.join(PREDICTIONS_DIR, f"sample_{i}_original.png"), img)
        
        # 2. True Mask
        fig_mask_true, ax_mask_true = plt.subplots(figsize=(4, 4))
        ax_mask_true.imshow(true_mask, cmap=cmap, norm=norm, interpolation='nearest')
        ax_mask_true.axis('off')
        fig_mask_true.savefig(os.path.join(PREDICTIONS_DIR, f"sample_{i}_true_mask.png"), bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig_mask_true)
        
        # 3. Predicted Mask
        fig_mask_pred, ax_mask_pred = plt.subplots(figsize=(4, 4))
        ax_mask_pred.imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
        ax_mask_pred.axis('off')
        fig_mask_pred.savefig(os.path.join(PREDICTIONS_DIR, f"sample_{i}_pred_mask.png"), bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig_mask_pred)
        
    plt.tight_layout()
    grid_path = os.path.join(PREDICTIONS_DIR, 'prediction_grid.png')
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 6x3 prediction grid to {grid_path}")
    
    avg_iou = total_iou / 6.0
    print(f"\nAverage IoU across 6 samples: {avg_iou:.4f}")

if __name__ == "__main__":
    visualize_predictions()
