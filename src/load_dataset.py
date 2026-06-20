import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

# Ensure the project root is in the path so we can import setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup.train_config import (
    IMAGE_SIZE, UNET_BATCH_SIZE, PLOTS_DIR, METRICS_DIR, NUM_CLASSES
)

def preprocess(sample: dict) -> tuple:
    """
    Preprocess a raw dataset sample by resizing and normalizing.

    Resizes images and masks to target IMAGE_SIZE, normalizes image pixels
    to [0.0, 1.0], and remaps segmentation mask class labels from 1,2,3 to 0,1,2.

    Args:
        sample (dict): Raw dictionary containing 'image' and 'segmentation_mask' keys.

    Returns:
        tuple: (preprocessed_image, preprocessed_mask) where:
            - preprocessed_image: Float tensor of shape (IMAGE_SIZE, IMAGE_SIZE, 3)
            - preprocessed_mask: Int tensor of shape (IMAGE_SIZE, IMAGE_SIZE, 1)
    """
    image = tf.image.resize(sample['image'], [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.image.resize(
        sample['segmentation_mask'], [IMAGE_SIZE, IMAGE_SIZE],
        method='nearest'
    )
    mask = tf.cast(mask, tf.int32) - 1  # remap 1,2,3 → 0,1,2
    return image, mask

def get_datasets() -> tuple:
    """
    Load the Oxford-IIIT Pet dataset and build TF dataset pipelines.

    Loads training and testing splits, performs resizing/normalization, sets up
    caching and prefetching, and generates sample visualizations.

    Returns:
        tuple: (train_ds, test_ds) as tf.data.Dataset pipelines ready for training.
    """
    print("Loading Oxford-IIIT Pet dataset...")
    dataset, info = tfds.load('oxford_iiit_pet', with_info=True)
    
    train_size = info.splits['train'].num_examples
    test_size = info.splits['test'].num_examples
    
    print(f"Train dataset size: {train_size}")
    print(f"Test dataset size: {test_size}")
    
    # Save dataset info
    os.makedirs(METRICS_DIR, exist_ok=True)
    dataset_info = {
        "train_size": train_size,
        "test_size": test_size,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES
    }
    info_path = os.path.join(METRICS_DIR, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    print(f"Dataset info saved to {info_path}")
    
    # Build tf.data pipeline
    train_ds = dataset['train'].shuffle(1000).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(UNET_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_ds = dataset['test'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(UNET_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Visualize 3 sample images
    visualize_samples(train_ds)
    
    return train_ds, test_ds

def visualize_samples(dataset):
    """Save a sample visualization grid showing original images alongside ground truth masks."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Take one batch and select the first 3 images
    for images, masks in dataset.take(1):
        fig, axes = plt.subplots(3, 2, figsize=(9, 12))
        fig.suptitle("Oxford-IIIT Pet Dataset - Training Samples", fontsize=16, fontweight='bold', y=0.98)
        
        # Colormap: 0=red, 1=green, 2=blue
        cmap = ListedColormap(['red', 'green', 'blue'])
        
        for i in range(3):
            # Original Image
            axes[i, 0].imshow(images[i].numpy())
            axes[i, 0].set_title(f"Sample {i+1}: Original Image", fontsize=12)
            axes[i, 0].axis('off')
            
            # Mask
            im = axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap=cmap, vmin=0, vmax=2)
            axes[i, 1].set_title(f"Sample {i+1}: Ground Truth Mask", fontsize=12)
            axes[i, 1].axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(PLOTS_DIR, 'sample_dataset.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved sample visualization to {plot_path}")

if __name__ == '__main__':
    train_dataset, test_dataset = get_datasets()
