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

def preprocess(sample):
    image = tf.image.resize(sample['image'], [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.image.resize(
        sample['segmentation_mask'], [IMAGE_SIZE, IMAGE_SIZE],
        method='nearest'
    )
    mask = tf.cast(mask, tf.int32) - 1  # remap 1,2,3 → 0,1,2
    return image, mask

def load_data():
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Take one batch and select the first 3 images
    for images, masks in dataset.take(1):
        fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        
        # Colormap: 0=red, 1=green, 2=blue
        cmap = ListedColormap(['red', 'green', 'blue'])
        
        for i in range(3):
            # Original Image
            axes[i, 0].imshow(images[i].numpy())
            axes[i, 0].set_title("Image")
            axes[i, 0].axis('off')
            
            # Mask
            axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap=cmap, vmin=0, vmax=2)
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, 'sample_dataset.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved sample visualization to {plot_path}")

if __name__ == '__main__':
    train_dataset, test_dataset = load_data()
