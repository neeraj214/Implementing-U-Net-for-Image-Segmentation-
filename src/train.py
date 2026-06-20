import os
import sys
import json
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup.train_config import (
    IMAGE_SIZE, CHANNELS, NUM_CLASSES, UNET_FILTERS, UNET_DROPOUT,
    UNET_LR, UNET_PATIENCE, UNET_EPOCHS, MODELS_DIR, METRICS_DIR, DEVICE
)
from src.unet_model import build_unet, MeanIoU
from src.load_dataset import get_datasets

def main():
    print("Loading datasets...")
    train_ds, test_ds = get_datasets()
    
    print(f"Building U-Net on {DEVICE}...")
    with tf.device(DEVICE):
        model = build_unet(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            num_classes=NUM_CLASSES,
            filters=UNET_FILTERS,
            dropout=UNET_DROPOUT
        )
        
        model.compile(
            optimizer=Adam(learning_rate=UNET_LR),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy', MeanIoU(num_classes=NUM_CLASSES)]
        )
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        
        best_model_path = os.path.join(MODELS_DIR, 'unet_best.h5')
        
        if os.path.exists(best_model_path):
            print(f"Found existing model weights at {best_model_path}.")
            print("Loading weights to continue training...")
            try:
                model.load_weights(best_model_path)
                print("Successfully loaded weights.")
            except Exception as e:
                print(f"Warning: Failed to load weights ({e}). Training will start from scratch.")
        
        callbacks = [
            EarlyStopping(patience=UNET_PATIENCE, restore_best_weights=True),
            ModelCheckpoint(best_model_path, save_best_only=True),
            ReduceLROnPlateau(patience=3, factor=0.5)
        ]
        
        print("Starting training...")
        start_time = time.time()
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=UNET_EPOCHS,
            callbacks=callbacks
        )
        training_time = time.time() - start_time
        
        # Save history
        history_path = os.path.join(METRICS_DIR, 'train_history.json')
        # Convert values to float for JSON serialization
        history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
            
        # Save final metadata
        final_epoch = len(history.epoch) - 1
        
        # Keras names MeanIoU metric as 'mean_io_u' or 'mean_io_u_1'
        def _find_history_key(h, candidates):
            for c in candidates:
                if c in h:
                    return c
            for c in candidates:
                for k in h:
                    if c in k and 'val' not in k:
                        return k
            return None

        def _find_val_history_key(h, candidates):
            for c in candidates:
                if c in h:
                    return c
            for c in candidates:
                for k in h:
                    if c in k:
                        return k
            return None

        train_iou_key = _find_history_key(history.history, ['iou', 'mean_io_u', 'mean_iou'])
        val_iou_key = _find_val_history_key(history.history, ['val_iou', 'val_mean_io_u', 'val_mean_iou'])
        
        meta_data = {
            "final_accuracy": float(history.history['accuracy'][final_epoch]),
            "final_iou": float(history.history[train_iou_key][final_epoch]) if train_iou_key else 0.0,
            "val_accuracy": float(history.history['val_accuracy'][final_epoch]),
            "val_iou": float(history.history[val_iou_key][final_epoch]) if val_iou_key else 0.0,
            "epochs_trained": final_epoch + 1,
            "training_time": training_time
        }
        
        meta_path = os.path.join(METRICS_DIR, 'train_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
            
        print("\n--- Training Summary ---")
        print(f"Time: {training_time:.2f}s")
        print(f"Epochs: {meta_data['epochs_trained']}")
        print(f"Final Val Accuracy: {meta_data['val_accuracy']:.4f}")
        print(f"Final Val IoU: {meta_data['val_iou']:.4f}")
        print(f"Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()
