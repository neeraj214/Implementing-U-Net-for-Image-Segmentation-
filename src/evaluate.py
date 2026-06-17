import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from setup.train_config import (
    IMAGE_SIZE, NUM_CLASSES, CHANNELS, UNET_BATCH_SIZE, MODELS_DIR, METRICS_DIR, CLASS_NAMES
)
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

def evaluate_model():
    print("Loading Oxford-IIIT Pet Test Dataset...")
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    test_images = dataset['test'].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_images.batch(UNET_BATCH_SIZE)
    
    model_path = os.path.join(MODELS_DIR, 'unet_best.h5')
    print(f"Loading model from: {model_path}")
    
    # Custom objects required to load model correctly
    custom_objs = {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou': iou
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
    
    print("Evaluating model...")
    results = model.evaluate(test_dataset, return_dict=True)
    
    test_accuracy = results.get('accuracy', 0.0)
    mean_iou = results.get('iou', 0.0)
    
    print("Computing per-class IoU...")
    # Calculate per-class IoU over the test set
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    for images, masks in test_dataset:
        preds = model.predict(images, verbose=0)
        preds_classes = np.argmax(preds, axis=-1)
        true_classes = np.squeeze(masks.numpy(), axis=-1)
        
        preds_f = preds_classes.flatten()
        true_f = true_classes.flatten()
        
        # Accumulate confusion matrix
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                conf_matrix[i, j] += np.sum((true_f == i) & (preds_f == j))
                
    per_class_iou = {}
    for i in range(NUM_CLASSES):
        # IoU = TP / (TP + FP + FN)
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        iou_val = tp / (tp + fp + fn + 1e-7)
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"
        per_class_iou[class_name] = float(iou_val)
        
    print("\n" + "="*40)
    print(f"{'Evaluation Results':^40}")
    print("="*40)
    print(f"{'Test Accuracy':<20} | {test_accuracy:.4f}")
    print(f"{'Mean IoU':<20} | {mean_iou:.4f}")
    print("-" * 40)
    print(f"{'Per-Class IoU':^40}")
    print("-" * 40)
    for cls_name, cls_iou in per_class_iou.items():
        print(f"{cls_name:<20} | {cls_iou:.4f}")
    print("="*40 + "\n")
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    eval_results_path = os.path.join(METRICS_DIR, 'eval_results.json')
    
    eval_data = {
        "test_accuracy": float(test_accuracy),
        "mean_iou": float(mean_iou),
        "per_class_iou": per_class_iou
    }
    
    with open(eval_results_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
        
    print(f"Results saved to {eval_results_path}")

if __name__ == '__main__':
    evaluate_model()
