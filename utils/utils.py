import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient metric for binary or multi-class segmentation.

    For multi-class predictions (softmax output), computes the mean Dice
    coefficient across all classes using one-hot encoding.

    Args:
        y_true: Ground truth tensor. Shape (B, H, W) or (B, H, W, 1) with
                integer class labels, or (B, H, W, C) one-hot encoded.
        y_pred: Predicted probabilities. Shape (B, H, W, C) for multi-class
                or (B, H, W, 1) for binary segmentation.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice coefficient averaged over classes.
    """
    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.shape(y_pred)[-1]

    # Flatten spatial dims, keep batch and class dims
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    # Convert sparse labels to one-hot if needed
    if len(y_true.shape) <= 3 or (len(y_true.shape) == 4 and y_true.shape[-1] == 1):
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_f = tf.one_hot(y_true_flat, num_classes, dtype=tf.float32)
    else:
        y_true_f = tf.reshape(y_true, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice_per_class)


def dice_loss(y_true, y_pred):
    """
    Dice Loss = 1 - Dice Coefficient.

    Can be used as the training loss for segmentation tasks.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def iou(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU / Jaccard) metric.

    Handles multi-class predictions by converting to one-hot encoding
    and computing per-class IoU, then averaging.

    Args:
        y_true: Ground truth tensor. Shape (B, H, W) or (B, H, W, 1).
        y_pred: Predicted probabilities. Shape (B, H, W, C).
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar mean IoU averaged over classes.
    """
    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.shape(y_pred)[-1]

    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    if len(y_true.shape) <= 3 or (len(y_true.shape) == 4 and y_true.shape[-1] == 1):
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_f = tf.one_hot(y_true_flat, num_classes, dtype=tf.float32)
    else:
        y_true_f = tf.reshape(y_true, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = (tf.reduce_sum(y_true_f, axis=0)
             + tf.reduce_sum(y_pred_f, axis=0)
             - intersection)
    iou_per_class = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou_per_class)
