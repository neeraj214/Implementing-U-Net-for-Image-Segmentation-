import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup.train_config import IMAGE_SIZE, CHANNELS, NUM_CLASSES, UNET_FILTERS, UNET_DROPOUT, PLOTS_DIR

def unet_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    """
    Standard U-Net Conv2D block with Batch Normalization and optional Dropout.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the Conv2D layers.
        dropout (float): Dropout rate. Defaults to 0.0 (no dropout).

    Returns:
        tf.Tensor: Output activation tensor.
    """
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x

def build_unet(input_shape: tuple = (128, 128, 3), num_classes: int = 3, filters: int = 64, dropout: float = 0.3) -> tf.keras.Model:
    """
    Build a standard U-Net Architecture for Image Segmentation.

    It consists of an Encoder (contracting path), a Bottleneck, and a
    Decoder (expansive path) with skip connections.

    Args:
        input_shape (tuple): Dimension of the input image (H, W, C). Defaults to (128, 128, 3).
        num_classes (int): Number of target semantic classes. Defaults to 3.
        filters (int): Base filter size. Doubles at each encoding level. Defaults to 64.
        dropout (float): Dropout probability for deeper layers. Defaults to 0.3.

    Returns:
        tf.keras.Model: Compiled or uncompiled Keras Model instance.
    """
    inputs = Input(shape=input_shape)
    
    # ENCODER
    c1 = unet_block(inputs, filters, dropout=0)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = unet_block(p1, filters * 2, dropout=0)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = unet_block(p2, filters * 4, dropout=0)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = unet_block(p3, filters * 8, dropout=dropout)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # BOTTLENECK
    bn = unet_block(p4, filters * 16, dropout=dropout)
    
    # DECODER
    u1 = UpSampling2D((2, 2))(bn)
    u1 = Concatenate()([u1, c4])
    d1 = unet_block(u1, filters * 8, dropout=dropout)
    
    u2 = UpSampling2D((2, 2))(d1)
    u2 = Concatenate()([u2, c3])
    d2 = unet_block(u2, filters * 4, dropout=0)
    
    u3 = UpSampling2D((2, 2))(d2)
    u3 = Concatenate()([u3, c2])
    d3 = unet_block(u3, filters * 2, dropout=0)
    
    u4 = UpSampling2D((2, 2))(d3)
    u4 = Concatenate()([u4, c1])
    d4 = unet_block(u4, filters, dropout=0)
    
    # OUTPUT
    outputs = Conv2D(num_classes, 1, activation='softmax')(d4)
    
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        # Ensure y_true has the correct shape for processing
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

if __name__ == '__main__':
    # Build model and print summary
    model = build_unet(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        num_classes=NUM_CLASSES,
        filters=UNET_FILTERS,
        dropout=UNET_DROPOUT
    )
    model.summary()
    
    # Save model architecture plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'unet_architecture.png')
    
    try:
        tf.keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        print(f"✅ Saved model architecture plot to {plot_path}")
    except ImportError as e:
        print(f"⚠️ Could not plot model. Ensure pydot and graphviz are installed. Error: {e}")
