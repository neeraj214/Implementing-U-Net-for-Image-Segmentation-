import tensorflow as tf
import tensorflow_datasets as tfds
import os
from models.unet import build_unet
from utils.utils import dice_loss, dice_coef, iou
from setup.train_config import (
    IMAGE_SIZE, NUM_CLASSES, CHANNELS, UNET_FILTERS, UNET_DROPOUT,
    UNET_LR, UNET_EPOCHS, UNET_BATCH_SIZE, MODELS_DIR
)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (IMAGE_SIZE, IMAGE_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (IMAGE_SIZE, IMAGE_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def train_model():
    print("Loading Oxford-IIIT Pet Dataset...")
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    
    train_length = info.splits['train'].num_examples
    
    train_images = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_images.cache().shuffle(1000).batch(UNET_BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_images.batch(UNET_BATCH_SIZE)
    
    print("Building U-Net Model...")
    model = build_unet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), n_filters=UNET_FILTERS, dropout=UNET_DROPOUT, n_classes=NUM_CLASSES)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=UNET_LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', dice_coef, iou])
                  
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_filepath = os.path.join(MODELS_DIR, 'unet_oxford_pets.h5')
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    print("Starting Training...")
    history = model.fit(train_dataset, epochs=UNET_EPOCHS,
                        validation_data=test_dataset,
                        callbacks=[model_checkpoint])
                        
    print(f"Training completed. Model saved at {checkpoint_filepath}")

if __name__ == '__main__':
    train_model()
