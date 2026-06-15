import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
from PIL import Image

# Import config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from setup.train_config import (
    IMAGE_SIZE, SAMPLES_DIR, PLOTS_DIR, RANDOM_SEED
)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Step 1: Download sample image
os.makedirs(SAMPLES_DIR, exist_ok=True)
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_path = os.path.join(SAMPLES_DIR, "demo_image.jpg")
urllib.request.urlretrieve(img_url, img_path)
print(f"✅ Downloaded sample image to {img_path}")

# Step 2: Preprocess image
original_img = np.array(Image.open(img_path).convert("RGB"))
img = Image.fromarray(original_img).resize((IMAGE_SIZE, IMAGE_SIZE))
img_np = np.array(img) / 255.0
img_input = np.expand_dims(img_np, axis=0)

# Step 3: Build minimal U-Net
def unet_block(x, filters):
    c = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    c = Conv2D(filters, (3, 3), activation="relu", padding="same")(c)
    return c

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Encoder
c1 = unet_block(inputs, 32)
p1 = MaxPooling2D((2, 2))(c1)
c2 = unet_block(p1, 64)
p2 = MaxPooling2D((2, 2))(c2)

# Bottleneck
c3 = unet_block(p2, 128)

# Decoder
u4 = UpSampling2D((2, 2))(c3)
u4 = Concatenate()([u4, c2])
c4 = unet_block(u4, 64)
u5 = UpSampling2D((2, 2))(c4)
u5 = Concatenate()([u5, c1])
c5 = unet_block(u5, 32)

# Output
outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()

# Step 4: Run forward pass
prediction = model.predict(img_input, verbose=0)

# Step 5: Plot figure
os.makedirs(PLOTS_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(img_np)
axes[1].set_title("Preprocessed (128x128)")
axes[1].axis("off")

axes[2].imshow(prediction[0, :, :, 0], cmap="gray")
axes[2].set_title("Raw Predicted Mask (Random Weights)")
axes[2].axis("off")

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "single_image_demo.png")
plt.savefig(plot_path)
print(f"✅ Plot saved to {plot_path}")

print(f"✅ Input shape: {img_input.shape}")
print(f"✅ Output shape: {prediction.shape}")
