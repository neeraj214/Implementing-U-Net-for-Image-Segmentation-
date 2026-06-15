import tensorflow as tf
import os

# ── GPU SETUP ──────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEVICE = '/GPU:0'
    print(f"✅ Using GPU: {gpus[0].name}")
else:
    DEVICE = '/CPU:0'
    print("⚠️  Using CPU - training will be slow")

# ── PATHS ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
METRICS_DIR     = os.path.join(BASE_DIR, 'outputs', 'metrics')
PLOTS_DIR       = os.path.join(BASE_DIR, 'outputs', 'plots')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'outputs', 'predictions')
SAMPLES_DIR     = os.path.join(BASE_DIR, 'data', 'samples')

# ── DATASET ────────────────────────────────────────────
IMAGE_SIZE   = 128          # resize all images to 128x128
NUM_CLASSES  = 3            # Oxford Pet: pet, background, border
CHANNELS     = 3

# ── U-NET HYPERPARAMETERS ──────────────────────────────
UNET_FILTERS      = 64     # base filters (doubles each level)
UNET_DROPOUT      = 0.3
UNET_LR           = 0.001
UNET_EPOCHS       = 30     # increase to 50 if time allows
UNET_BATCH_SIZE   = 16     # reduce to 8 if GPU OOM (RTX 2050 4GB)
UNET_PATIENCE     = 7
VAL_SPLIT         = 0.1
RANDOM_SEED       = 42

# ── CLASS LABELS ───────────────────────────────────────
# Oxford Pet masks: 1=pet, 2=background, 3=border
# Remapped to: 0=pet, 1=background, 2=border
CLASS_NAMES = ['Pet', 'Background', 'Border']

tf.random.set_seed(RANDOM_SEED)
