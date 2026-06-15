import tensorflow as tf
import subprocess

print("="*50)
print("GPU ENVIRONMENT CHECK")
print("="*50)
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {gpu.name}")
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.free,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        print(f"Details: {result.stdout.strip()}")
    except:
        print("nvidia-smi not found in PATH")

    with tf.device('/GPU:0'):
        a = tf.random.normal([2000, 2000])
        b = tf.matmul(a, a)
    print(f"✅ GPU test passed - shape: {b.shape}")
    print("✅ Ready for U-Net training")
else:
    print("⚠️  No GPU detected")
    print("Install CUDA 11.8 + cuDNN 8.6 for RTX 2050")
    print("See setup/gpu_setup_guide.md")
