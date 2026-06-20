import os
import json
import matplotlib.pyplot as plt

# Ensure matplotlib does not try to open a GUI window
plt.switch_backend('agg')

# It's assumed setup/train_config.py exists and exports METRICS_DIR, PLOTS_DIR.
# If they are not available, we can construct the paths manually.
try:
    from setup.train_config import METRICS_DIR, PLOTS_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    METRICS_DIR = os.path.join(BASE_DIR, 'outputs', 'metrics')
    PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')

def _find_key(history, candidates):
    """Return the first matching key from candidates list, or None."""
    for candidate in candidates:
        if candidate in history:
            return candidate
    # Also do a partial-match search for flexibility
    for candidate in candidates:
        for k in history:
            if candidate in k:
                return k
    return None


def plot_training_curves():
    history_path = os.path.join(METRICS_DIR, 'train_history.json')
    if not os.path.exists(history_path):
        print(f"Error: {history_path} does not exist.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Determine number of epochs
    # Use loss as the base since it's almost always present
    epochs = range(1, len(history.get('loss', [])) + 1)
    if not epochs:
        print("Error: Empty training history.")
        return

    # Keras names MeanIoU metric as 'mean_io_u' or 'mean_io_u_1' (not 'iou')
    iou_key = _find_key(history, ['iou', 'mean_io_u', 'mean_iou'])
    val_iou_key = _find_key(history, ['val_iou', 'val_mean_io_u', 'val_mean_iou'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("U-Net Training Performance Curves", fontsize=20, fontweight='bold', y=0.98)
    
    # [0,0] Train vs Val accuracy
    ax = axes[0, 0]
    if 'accuracy' in history:
        ax.plot(epochs, history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], label='Val Accuracy', color='orange', linewidth=2)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # [0,1] Train vs Val loss
    ax = axes[0, 1]
    if 'loss' in history:
        ax.plot(epochs, history['loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax.set_title('Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # [1,0] Train vs Val IoU
    ax = axes[1, 0]
    if iou_key and iou_key in history:
        ax.plot(epochs, history[iou_key], label='Train IoU', color='blue', linewidth=2)
    if val_iou_key and val_iou_key in history:
        ax.plot(epochs, history[val_iou_key], label='Val IoU', color='orange', linewidth=2)
    ax.set_title('Intersection over Union (IoU)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # [1,1] Learning rate over epochs
    ax = axes[1, 1]
    if 'lr' in history:
        ax.plot(epochs, history['lr'], label='Learning Rate', color='green', linewidth=2)
        ax.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.set_title('Learning Rate (Not Available)', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully at {save_path}")

    # Best epochs
    if 'val_accuracy' in history and history['val_accuracy']:
        best_acc_epoch = history['val_accuracy'].index(max(history['val_accuracy'])) + 1
        print(f"Best Val Accuracy Epoch: {best_acc_epoch} (Value: {max(history['val_accuracy']):.4f})")
    
    if val_iou_key and val_iou_key in history and history[val_iou_key]:
        val_iou_vals = history[val_iou_key]
        best_iou_epoch = val_iou_vals.index(max(val_iou_vals)) + 1
        print(f"Best Val IoU Epoch: {best_iou_epoch} (Value: {max(val_iou_vals):.4f})")

if __name__ == "__main__":
    plot_training_curves()
