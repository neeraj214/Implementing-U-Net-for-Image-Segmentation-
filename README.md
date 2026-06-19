# U-Net Image Segmentation

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)

Semantic segmentation of the Oxford-IIIT Pet Dataset using a custom U-Net architecture built with TensorFlow/Keras.

[**Live Demo**](#) *(Placeholder)*

---

## 🧠 U-Net Architecture

The model follows the classic U-Net architecture, featuring an encoder to capture context and a symmetric decoder that enables precise localization. Skip connections pass high-resolution features directly from the encoder to the decoder.

```text
Input (128x128x3)
  │
  ▼
[Enc1] ←──────────────────────────────────────────→ [Dec1]
  │                                                   ▲
  ▼                                                   │
[Enc2] ←─────────────────────────────────────→ [Dec2]
  │                                              ▲
  ▼                                              │
[Enc3] ←────────────────────────────────→ [Dec3]
  │                                         ▲
  ▼                                         │
[Enc4] ←───────────────────────────→ [Dec4]
  │                                    ▲
  ▼                                    │
  └─────────► [Bottleneck] ────────────┘
```

---

## 📊 Dataset

**Oxford-IIIT Pet Dataset** (loaded via `tensorflow-datasets`)
- **Total Images**: ~7,349 images
- **Classes**: 
  - 🔴 Pet (Foreground)  — class 0
  - 🟢 Background        — class 1
  - 🔵 Border (Boundary) — class 2

---

## 📈 Results

| Metric | Score |
| ------ | ----- |
| Test Accuracy | ... |
| Mean IoU | ... |
| Pet IoU | ... |
| Background IoU | ... |
| Border IoU | ... |

> Run `python src/evaluate.py` after training to populate the results table.

---

## 🚀 Setup & Installation

### 1. Backend Setup
Create a virtual environment, install dependencies, and run the FastAPI server:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

# Check GPU availability
python setup/check_gpu.py

# Run full pipeline (single demo → dataset load → train → evaluate → plot → visualize)
.\run_pipeline.ps1        # PowerShell (Windows)

# Or run individual steps
python src/train.py       # Train the model
python src/evaluate.py    # Evaluate on test set
python src/plot_curves.py # Plot training curves
python src/visualize.py   # Visualize predictions

# Start Backend Server
uvicorn backend.main:app --reload
```

### 2. Frontend Setup
Install npm packages and run the Vite dev server:
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The frontend proxies `/api` requests to the FastAPI backend running on port 8000.

---

## 🏗️ Project Structure

```
├── backend/              # FastAPI server
│   ├── main.py           # Segmentation API endpoints
│   ├── requirements.txt  # Backend Python deps
│   └── Dockerfile        # Docker container config
├── frontend/             # React + Vite frontend
│   └── src/
│       ├── components/   # SegmentationViewer, MetricsDashboard
│       └── api/          # API service layer
├── models/               # U-Net model definition (models/unet.py)
├── src/                  # Training / evaluation pipeline
│   ├── unet_model.py     # Full U-Net with MeanIoU metric
│   ├── train.py          # Training loop + callbacks
│   ├── evaluate.py       # Per-class IoU evaluation
│   ├── plot_curves.py    # Training curve plots
│   ├── visualize.py      # Prediction visualizations
│   └── load_dataset.py   # Oxford Pet TFDS loader
├── setup/
│   ├── train_config.py   # Hyperparameters & paths
│   └── check_gpu.py      # GPU environment check
├── utils/
│   └── utils.py          # Dice loss, dice coef, IoU metrics
├── outputs/              # Saved plots, metrics, predictions
├── requirements.txt      # Root Python deps
└── run_pipeline.ps1      # End-to-end pipeline script (PowerShell)
```

---

## 🌍 Deployment

- **Backend**: Deployable to Render (use `backend/Dockerfile`).
- **Frontend**: Deployable to Vercel.

---

## 👨‍💻 Author

Created by [**neeraj214**](https://github.com/neeraj214)
