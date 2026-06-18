# U-Net Image Segmentation

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Semantic segmentation of the Oxford-IIIT Pet Dataset using a custom U-Net architecture.

[**Live Demo**](#) *(Placeholder)*

---

## 🧠 U-Net Architecture

The model follows the classic U-Net architecture, featuring an encoder to capture context and a symmetric decoder that enables precise localization. Skip connections pass high-resolution features directly from the encoder to the decoder.

```text
Input (256x256x3)
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

**Oxford-IIIT Pet Dataset**
- **Total Images**: 7,349 images
- **Classes**: 
  - 🔴 Pet (Foreground)
  - 🟢 Background
  - 🔵 Border (Boundary between pet and background)

---

## 📈 Results

| Metric | Score |
| ------ | ----- |
| Test Accuracy | ... |
| Mean IoU | ... |
| Pet IoU | ... |
| Background IoU | ... |
| Border IoU | ... |

---

## 🚀 Setup & Installation

### 1. Backend Setup
Create a virtual environment, install dependencies, and run the FastAPI server:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r backend/requirements.txt

# Check GPU availability and train (optional)
python src/check_gpu.py
python src/train.py

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

---

## 🌍 Deployment

- **Backend**: Deployed to Render.
- **Frontend**: Deployed to Vercel.

---

## 👨‍💻 Author

Created by [**neeraj214**](https://github.com/neeraj214)
