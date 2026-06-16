import os
import sys
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Ensure the parent directory is in the sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_mask, create_mask_image
from setup.train_config import MODELS_DIR

app = FastAPI(title="U-Net Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = os.path.join(MODELS_DIR, 'unet_oxford_pets.h5')

@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    temp_img_path = f"temp_{file.filename}"
    with open(temp_img_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # Predict the mask
        mask = predict_mask(temp_img_path, WEIGHTS_PATH)
        mask_img = create_mask_image(mask)
        
        # Save to memory and return
        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
    finally:
        # Cleanup
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
