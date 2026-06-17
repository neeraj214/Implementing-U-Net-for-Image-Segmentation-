FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/unet_best.h5 /app/models/
COPY outputs/metrics/ /app/outputs/metrics/
# Copied utils so main.py can import custom objects (dice_coef, etc.)
COPY utils/ /app/utils/
COPY backend/main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
