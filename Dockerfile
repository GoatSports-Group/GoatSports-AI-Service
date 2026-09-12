FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-vision.txt .
RUN pip install --no-cache-dir -r requirements-vision.txt

COPY app ./app
COPY scripts/download_vision_models.py ./scripts/download_vision_models.py
COPY scripts/warmup_ocr_models.py ./scripts/warmup_ocr_models.py
RUN python scripts/download_vision_models.py
RUN python scripts/warmup_ocr_models.py

EXPOSE 7081

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7081"]
