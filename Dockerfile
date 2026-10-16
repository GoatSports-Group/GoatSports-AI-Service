FROM python:3.12-slim AS demand-forecast-model

WORKDIR /export

COPY requirements-forecast-export.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install --no-cache-dir -r requirements-forecast-export.txt

COPY app/infrastructure/__init__.py ./app/infrastructure/__init__.py
COPY app/infrastructure/forecast ./app/infrastructure/forecast
COPY scripts/export_chronos_bolt.py ./scripts/export_chronos_bolt.py
RUN python scripts/export_chronos_bolt.py /export/out

FROM python:3.12-slim AS ocr-onnx-model

WORKDIR /export

RUN apt-get update     && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1     && rm -rf /var/lib/apt/lists/*
COPY requirements-ocr-export.txt .
RUN pip install --no-cache-dir -r requirements-ocr-export.txt
COPY scripts/export_ocr_onnx.py ./scripts/export_ocr_onnx.py
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python scripts/export_ocr_onnx.py /export/out

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FACE_DETECTOR_MODEL_PATH=/app/models/face_detection_yunet_2023mar.onnx \
    FACE_RECOGNITION_MODEL_PATH=/app/models/face_recognition_sface_2021dec.onnx \
    FACE_ATTRIBUTE_MODEL_PATH=/app/models/face_attrib_net.onnx \
    LIVENESS_MODEL_PATH=/app/models/silent_face_anti_spoofing.onnx \
    DEMAND_FORECAST_MODEL_PATH=/app/models/chronos-bolt-tiny.onnx

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
COPY scripts/download_text_embedding_model.py ./scripts/download_text_embedding_model.py
RUN python scripts/download_vision_models.py
RUN python scripts/download_text_embedding_model.py
RUN python scripts/warmup_ocr_models.py
COPY --from=demand-forecast-model /export/out/ /app/models/
# OCR CCCD chạy bằng ONNX Runtime (~1–3 s/ảnh thay vì 45–90 s của Paddle CPU); thiếu file thì tự quay về Paddle.
COPY --from=ocr-onnx-model /export/out/ /app/models/
# Kho tri thức chatbot: đồng bộ vào MongoDB mỗi lần khởi động (file mới / đã sửa thì nạp, file đã xóa thì gỡ).
COPY knowledge ./knowledge

EXPOSE 7081

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7081"]
