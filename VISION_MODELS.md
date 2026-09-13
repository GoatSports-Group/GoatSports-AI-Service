# Open-source vision models

The AI service downloads and checksum-verifies these model artifacts while building its Docker image:

- OpenCV Zoo YuNet face detector — MIT license.
- OpenCV Zoo SFace face recognizer — Apache-2.0 license.
- Facenox MiniFASNet V2 SE anti-spoofing model — Apache-2.0 license.
- Qualcomm Facial-Attribute-Detection model — BSD-3-Clause license; the quantized ONNX export
  detects eye openness, eyeglasses, face masks, and sunglasses for realtime capture readiness.
- PaddleOCR `PP-OCRv5_mobile_det` and `latin_PP-OCRv5_mobile_rec` — Apache-2.0 project.

The binaries are intentionally excluded from Git. Run `python scripts/download_vision_models.py`
for local development. Do not replace a model without reviewing its license, evaluating it on a
representative Vietnamese dataset, and updating the pinned SHA-256 digest.

PaddleOCR is warmed up during FastAPI startup, so the service becomes ready only after its OCR
runtime is initialized. oneDNN is disabled by default for this pipeline because the current
Windows Paddle runtime cannot execute the cached OCR graph with oneDNN. Override
`IDENTITY_OCR_ENABLE_MKLDNN` only after validating the target runtime.
