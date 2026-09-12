from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
    "silent_face_anti_spoofing.onnx": (
        "https://github.com/facenox/face-antispoof-onnx/releases/download/"
        "v1.0.0/best_model.onnx"
    ),
}
MODEL_SHA256 = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    "silent_face_anti_spoofing.onnx": "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6",
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for file_name, url in MODELS.items():
        target = MODEL_DIR / file_name
        if not target.exists() or target.stat().st_size < 100_000:
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, target)
        if target.stat().st_size < 100_000:
            raise RuntimeError(f"Downloaded model is invalid: {file_name}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != MODEL_SHA256[file_name]:
            raise RuntimeError(f"Checksum mismatch for downloaded model: {file_name}")
        print(f"{file_name}: sha256={digest}")


if __name__ == "__main__":
    main()
