from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
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
FACE_ATTRIBUTE_ARCHIVE_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/"
    "models/face_attrib_net/releases/v0.62.2/face_attrib_net-onnx-w8a8.zip"
)
FACE_ATTRIBUTE_ARCHIVE_SHA256 = "aa708ecb873e19a7416cd19380561094dd5133a0b5d9b941834cfbfb7fdc3d39"
FACE_ATTRIBUTE_FILES = {
    "face_attrib_net.onnx": "05d91643be5ec8bbd1a9ebf770097b2b0dfb6f30534a82a4500b74099393812b",
    "face_attrib_net.data": "52272b2ec061dce9afe8e0fa88732083943341a5aa48f5fdbf02778d31f5b4d5",
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
    install_face_attribute_model()


def install_face_attribute_model() -> None:
    if all(_valid_model(MODEL_DIR / name, digest) for name, digest in FACE_ATTRIBUTE_FILES.items()):
        print("face_attrib_net.onnx and external data are ready.")
        return

    archive_override = os.getenv("FACE_ATTRIBUTE_ARCHIVE", "").strip()
    with tempfile.TemporaryDirectory(prefix="goatsports-face-attribute-") as temp_dir:
        archive = Path(archive_override) if archive_override else Path(temp_dir) / "face-attrib.zip"
        if not archive_override:
            print("Downloading Qualcomm facial attribute model...")
            urllib.request.urlretrieve(FACE_ATTRIBUTE_ARCHIVE_URL, archive)
        if not archive.is_file():
            raise RuntimeError(f"Facial attribute archive does not exist: {archive}")
        if _sha256(archive) != FACE_ATTRIBUTE_ARCHIVE_SHA256:
            raise RuntimeError("Checksum mismatch for facial attribute archive")

        with zipfile.ZipFile(archive) as package:
            entries = {Path(name).name: name for name in package.namelist()}
            for file_name, expected_digest in FACE_ATTRIBUTE_FILES.items():
                entry = entries.get(file_name)
                if entry is None:
                    raise RuntimeError(f"Facial attribute archive is missing {file_name}")
                target = MODEL_DIR / file_name
                temporary_target = Path(temp_dir) / file_name
                with package.open(entry) as source, temporary_target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if _sha256(temporary_target) != expected_digest:
                    raise RuntimeError(f"Checksum mismatch for extracted model: {file_name}")
                shutil.copy2(temporary_target, target)
                print(f"{file_name}: sha256={expected_digest}")


def _valid_model(path: Path, expected_digest: str) -> bool:
    return path.is_file() and _sha256(path) == expected_digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
