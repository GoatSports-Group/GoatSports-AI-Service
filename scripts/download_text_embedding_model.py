"""Tải model embedding cho chatbot RAG: multilingual-e5-small (MIT, intfloat), bản ONNX lượng tử hóa của Xenova.

Ghim commit và SHA-256 giống các model vision: đổi model thì phải xem lại giấy phép, chạy lại
scripts/eval_rag.py để chọn ngưỡng relevance_floor mới, rồi cập nhật cả hai giá trị dưới đây.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
BASE_URL = f"https://huggingface.co/Xenova/multilingual-e5-small/resolve/{REVISION}"
FILES = {
    "e5-small-model.onnx": ("onnx/model_quantized.onnx", "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"),
    "e5-small-tokenizer.json": ("tokenizer.json", "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (remote, expected) in FILES.items():
        target = MODEL_DIR / name
        if not target.is_file() or _sha256(target) != expected:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(f"{BASE_URL}/{remote}", target)
        digest = _sha256(target)
        if digest != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {digest}")
        print(f"{name}: sha256={digest}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
