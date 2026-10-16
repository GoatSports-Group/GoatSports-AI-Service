"""Export the PP-OCRv5 models used for CCCD OCR to ONNX (+ the recogniser dictionary).

Usage:  python scripts/export_ocr_onnx.py [out_dir=models]

Run it in an environment with requirements-ocr-export.txt (the Dockerfile stage `ocr-onnx-model` does). On Windows,
paddle2onnx 2.1 only loads next to paddlepaddle 3.1.x — use a separate venv, not the ai-service one (3.3.x).
Why: Paddle 3.x on CPU without oneDNN needs 45–90 s per card; the same models on ONNX Runtime take ~1–3 s.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DETECTION = "PP-OCRv5_mobile_det"
RECOGNITION = "latin_PP-OCRv5_mobile_rec"


def model_dir(name: str) -> Path:
    """Download (once) through PaddleOCR and return the Paddle inference directory."""
    target = Path.home() / ".paddlex" / "official_models" / name
    if not (target / "inference.json").is_file():
        from paddleocr import TextDetection, TextRecognition

        (TextDetection if name == DETECTION else TextRecognition)(model_name=name)
    if not (target / "inference.json").is_file():
        raise RuntimeError(f"PaddleOCR did not download {name} to {target}")
    return target


def export(name: str, out_dir: Path) -> None:
    source = model_dir(name)
    target = out_dir / f"{name}.onnx"
    subprocess.run(
        [
            sys.executable, "-m", "paddle2onnx.command",
            "--model_dir", str(source),
            "--model_filename", "inference.json",
            "--params_filename", "inference.pdiparams",
            "--save_file", str(target),
            "--opset_version", "14",
        ],
        check=True,
    )
    print(f"{target}: {target.stat().st_size // 1024} KB")


def export_dictionary(out_dir: Path) -> None:
    import yaml

    config = yaml.safe_load((model_dir(RECOGNITION) / "inference.yml").read_text(encoding="utf-8"))
    characters = [str(character) for character in config["PostProcess"]["character_dict"]]
    target = out_dir / f"{RECOGNITION}.dict.txt"
    target.write_text("\n".join(characters) + "\n", encoding="utf-8", newline="\n")
    print(f"{target}: {len(characters)} characters")


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    export(DETECTION, out_dir)
    export(RECOGNITION, out_dir)
    export_dictionary(out_dir)


if __name__ == "__main__":
    main()
