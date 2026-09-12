import os

import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR  # noqa: E402


def main() -> None:
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        cpu_threads=4,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
    )
    ocr.predict(np.full((96, 320, 3), 255, dtype=np.uint8))
    print("PaddleOCR Vietnamese models are ready.")


if __name__ == "__main__":
    main()
