"""PP-OCRv5 text detection + recognition on ONNX Runtime.

Paddle 3.x on CPU without oneDNN (oneDNN crashes under the PIR executor on Windows) runs the SVTR recogniser at
~2–4 s per text line, so one ID card took 45–90 s. The same models exported to ONNX (scripts/export_ocr_onnx.py)
run in well under a second. Pre/post-processing mirrors PaddleOCR: DetResizeForTest + DBPostProcess,
RecResizeImg + CTCLabelDecode, top-to-bottom / left-to-right line order.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

from app.application.exception import CapabilityUnavailableException
from app.infrastructure.config.settings import Settings
from app.infrastructure.vision.open_source_identity import (
    PaddleOcrCitizenIdAdapter,
    _decode_image,
    _vision_modules,
)

DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)
DB_THRESH = 0.3
DB_BOX_THRESH = 0.6
DB_UNCLIP_RATIO = 1.5
DB_MAX_CANDIDATES = 1000
DB_MIN_SIZE = 3
REC_HEIGHT = 48
REC_MIN_WIDTH = 320
REC_BATCH = 8


def onnx_ocr_available(settings: Settings) -> bool:
    return all(
        Path(path).is_file()
        for path in (
            settings.IDENTITY_OCR_ONNX_DET_MODEL_PATH,
            settings.IDENTITY_OCR_ONNX_REC_MODEL_PATH,
            settings.IDENTITY_OCR_ONNX_DICT_PATH,
        )
    )


class OnnxTextReader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._det = None
        self._rec = None
        self._characters: list[str] = []
        self._lock = threading.Lock()

    def read(self, image) -> tuple[list[str], list[float]]:
        """Return recognised lines in reading order with their confidences."""
        self._load()
        boxes = sorted_boxes(self._detect(image))
        crops = [rotate_crop(image, box) for box in boxes]
        results = self._recognize(crops)
        lines = [text for text, _ in results if text.strip()]
        confidences = [score for text, score in results if text.strip()]
        return lines, confidences

    def warm_up(self) -> None:
        _, np = _vision_modules()
        self.read(np.full((96, 320, 3), 255, dtype=np.uint8))

    def _load(self) -> None:
        if self._det is not None:
            return
        with self._lock:
            if self._det is not None:
                return
            try:
                import onnxruntime
            except ImportError as exception:
                raise CapabilityUnavailableException(
                    "ONNX Runtime is not installed. Install requirements-vision.txt."
                ) from exception
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = max(1, self.settings.IDENTITY_OCR_CPU_THREADS)
            # Model xuất từ paddle2onnx còn vài initializer thừa: ORT cảnh báo từng cái lúc nạp, không ảnh hưởng.
            options.log_severity_level = 3
            providers = ["CPUExecutionProvider"]
            self._characters = (
                ["blank"]
                + Path(self.settings.IDENTITY_OCR_ONNX_DICT_PATH).read_text(encoding="utf-8").splitlines()
                + [" "]
            )
            self._rec = onnxruntime.InferenceSession(
                self.settings.IDENTITY_OCR_ONNX_REC_MODEL_PATH, options, providers=providers
            )
            self._det = onnxruntime.InferenceSession(
                self.settings.IDENTITY_OCR_ONNX_DET_MODEL_PATH, options, providers=providers
            )

    def _detect(self, image) -> list:
        cv2, np = _vision_modules()
        height, width = image.shape[:2]
        limit = self.settings.IDENTITY_OCR_MAX_SIDE
        ratio = limit / max(height, width) if max(height, width) > limit else 1.0
        resized_h = max(32, int(round(height * ratio / 32) * 32))
        resized_w = max(32, int(round(width * ratio / 32) * 32))
        resized = cv2.resize(image, (resized_w, resized_h)).astype(np.float32) / 255.0
        normalized = (resized - np.array(DET_MEAN, np.float32)) / np.array(DET_STD, np.float32)
        tensor = normalized.transpose(2, 0, 1)[None, ...]
        probability = self._det.run(None, {self._det.get_inputs()[0].name: tensor})[0][0, 0]
        return db_boxes(probability, width / resized_w, height / resized_h, width, height)

    def _recognize(self, crops: list) -> list[tuple[str, float]]:
        cv2, np = _vision_modules()
        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        # Gom ảnh có tỉ lệ gần nhau vào một lô để phần đệm (padding) ít nhất, như PaddleOCR.
        order = sorted(range(len(crops)), key=lambda index: crops[index].shape[1] / max(1, crops[index].shape[0]))
        input_name = self._rec.get_inputs()[0].name
        for start in range(0, len(order), REC_BATCH):
            batch = order[start:start + REC_BATCH]
            max_ratio = max(
                REC_MIN_WIDTH / REC_HEIGHT,
                *(crops[index].shape[1] / max(1, crops[index].shape[0]) for index in batch),
            )
            target_width = int(REC_HEIGHT * max_ratio)
            tensor = np.zeros((len(batch), 3, REC_HEIGHT, target_width), dtype=np.float32)
            for row, index in enumerate(batch):
                crop = crops[index]
                width = min(target_width, int(math.ceil(REC_HEIGHT * crop.shape[1] / max(1, crop.shape[0]))))
                resized = cv2.resize(crop, (max(1, width), REC_HEIGHT)).astype(np.float32)
                tensor[row, :, :, :resized.shape[1]] = ((resized / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
            output = self._rec.run(None, {input_name: tensor})[0]
            for row, index in enumerate(batch):
                results[index] = ctc_decode(output[row], self._characters)
        return results


def ctc_decode(probabilities, characters: list[str]) -> tuple[str, float]:
    """Greedy CTC: argmax per step, merge repeats, drop the blank (index 0)."""
    _, np = _vision_modules()
    indices = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    text: list[str] = []
    kept: list[float] = []
    previous = -1
    for index, score in zip(indices.tolist(), scores.tolist()):
        if index != previous and index != 0 and index < len(characters):
            text.append(characters[index])
            kept.append(score)
        previous = index
    return "".join(text), float(np.mean(kept)) if kept else 0.0


def db_boxes(probability, scale_x: float, scale_y: float, width: int, height: int) -> list:
    """DBPostProcess: threshold the probability map, one min-area box per region, expand, map back."""
    cv2, np = _vision_modules()
    bitmap = (probability > DB_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours[:DB_MAX_CANDIDATES]:
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < DB_MIN_SIZE:
            continue
        if box_score(probability, mini_box(cv2.boxPoints(rect))) < DB_BOX_THRESH:
            continue
        (center_x, center_y), (rect_w, rect_h), angle = rect
        # Unclip: lề d = diện tích × tỉ lệ / chu vi (như pyclipper); với hình chữ nhật, khung bao nhỏ nhất của
        # đa giác đã nới chính là hình chữ nhật dài thêm 2d mỗi chiều.
        distance = rect_w * rect_h * DB_UNCLIP_RATIO / (2 * (rect_w + rect_h))
        expanded = ((center_x, center_y), (rect_w + 2 * distance, rect_h + 2 * distance), angle)
        if min(expanded[1]) < DB_MIN_SIZE + 2:
            continue
        box = mini_box(cv2.boxPoints(expanded))
        box[:, 0] = np.clip(np.round(box[:, 0] * scale_x), 0, width)
        box[:, 1] = np.clip(np.round(box[:, 1] * scale_y), 0, height)
        boxes.append(box.astype(np.float32))
    return boxes


def mini_box(points):
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left."""
    _, np = _vision_modules()
    points = sorted(points.tolist(), key=lambda point: point[0])
    left = sorted(points[:2], key=lambda point: point[1])
    right = sorted(points[2:], key=lambda point: point[1])
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def box_score(probability, box) -> float:
    """Mean probability inside the box polygon (DB "fast" score)."""
    cv2, np = _vision_modules()
    height, width = probability.shape
    x_min = int(np.clip(np.floor(box[:, 0].min()), 0, width - 1))
    x_max = int(np.clip(np.ceil(box[:, 0].max()), 0, width - 1))
    y_min = int(np.clip(np.floor(box[:, 1].min()), 0, height - 1))
    y_max = int(np.clip(np.ceil(box[:, 1].max()), 0, height - 1))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = box.copy()
    shifted[:, 0] -= x_min
    shifted[:, 1] -= y_min
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(probability[y_min:y_max + 1, x_min:x_max + 1], mask)[0])


def sorted_boxes(boxes: list) -> list:
    """Top-to-bottom, then left-to-right for boxes on the same line (within 10 px), as PaddleOCR."""
    ordered = sorted(boxes, key=lambda box: (box[0][1], box[0][0]))
    for index in range(len(ordered) - 1):
        for position in range(index, -1, -1):
            upper, lower = ordered[position], ordered[position + 1]
            if abs(lower[0][1] - upper[0][1]) < 10 and lower[0][0] < upper[0][0]:
                ordered[position], ordered[position + 1] = lower, upper
            else:
                break
    return ordered


def rotate_crop(image, box):
    """Perspective-crop a (possibly rotated) text box; tall crops are rotated to horizontal."""
    cv2, np = _vision_modules()
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(box.astype(np.float32), target)
    crop = cv2.warpPerspective(
        image, matrix, (max(1, width), max(1, height)), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = np.rot90(crop)
    return crop


class OnnxCitizenIdOcrAdapter(PaddleOcrCitizenIdAdapter):
    """Same CCCD parsing (QR first, then OCR fields); only the text reader runs on ONNX Runtime."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._reader = OnnxTextReader(settings)

    def warm_up(self) -> None:
        self._reader.warm_up()

    def close(self) -> None:
        """Nothing to release (no worker process, unlike the Paddle adapter)."""

    def _read(self, content: bytes) -> tuple[list[str], list[float]]:
        lines, confidences = self._reader.read(self._resize_for_ocr(_decode_image(content)))
        if not lines:
            raise ValueError("Không nhận diện được chữ trên CCCD.")
        return lines, confidences
