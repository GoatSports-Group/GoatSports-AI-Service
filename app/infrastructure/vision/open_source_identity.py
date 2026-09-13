from __future__ import annotations

import logging
import re
import threading
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.application.exception import CapabilityUnavailableException
from app.application.port.identity_verification import (
    DocumentRecognitionResult,
    FaceComparisonResult,
    FaceReadinessResult,
    ImageQualityResult,
    LivenessResult,
)
from app.domain.entity.models import IdentityDocumentData
from app.domain.enums import IdentityDocumentType
from app.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


def _vision_modules():
    try:
        import cv2
        import numpy as np
    except ImportError as exception:
        raise CapabilityUnavailableException(
            "OpenCV vision dependencies are not installed. Install requirements-vision.txt."
        ) from exception
    return cv2, np


def _decode_image(content: bytes):
    cv2, np = _vision_modules()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được định dạng ảnh.")
    return image


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.upper())
    return " ".join(
        "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
        .replace("Đ", "D")
        .split()
    )


class OpenCvImageQualityAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess(self, image: bytes) -> ImageQualityResult:
        cv2, np = _vision_modules()
        decoded = _decode_image(image)
        height, width = decoded.shape[:2]
        if min(width, height) < self.settings.IDENTITY_MIN_IMAGE_EDGE:
            return ImageQualityResult(0.0, False, "IMAGE_TOO_SMALL")

        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        glare_ratio = float(np.mean(gray >= 250))
        blur_score = min(1.0, blur / self.settings.IDENTITY_BLUR_TARGET)
        brightness_score = max(0.0, 1.0 - abs(brightness - 132.0) / 132.0)
        glare_score = max(0.0, 1.0 - glare_ratio / self.settings.IDENTITY_MAX_GLARE_RATIO)
        score = round(0.5 * blur_score + 0.3 * brightness_score + 0.2 * glare_score, 4)

        if blur < self.settings.IDENTITY_MIN_BLUR_SCORE:
            return ImageQualityResult(score, False, "IMAGE_BLURRY")
        if brightness < 35 or brightness > 225:
            return ImageQualityResult(score, False, "IMAGE_LIGHTING_INVALID")
        if glare_ratio > self.settings.IDENTITY_MAX_GLARE_RATIO:
            return ImageQualityResult(score, False, "IMAGE_GLARE")
        return ImageQualityResult(score, True)


class PaddleOcrCitizenIdAdapter:
    FRONT_MARKERS = ("CAN CUOC", "CONG DAN", "HO VA TEN", "FULL NAME", "SO DINH DANH")
    FIELD_LABELS = (
        "HO VA TEN",
        "FULL NAME",
        "NGAY SINH",
        "DATE OF BIRTH",
        "GIOI TINH",
        "SEX",
        "QUOC TICH",
        "NATIONALITY",
        "QUE QUAN",
        "PLACE OF ORIGIN",
        "NOI THUONG TRU",
        "NOI CU TRU",
        "PLACE OF RESIDENCE",
        "CO GIA TRI DEN",
        "DATE OF EXPIRY",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ocr = None
        self._lock = threading.Lock()

    def recognize_front(self, image: bytes) -> DocumentRecognitionResult:
        qr_document = self._document_from_qr_image(image)
        if qr_document is not None:
            logger.info("CCCD front recognized from QR code")
            return DocumentRecognitionResult(
                data=qr_document,
                confidence=1.0,
                expected_side=True,
                raw_lines=(),
            )

        lines, confidences = self._read(image)
        return self._front_result(lines, confidences)

    def _document_from_qr_image(self, content: bytes) -> IdentityDocumentData | None:
        cv2, _ = _vision_modules()
        image = _decode_image(content)
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side > 1600:
            scale = 1600 / longest_side
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        detector = cv2.QRCodeDetector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variants = (image, gray, cv2.equalizeHist(gray))
        for variant in variants:
            payload, _, _ = detector.detectAndDecode(variant)
            document = self._document_from_qr_payload(payload)
            if document is not None:
                return document
        return None

    @classmethod
    def _document_from_qr_payload(cls, payload: str) -> IdentityDocumentData | None:
        if not payload:
            return None
        values = [value.strip() for value in payload.split("|")]
        if len(values) < 6:
            return None

        identity_match = re.search(r"(?<!\d)(\d{12})(?!\d)", values[0])
        full_name = values[2][:150].strip()
        if identity_match is None or not full_name:
            return None

        gender = cls._normalize_gender_value(values[4])
        residence = values[5][:500].strip() or None
        return IdentityDocumentData(
            document_type=IdentityDocumentType.CITIZEN_ID_CHIP,
            identity_number=identity_match.group(1),
            full_name=full_name,
            date_of_birth=cls._compact_date(values[3]),
            gender=gender,
            nationality="Việt Nam",
            place_of_residence=residence,
        )

    @staticmethod
    def _compact_date(value: str):
        digits = re.sub(r"\D", "", value)
        if len(digits) != 8:
            return None
        try:
            return datetime.strptime(digits, "%d%m%Y").date()
        except ValueError:
            return None

    @staticmethod
    def _normalize_gender_value(value: str) -> str | None:
        normalized = _normalize(value)
        if re.match(r"^(NAM|MALE)\b", normalized):
            return "Nam"
        if re.match(r"^(NU|FEMALE)\b", normalized):
            return "Nữ"
        cleaned = value.strip()
        return cleaned[:20] or None

    def _front_result(
        self,
        lines: list[str],
        confidences: list[float],
    ) -> DocumentRecognitionResult:
        normalized = [_normalize(line) for line in lines]
        document_type = self._document_type(normalized)
        identity_number = self._identity_number(lines)
        if document_type == IdentityDocumentType.UNKNOWN and identity_number is not None:
            document_type = IdentityDocumentType.CITIZEN_ID_CHIP
        marker_count = self._marker_count(normalized, self.FRONT_MARKERS)
        expected_side = identity_number is not None or (
            document_type != IdentityDocumentType.UNKNOWN and marker_count >= 1
        )
        logger.info(
            "CCCD front classified expected_side=%s markers=%d identity_number_found=%s",
            expected_side,
            marker_count,
            identity_number is not None,
        )
        data = IdentityDocumentData(
            document_type=document_type,
            identity_number=identity_number,
            full_name=self._field_after(lines, ("HO VA TEN", "FULL NAME")),
            date_of_birth=self._date_after(lines, ("NGAY SINH", "DATE OF BIRTH")),
            expiry_date=self._date_after(lines, ("CO GIA TRI DEN", "DATE OF EXPIRY")),
            gender=self._gender_after(lines),
            nationality=self._bounded_field_after(lines, ("QUOC TICH", "NATIONALITY"), 80),
            place_of_origin=self._bounded_field_after(lines, ("QUE QUAN", "PLACE OF ORIGIN"), 500),
            place_of_residence=self._bounded_field_after(
                lines,
                ("NOI THUONG TRU", "NOI CU TRU", "PLACE OF RESIDENCE"),
                500,
            ),
        )
        return DocumentRecognitionResult(
            data=data,
            confidence=self._average(confidences),
            expected_side=expected_side,
            raw_lines=tuple(lines),
        )

    def warm_up(self) -> None:
        """Load the OCR models and execute the runtime before accepting traffic."""
        _, np = _vision_modules()
        image = np.full((96, 320, 3), 255, dtype=np.uint8)
        with self._lock:
            self._initialize_ocr()
            self._predict(image)

    def _read(self, content: bytes) -> tuple[list[str], list[float]]:
        image = self._resize_for_ocr(_decode_image(content))
        with self._lock:
            self._initialize_ocr()
            result = self._predict(image)
        lines: list[str] = []
        confidences: list[float] = []
        self._collect_ocr_values(result, lines, confidences)
        if not lines:
            raise ValueError("Không nhận diện được chữ trên CCCD.")
        return lines, confidences

    def _resize_for_ocr(self, image):
        """Bound all Paddle pipeline work, not only its text detector input."""
        cv2, _ = _vision_modules()
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.settings.IDENTITY_OCR_MAX_SIDE:
            return image

        scale = self.settings.IDENTITY_OCR_MAX_SIDE / longest_side
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        return cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    def _initialize_ocr(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exception:
            raise CapabilityUnavailableException(
                "PaddleOCR is not installed. Install requirements-vision.txt."
            ) from exception
        try:
            self._ocr = PaddleOCR(
                text_detection_model_name=self.settings.IDENTITY_OCR_TEXT_DETECTION_MODEL,
                text_recognition_model_name=self.settings.IDENTITY_OCR_TEXT_RECOGNITION_MODEL,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=self.settings.IDENTITY_OCR_ENABLE_MKLDNN,
                cpu_threads=self.settings.IDENTITY_OCR_CPU_THREADS,
                text_recognition_batch_size=self.settings.IDENTITY_OCR_RECOGNITION_BATCH_SIZE,
                text_det_limit_side_len=self.settings.IDENTITY_OCR_MAX_SIDE,
                text_det_limit_type="max",
            )
        except Exception as exception:
            raise CapabilityUnavailableException(
                "Không thể khởi tạo PaddleOCR. Hãy kiểm tra model và Paddle runtime."
            ) from exception

    def _predict(self, image):
        try:
            return self._ocr.predict(image)
        except AttributeError:
            return self._ocr.ocr(image)
        except NotImplementedError as exception:
            raise CapabilityUnavailableException(
                "PaddleOCR không tương thích với oneDNN hiện tại. "
                "Đặt IDENTITY_OCR_ENABLE_MKLDNN=false rồi khởi động lại ai-service."
            ) from exception

    def _collect_ocr_values(
        self,
        value: Any,
        lines: list[str],
        confidences: list[float],
    ) -> None:
        if hasattr(value, "json"):
            payload = value.json() if callable(value.json) else value.json
            self._collect_ocr_values(payload, lines, confidences)
            return
        if isinstance(value, dict):
            texts = value.get("rec_texts")
            scores = value.get("rec_scores")
            if isinstance(texts, list):
                lines.extend(str(text).strip() for text in texts if str(text).strip())
                if isinstance(scores, list):
                    confidences.extend(float(score) for score in scores)
                return
            for nested in value.values():
                self._collect_ocr_values(nested, lines, confidences)
            return
        if isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], (int, float))
            ):
                lines.append(value[0].strip())
                confidences.append(float(value[1]))
                return
            for nested in value:
                self._collect_ocr_values(nested, lines, confidences)

    @classmethod
    def _document_type(cls, lines: list[str]) -> IdentityDocumentType:
        joined = " ".join(lines)
        if cls._contains_marker(joined, "CAN CUOC CONG DAN") or cls._contains_marker(
            joined,
            "CITIZEN IDENTITY CARD",
        ):
            return IdentityDocumentType.CITIZEN_ID_CHIP
        if cls._contains_marker(joined, "THE CAN CUOC") or cls._contains_marker(
            joined,
            "IDENTITY CARD",
        ):
            return IdentityDocumentType.IDENTITY_CARD_2024
        return IdentityDocumentType.UNKNOWN

    @staticmethod
    def _identity_number(lines: list[str]) -> str | None:
        for line in lines:
            match = re.search(r"(?<!\d)(\d{12})(?!\d)", line.replace(" ", ""))
            if match:
                return match.group(1)
        return None

    @classmethod
    def _marker_count(cls, lines: list[str], markers: tuple[str, ...]) -> int:
        joined = " ".join(lines)
        return sum(cls._contains_marker(joined, marker) for marker in markers)

    @staticmethod
    def _contains_marker(text: str, marker: str) -> bool:
        normalized_text = _normalize(text)
        normalized_marker = _normalize(marker)
        if normalized_marker in normalized_text:
            return True

        words = normalized_text.split()
        marker_words = normalized_marker.split()
        minimum_size = max(1, len(marker_words) - 1)
        maximum_size = min(len(words), len(marker_words) + 1)
        for size in range(minimum_size, maximum_size + 1):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start : start + size])
                if SequenceMatcher(None, candidate, normalized_marker).ratio() >= 0.72:
                    return True
        return False

    @classmethod
    def _field_after(cls, lines: list[str], labels: tuple[str, ...]) -> str | None:
        for index, line in enumerate(lines):
            segments = re.split(r"[:：]", line)
            for segment_index, segment in enumerate(segments):
                normalized_segment = _normalize(segment)
                if not any(label in normalized_segment for label in labels):
                    continue
                if segment_index + 1 < len(segments):
                    value = cls._trim_at_field_label(segments[segment_index + 1])
                    if value:
                        return value
                if index + 1 < len(lines):
                    value = cls._trim_at_field_label(lines[index + 1])
                    if value:
                        return value
        return None

    @classmethod
    def _trim_at_field_label(cls, value: str) -> str:
        cleaned = value.strip(" /:-")
        for position in range(len(cleaned)):
            normalized_tail = _normalize(cleaned[position:])
            if any(normalized_tail.startswith(label) for label in cls.FIELD_LABELS):
                return cleaned[:position].strip(" /:-")
        return cleaned

    @classmethod
    def _bounded_field_after(
        cls,
        lines: list[str],
        labels: tuple[str, ...],
        max_length: int,
    ) -> str | None:
        value = cls._field_after(lines, labels)
        return value if value and len(value) <= max_length else None

    @classmethod
    def _gender_after(cls, lines: list[str]) -> str | None:
        value = cls._field_after(lines, ("GIOI TINH", "SEX"))
        if not value:
            return None
        return cls._normalize_gender_value(value)

    @classmethod
    def _date_after(cls, lines: list[str], labels: tuple[str, ...]):
        value = cls._field_after(lines, labels)
        if not value:
            return None
        match = re.search(r"(\d{2})[/-](\d{2})[/-](\d{4})", value)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(0).replace("-", "/"), "%d/%m/%Y").date()
        except ValueError:
            return None

    @staticmethod
    def _average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0


class OpenCvFaceReadinessAdapter:
    """Fast, stateless checks used while the owner is in front of the camera."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detector = None
        self._attribute_session = None
        self._lock = threading.Lock()

    def analyze(self, frame: bytes) -> FaceReadinessResult:
        cv2, np = _vision_modules()
        image = _decode_image(frame)
        with self._lock:
            self._load_models(cv2)
            height, width = image.shape[:2]
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(image)

            face_count = 0 if faces is None else len(faces)
            if face_count == 0:
                return self._result("FACE_NOT_FOUND", "Đưa khuôn mặt vào giữa khung.", 0)
            if face_count != 1:
                return self._result(
                    "MULTIPLE_FACES", "Chỉ để một người xuất hiện trước camera.", face_count
                )

            face = faces[0]
            x, y, face_width, face_height = (float(value) for value in face[:4])
            normalized_box = (
                round(x / width, 4), round(y / height, 4),
                round(face_width / width, 4), round(face_height / height, 4),
            )
            center_x = (x + face_width / 2) / width
            center_y = (y + face_height / 2) / height
            centered = (
                abs(center_x - 0.5) <= 0.13
                and abs(center_y - 0.47) <= 0.16
                and 0.20 <= face_width / width <= 0.58
                and 0.30 <= face_height / height <= 0.84
            )

            right_eye = (float(face[4]), float(face[5]))
            left_eye = (float(face[6]), float(face[7]))
            nose = (float(face[8]), float(face[9]))
            right_mouth = (float(face[10]), float(face[11]))
            left_mouth = (float(face[12]), float(face[13]))
            eye_mid = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)
            mouth_mid = (
                (right_mouth[0] + left_mouth[0]) / 2,
                (right_mouth[1] + left_mouth[1]) / 2,
            )
            eye_distance = max(1.0, abs(left_eye[0] - right_eye[0]))
            eye_to_mouth = max(1.0, mouth_mid[1] - eye_mid[1])
            yaw_ratio = (nose[0] - eye_mid[0]) / eye_distance
            pitch_ratio = (nose[1] - eye_mid[1]) / eye_to_mouth
            roll_degrees = float(
                np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
            )
            frontal = (
                abs(yaw_ratio) <= 0.14
                and 0.35 <= pitch_ratio <= 0.72
                and abs(roll_degrees) <= 7.5
            )

            face_crop = self._crop_face(image, x, y, face_width, face_height)
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            attributes = self._attributes(cv2, np, face_crop)
            eyes_open = (
                attributes[0] >= self.settings.FACE_ATTRIBUTE_EYE_OPEN_THRESHOLD
                and attributes[1] >= self.settings.FACE_ATTRIBUTE_EYE_OPEN_THRESHOLD
            )
            glasses_free = (
                attributes[2] < self.settings.FACE_ATTRIBUTE_GLASSES_THRESHOLD
                and attributes[4] < self.settings.FACE_ATTRIBUTE_GLASSES_THRESHOLD
            )
            mask_free = attributes[3] < self.settings.FACE_ATTRIBUTE_MASK_THRESHOLD

        common = dict(
            face_count=1,
            brightness=round(brightness, 2),
            centered=centered,
            frontal=frontal,
            glasses_free=glasses_free,
            eyes_open=eyes_open,
            mask_free=mask_free,
            face_box=normalized_box,
            yaw_ratio=round(yaw_ratio, 4),
            pitch_ratio=round(pitch_ratio, 4),
            roll_degrees=round(roll_degrees, 2),
        )
        if not centered:
            return self._result(
                "FACE_NOT_CENTERED",
                "Căn khuôn mặt vào giữa khung oval và giữ đúng khoảng cách.",
                **common,
            )
        if brightness < 55:
            return self._result("FACE_TOO_DARK", "Khuôn mặt chưa đủ sáng.", **common)
        if brightness > 215:
            return self._result(
                "FACE_TOO_BRIGHT", "Ánh sáng đang quá mạnh, hãy tránh vùng bị lóa.", **common
            )
        if not frontal:
            return self._result(
                "FACE_NOT_FRONTAL", "Nhìn thẳng vào camera, không nghiêng hoặc xoay mặt.", **common
            )
        if not glasses_free:
            return self._result(
                "GLASSES_DETECTED", "Vui lòng tháo kính trước khi xác minh.", **common
            )
        if not mask_free:
            return self._result(
                "FACE_COVERED", "Vui lòng bỏ khẩu trang hoặc vật che khuôn mặt.", **common
            )
        if not eyes_open:
            return self._result(
                "EYES_NOT_OPEN", "Mở mắt tự nhiên và nhìn thẳng vào camera.", **common
            )
        return self._result("READY", "Đúng tư thế, hãy giữ yên.", ready=True, **common)

    def warm_up(self) -> None:
        cv2, np = _vision_modules()
        with self._lock:
            self._load_models(cv2)
            input_meta = self._attribute_session.get_inputs()[0]
            input_dtype = np.uint8 if input_meta.type == "tensor(uint8)" else np.float32
            self._attribute_session.run(
                None,
                {
                    input_meta.name: np.zeros((1, 3, 128, 128), dtype=input_dtype)
                },
            )

    def _load_models(self, cv2) -> None:
        if self._detector is not None and self._attribute_session is not None:
            return
        detector_path = Path(self.settings.FACE_DETECTOR_MODEL_PATH)
        attribute_path = Path(self.settings.FACE_ATTRIBUTE_MODEL_PATH)
        if not detector_path.is_file() or not attribute_path.is_file():
            raise CapabilityUnavailableException(
                "Realtime face models are missing. Run scripts/download_vision_models.py."
            )
        try:
            import onnxruntime as ort
        except ImportError as exception:
            raise CapabilityUnavailableException(
                "ONNX Runtime is not installed. Install requirements-vision.txt."
            ) from exception
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320))
        self._attribute_session = ort.InferenceSession(
            str(attribute_path), providers=["CPUExecutionProvider"]
        )

    def _attributes(self, cv2, np, face_crop):
        resized = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_meta = self._attribute_session.get_inputs()[0]
        if input_meta.type == "tensor(uint8)":
            tensor = np.transpose(rgb.astype(np.uint8), (2, 0, 1))[None, ...]
        else:
            tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        output = self._attribute_session.run(
            None, {input_meta.name: tensor}
        )[0]
        probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
        if output.dtype == np.uint8:
            probabilities *= 0.00390625
        if probabilities.size < 5:
            raise CapabilityUnavailableException("Facial attribute model returned an invalid result.")
        return probabilities

    @staticmethod
    def _crop_face(image, x: float, y: float, width: float, height: float):
        image_height, image_width = image.shape[:2]
        size = max(width, height) * 1.25
        center_x = x + width / 2
        center_y = y + height / 2
        left = max(0, round(center_x - size / 2))
        top = max(0, round(center_y - size / 2))
        right = min(image_width, round(center_x + size / 2))
        bottom = min(image_height, round(center_y + size / 2))
        return image[top:bottom, left:right]

    @staticmethod
    def _result(
        reason_code: str,
        message: str,
        face_count: int,
        ready: bool = False,
        **values,
    ) -> FaceReadinessResult:
        return FaceReadinessResult(
            ready=ready,
            reason_code=reason_code,
            message=message,
            face_count=face_count,
            **values,
        )


class OpenCvFaceVerificationAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detector = None
        self._recognizer = None
        self._lock = threading.Lock()

    def contains_face(self, image: bytes) -> bool:
        cv2, _ = _vision_modules()
        decoded = _decode_image(image)
        with self._lock:
            self._load_models(cv2)
            height, width = decoded.shape[:2]
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(decoded)
        return faces is not None and len(faces) > 0

    def compare(self, identity_image: bytes, live_image: bytes) -> FaceComparisonResult:
        cv2, _ = _vision_modules()
        identity = _decode_image(identity_image)
        live = _decode_image(live_image)
        with self._lock:
            self._load_models(cv2)
            identity_feature = self._feature(cv2, identity)
            live_feature = self._feature(cv2, live)
            similarity = float(
                self._recognizer.match(
                    identity_feature,
                    live_feature,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
        return FaceComparisonResult(
            similarity=round(similarity, 4),
            matched=similarity >= self.settings.IDENTITY_FACE_MATCH_THRESHOLD,
        )

    def warm_up(self) -> None:
        cv2, _ = _vision_modules()
        with self._lock:
            self._load_models(cv2)

    def _load_models(self, cv2) -> None:
        if self._detector is not None:
            return
        detector_path = Path(self.settings.FACE_DETECTOR_MODEL_PATH)
        recognizer_path = Path(self.settings.FACE_RECOGNITION_MODEL_PATH)
        if not detector_path.is_file() or not recognizer_path.is_file():
            raise CapabilityUnavailableException(
                "YuNet/SFace model files are missing. Run scripts/download_vision_models.py."
            )
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320))
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def _feature(self, cv2, image):
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) != 1:
            raise ValueError("Ảnh xác minh phải có đúng một khuôn mặt rõ ràng.")
        aligned = self._recognizer.alignCrop(image, faces[0])
        return self._recognizer.feature(aligned)


class OnnxPassiveLivenessAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = None
        self._detector = None
        self._lock = threading.Lock()

    def verify(self, frames: list[bytes]) -> LivenessResult:
        cv2, np = _vision_modules()
        with self._lock:
            self._load_model()
            input_meta = self._session.get_inputs()[0]
            shape = input_meta.shape
            height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else 80
            width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else 80
            scores: list[tuple[float, bytes]] = []
            for frame in frames:
                image = _decode_image(frame)
                face = self._crop_single_face(cv2, image)
                resized = self._letterbox(cv2, np, face, width, height)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                tensor = np.transpose(resized, (2, 0, 1))[None, ...]
                output = np.asarray(
                    self._session.run(None, {input_meta.name: tensor})[0]
                ).reshape(-1)
                probabilities = (
                    output
                    if np.all(output >= 0) and abs(float(np.sum(output)) - 1.0) < 0.01
                    else self._softmax(np, output)
                )
                live_index = min(
                    self.settings.LIVENESS_LIVE_CLASS_INDEX,
                    len(probabilities) - 1,
                )
                scores.append((float(probabilities[live_index]), frame))
        best_score, representative = max(scores, key=lambda item: item[0])
        average_score = sum(score for score, _ in scores) / len(scores)
        return LivenessResult(
            score=round(average_score, 4),
            live=average_score >= self.settings.LIVENESS_THRESHOLD,
            representative_frame=representative if best_score >= self.settings.LIVENESS_THRESHOLD else None,
        )

    def warm_up(self) -> None:
        cv2, _ = _vision_modules()
        with self._lock:
            self._load_model()
            self._load_detector(cv2)

    def _load_model(self) -> None:
        if self._session is not None:
            return
        model_path = Path(self.settings.LIVENESS_MODEL_PATH)
        if not model_path.is_file():
            raise CapabilityUnavailableException(
                "Passive liveness ONNX model is missing. Configure LIVENESS_MODEL_PATH."
            )
        try:
            import onnxruntime
        except ImportError as exception:
            raise CapabilityUnavailableException(
                "ONNX Runtime is not installed. Install requirements-vision.txt."
            ) from exception
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def _crop_single_face(self, cv2, image):
        self._load_detector(cv2)

        image_height, image_width = image.shape[:2]
        self._detector.setInputSize((image_width, image_height))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) != 1:
            raise ValueError("Khung hình camera phải có đúng một khuôn mặt rõ ràng.")

        x, y, width, height = faces[0][:4]
        padding_x = width * 0.2
        padding_y = height * 0.2
        left = max(0, int(x - padding_x))
        top = max(0, int(y - padding_y))
        right = min(image_width, int(x + width + padding_x))
        bottom = min(image_height, int(y + height + padding_y))
        return image[top:bottom, left:right]

    def _load_detector(self, cv2) -> None:
        if self._detector is not None:
            return
        detector_path = Path(self.settings.FACE_DETECTOR_MODEL_PATH)
        if not detector_path.is_file():
            raise CapabilityUnavailableException(
                "YuNet face detector is missing. Run scripts/download_vision_models.py."
            )
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320))

    @staticmethod
    def _letterbox(cv2, np, image, target_width: int, target_height: int):
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas

    @staticmethod
    def _softmax(np, values):
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)
