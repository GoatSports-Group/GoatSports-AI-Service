import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.domain.entity.models import IdentityDocumentData
from app.domain.enums import IdentityDocumentType
from app.infrastructure.vision.open_source_identity import PaddleOcrCitizenIdAdapter
from app.infrastructure.vision.process_isolated_ocr import (
    ProcessIsolatedPaddleOcrCitizenIdAdapter,
)


class PaddleOcrCitizenIdAdapterTest(unittest.TestCase):
    def test_extracts_gender_and_nationality_from_the_same_ocr_line(self) -> None:
        lines = ["Giới tính / Sex: Nam Quốc tịch / Nationality: Việt Nam"]

        gender = PaddleOcrCitizenIdAdapter._gender_after(lines)
        nationality = PaddleOcrCitizenIdAdapter._field_after(
            lines,
            ("QUOC TICH", "NATIONALITY"),
        )

        self.assertEqual("Nam", gender)
        self.assertEqual("Việt Nam", nationality)

    def test_normalizes_gender_when_ocr_corrupts_the_following_label(self) -> None:
        lines = ["Giới tính / Sex: Nam Quc tch / Nationality: Vit Nam"]

        gender = PaddleOcrCitizenIdAdapter._gender_after(lines)

        self.assertEqual("Nam", gender)

    def test_warmup_uses_mobile_models_and_disables_mkldnn(self) -> None:
        predictor = Mock()
        paddle_ocr = Mock(return_value=predictor)
        settings = SimpleNamespace(
            IDENTITY_OCR_TEXT_DETECTION_MODEL="PP-OCRv5_mobile_det",
            IDENTITY_OCR_TEXT_RECOGNITION_MODEL="latin_PP-OCRv5_mobile_rec",
            IDENTITY_OCR_ENABLE_MKLDNN=False,
            IDENTITY_OCR_CPU_THREADS=4,
            IDENTITY_OCR_RECOGNITION_BATCH_SIZE=8,
            IDENTITY_OCR_MAX_SIDE=960,
        )
        numpy = SimpleNamespace(uint8="uint8", full=Mock(return_value="warmup-image"))

        with (
            patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=paddle_ocr)}),
            patch(
                "app.infrastructure.vision.open_source_identity._vision_modules",
                return_value=(None, numpy),
            ),
        ):
            PaddleOcrCitizenIdAdapter(settings).warm_up()

        paddle_ocr.assert_called_once_with(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            cpu_threads=4,
            text_recognition_batch_size=8,
            text_det_limit_side_len=960,
            text_det_limit_type="max",
        )
        predictor.predict.assert_called_once_with("warmup-image")

    def test_reads_identity_data_from_standard_citizen_id_qr_payload(self) -> None:
        document = PaddleOcrCitizenIdAdapter._document_from_qr_payload(
            "079203000001|123456789|NGUYEN VAN A|01012003|Nam|Nha Trang|01012023"
        )

        self.assertIsNotNone(document)
        self.assertEqual("079203000001", document.identity_number)
        self.assertEqual("NGUYEN VAN A", document.full_name)
        self.assertEqual(date(2003, 1, 1), document.date_of_birth)
        self.assertEqual("Nam", document.gender)

    def test_infers_supported_document_from_a_valid_identity_number(self) -> None:
        adapter = PaddleOcrCitizenIdAdapter(SimpleNamespace())

        result = adapter._front_result(
            ["CAN CU0C C0NG DAN", "So: 079203000001", "Ho va ten: NGUYEN VAN A"],
            [0.93, 0.95, 0.9],
        )

        self.assertEqual(IdentityDocumentType.CITIZEN_ID_CHIP, result.data.document_type)
        self.assertTrue(result.expected_side)

    def test_process_isolated_adapter_maps_worker_result(self) -> None:
        document = IdentityDocumentData(
            document_type=IdentityDocumentType.CITIZEN_ID_CHIP,
            identity_number="079203000001",
            full_name="NGUYEN VAN A",
        )
        future = Mock()
        future.result.return_value = (
            document.model_dump(mode="json"),
            0.95,
            True,
            ("079203000001",),
        )
        executor = Mock()
        executor.submit.return_value = future
        adapter = ProcessIsolatedPaddleOcrCitizenIdAdapter(
            SimpleNamespace(),
            executor=executor,
        )

        result = adapter.recognize_front(b"identity-image")

        self.assertEqual(document, result.data)
        self.assertEqual(0.95, result.confidence)
        self.assertTrue(result.expected_side)
        executor.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
