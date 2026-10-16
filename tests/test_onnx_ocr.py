import time
import unittest

import cv2
import numpy as np

from app.infrastructure.config.settings import Settings
from app.infrastructure.vision.onnx_ocr import (
    OnnxCitizenIdOcrAdapter,
    ctc_decode,
    db_boxes,
    onnx_ocr_available,
    sorted_boxes,
)


class OnnxOcrPostProcessTest(unittest.TestCase):
    def test_ctc_merges_repeats_and_drops_blank(self):
        characters = ["blank", "a", "b", " "]
        steps = np.array([
            [0.1, 0.8, 0.05, 0.05],  # a
            [0.1, 0.9, 0.0, 0.0],    # a (repeat, merged)
            [0.9, 0.05, 0.05, 0.0],  # blank
            [0.1, 0.7, 0.1, 0.1],    # a again after blank
            [0.0, 0.0, 0.2, 0.8],    # space
            [0.0, 0.0, 1.0, 0.0],    # b
        ])
        text, score = ctc_decode(steps, characters)
        self.assertEqual(text, "aa b")
        self.assertAlmostEqual(score, (0.8 + 0.7 + 0.8 + 1.0) / 4, places=5)

    def test_boxes_are_read_top_to_bottom_then_left_to_right(self):
        def box(x, y):
            return np.array([[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]], dtype=np.float32)

        ordered = sorted_boxes([box(50, 40), box(80, 12), box(5, 10), box(5, 41)])
        self.assertEqual([(int(item[0][0]), int(item[0][1])) for item in ordered], [(5, 10), (80, 12), (5, 41), (50, 40)])

    def test_db_finds_one_expanded_box_per_text_region_and_maps_back(self):
        probability = np.zeros((64, 128), dtype=np.float32)
        probability[20:30, 10:60] = 0.95
        probability[45:50, 80:84] = 0.2  # dưới ngưỡng
        boxes = db_boxes(probability, 2.0, 2.0, 256, 128)
        self.assertEqual(len(boxes), 1)
        x_min, y_min = boxes[0][:, 0].min(), boxes[0][:, 1].min()
        x_max, y_max = boxes[0][:, 0].max(), boxes[0][:, 1].max()
        # Vùng gốc 10..60 × 20..30 (×2 về ảnh gốc) được nới ra mỗi phía.
        self.assertLess(x_min, 20)
        self.assertGreater(x_max, 118)
        self.assertLess(y_min, 40)
        self.assertGreater(y_max, 58)


@unittest.skipUnless(onnx_ocr_available(Settings()), "PP-OCRv5 ONNX models not exported (scripts/export_ocr_onnx.py)")
class OnnxCitizenIdOcrAdapterTest(unittest.TestCase):
    def test_reads_a_phone_sized_card_in_seconds(self):
        image = np.full((3024, 4032, 3), 236, dtype=np.uint8)
        lines = ["CAN CUOC CONG DAN", "So / No: 079203012345", "Ho va ten / Full name:", "NGUYEN VAN AN",
                 "Ngay sinh / Date of birth: 01/01/1990"]
        for index, text in enumerate(lines):
            cv2.putText(image, text, (1300, 400 + index * 240), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (30, 30, 30), 8,
                        cv2.LINE_AA)
        content = cv2.imencode(".jpg", image)[1].tobytes()
        adapter = OnnxCitizenIdOcrAdapter(Settings())
        adapter.warm_up()

        started = time.perf_counter()
        result = adapter.recognize_front(content)
        elapsed = time.perf_counter() - started

        self.assertTrue(result.expected_side)
        self.assertEqual(result.data.identity_number, "079203012345")
        self.assertEqual(result.data.date_of_birth.isoformat(), "1990-01-01")
        # Paddle CPU cần 45–90 s cho cùng ảnh; ngưỡng rộng để máy CI chậm không đỏ oan.
        self.assertLess(elapsed, 15)


if __name__ == "__main__":
    unittest.main()
