import unittest

from app.application.service.vector_search import hybrid_text_scores


class AiAlgorithmTests(unittest.TestCase):
    def test_hybrid_search_understands_normalized_vietnamese_synonyms(self):
        scores = hybrid_text_scores(
            "chỗ đá banh gần đây",
            [
                "Sân bóng đá mini ở khu vực lân cận",
                "Giải đấu quần vợt dành cho người mới",
            ],
        )

        self.assertGreater(scores[0][0], scores[1][0])
        self.assertGreater(scores[0][0], 0.1)


if __name__ == "__main__":
    unittest.main()
