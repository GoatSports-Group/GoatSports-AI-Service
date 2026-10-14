"""multilingual-e5-small (MIT, intfloat) chạy ONNX trên CPU: vector 384 chiều hiểu nghĩa tiếng Việt.

Model e5 cần tiền tố "query: " cho câu hỏi và "passage: " cho đoạn tài liệu, rồi lấy trung bình các token
(mean pooling) và chuẩn hóa. Không có PyTorch lúc chạy: chỉ onnxruntime + tokenizers, giống Chronos-Bolt.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAX_TOKENS = 512
_BATCH = 16
_PAD_ID = 1  # XLM-RoBERTa


class E5OnnxEmbedder:
    model_name = "multilingual-e5-small-q8"
    # Điểm e5 dồn trong khoảng 0,7–0,9; dưới mức này là câu hỏi ngoài kho tri thức (đã đo bằng scripts/eval_rag.py).
    relevance_floor = 0.82

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {item.name for item in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(_MAX_TOKENS)
        self._lock = threading.Lock()  # tokenizer có trạng thái truncation dùng chung

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"query: {text}"])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            vectors.extend(self._embed([f"passage: {text}" for text in texts[start : start + _BATCH]]))
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            encodings = self._tokenizer.encode_batch(texts)
        length = max(len(encoding.ids) for encoding in encodings)
        ids = np.full((len(texts), length), _PAD_ID, dtype=np.int64)
        mask = np.zeros((len(texts), length), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids[row, : len(encoding.ids)] = encoding.ids
            mask[row, : len(encoding.ids)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.astype(float).tolist()


def load_text_embedder(model_path: str, tokenizer_path: str):
    """e5 khi có đủ file model; thiếu thì quay về vector băm từ (vẫn trả lời được, chỉ kém hiểu nghĩa)."""
    from app.application.service.vector_search import HashingEmbedder

    model, tokenizer = Path(model_path), Path(tokenizer_path)
    if model.is_file() and tokenizer.is_file():
        try:
            return E5OnnxEmbedder(model, tokenizer)
        except Exception:  # noqa: BLE001 - model hỏng thì vẫn chạy được bằng phương án dự phòng
            logger.exception("Could not load the e5 embedding model; using hashed word vectors")
    else:
        logger.warning("e5 embedding model not found at %s; using hashed word vectors", model)
    return HashingEmbedder()
