from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

EMBEDDING_DIMENSIONS = 256
_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_SYNONYMS = {
    "banh": "bóng",
    "đá banh": "bóng đá",
    "football": "bóng đá",
    "soccer": "bóng đá",
    "badminton": "cầu lông",
    "pickle ball": "pickleball",
    "chỗ chơi": "sân",
    "địa điểm": "sân",
    "gần đây": "lân cận",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    for source, target in sorted(_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)
    return " ".join(_WORD_PATTERN.findall(normalized))


def tokenize(value: str) -> list[str]:
    return [token for token in normalize_text(value).split() if len(token) > 1]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or len(first) != len(second):
        return 0.0
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)))


def local_embedding(value: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Create a deterministic word/character n-gram embedding for local deployments."""
    normalized = normalize_text(value)
    features = tokenize(normalized)
    compact = normalized.replace(" ", "_")
    features.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
    vector = [0.0] * dimensions
    for feature, count in Counter(features).items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(item * item for item in vector))
    return [item / norm for item in vector] if norm else vector


def tfidf_scores(query: str, documents: list[str]) -> list[float]:
    query_terms = Counter(tokenize(query))
    document_terms = [Counter(tokenize(document)) for document in documents]
    if not query_terms or not documents:
        return [0.0] * len(documents)

    document_count = len(documents)
    vocabulary = set(query_terms)
    inverse_document_frequency = {
        term: math.log((1 + document_count) / (1 + sum(term in terms for terms in document_terms))) + 1
        for term in vocabulary
    }
    query_vector = {
        term: (1 + math.log(count)) * inverse_document_frequency[term]
        for term, count in query_terms.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))

    scores: list[float] = []
    for terms in document_terms:
        vector = {
            term: (1 + math.log(terms[term])) * inverse_document_frequency[term]
            for term in vocabulary
            if terms[term]
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        dot = sum(query_vector[term] * vector.get(term, 0.0) for term in query_vector)
        scores.append(dot / (query_norm * norm) if query_norm and norm else 0.0)
    return scores


def hybrid_text_scores(
    query: str,
    documents: list[str],
    document_embeddings: list[list[float]] | None = None,
    tfidf_weight: float = 0.4,
) -> list[tuple[float, float, float]]:
    lexical_scores = tfidf_scores(query, documents)
    query_embedding = local_embedding(query)
    embeddings = document_embeddings or [local_embedding(document) for document in documents]
    results = []
    for lexical, embedding in zip(lexical_scores, embeddings):
        semantic = max(0.0, cosine_similarity(query_embedding, embedding))
        combined = lexical * tfidf_weight + semantic * (1.0 - tfidf_weight)
        results.append((combined, lexical, semantic))
    return results


class HashingEmbedder:
    """Phương án dự phòng khi chưa có model e5: vector băm từ và cụm 3 ký tự (khớp chữ, không hiểu nghĩa)."""

    model_name = "hash-ngram-256-v1"
    relevance_floor = 0.08

    def embed_query(self, text: str) -> list[float]:
        return local_embedding(text)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [local_embedding(text) for text in texts]
