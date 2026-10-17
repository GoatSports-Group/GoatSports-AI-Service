"""Run against disposable MongoDB and Redis instances, never production data."""

from datetime import datetime, timezone
from uuid import uuid4

from app.application.dto.request.models import (
    KnowledgeDocumentRequest,
    PostCandidate,
    SemanticDocumentRequest,
)
from app.application.service.activity_ingest_service import ActivityIngestService
from app.application.service.post_ranking_service import PostRankingService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.semantic_search_service import SemanticSearchService
from app.application.service.service_config import RagConfig, VectorSearchConfig
from app.application.service.vector_search import HashingEmbedder
from app.infrastructure.config.database import database_clients
from app.infrastructure.persistence.repository import MongoAiDocumentRepository


def run() -> None:
    database_clients.connect()
    repository = MongoAiDocumentRepository(database_clients.mongo)
    vector_config = VectorSearchConfig(enabled=False, index_name="embedding_vector_index")

    search = SemanticSearchService(repository, vector_config)
    football_venue_id = uuid4()
    search.index(
        SemanticDocumentRequest(
            document_type="VENUE",
            source_id=football_venue_id,
            content="Sân bóng đá mini lân cận, mặt cỏ nhân tạo và mở cửa buổi tối.",
        )
    )
    search.index(
        SemanticDocumentRequest(
            document_type="POST",
            source_id=uuid4(),
            content="Kỹ thuật giao cầu lông cho người mới.",
        )
    )
    semantic_results = search.search("chỗ đá banh gần đây", max_results=5)
    assert semantic_results and semantic_results[0].source_id == football_venue_id

    rag = RagChatbotService(repository, HashingEmbedder(), RagConfig(max_context_chunks=5))
    rag.ingest(
        KnowledgeDocumentRequest(
            title="Luật cầu lông cơ bản",
            content=(
                "Một trận cầu lông đánh đơn thi đấu theo thể thức ba hiệp thắng hai. "
                "Mỗi hiệp thông thường kết thúc ở 21 điểm và phải hơn đối thủ hai điểm."
            ),
            sport_type="BADMINTON",
            source_url="https://goatsports.local/knowledge/badminton-basic",
        )
    )
    retrieved = rag.retrieve("Một hiệp cầu lông bao nhiêu điểm?", "BADMINTON")
    assert retrieved and "21" in retrieved[0].chunk.content

    # Hành vi đi qua đúng đường thật (sự kiện -> ingest -> Mongo) rồi xếp hạng bằng dữ liệu đó.
    current_user, similar_user = uuid4(), uuid4()
    shared_post, suggested_post, other_post = uuid4(), uuid4(), uuid4()
    ingest = ActivityIngestService(repository)
    for user_id, post_id in ((current_user, shared_post), (similar_user, shared_post), (similar_user, suggested_post)):
        ingest.handle({"eventType": "activity.post.reaction.v1", "userId": str(user_id), "payload": {
            "postId": str(post_id), "authorId": str(uuid4()), "reaction": "LIKE", "active": True}})
    now = datetime.now(timezone.utc)
    scores = {item.post_id: item.score for item in PostRankingService(repository, snapshot_seconds=0).rank(
        current_user,
        [PostCandidate(post_id=post_id, author_id=uuid4(), published_at=now) for post_id in (other_post, suggested_post)],
    )}
    assert scores[suggested_post] > scores[other_post]
    for user_id in (current_user, similar_user):
        repository.delete_user_interactions(user_id)

    database_clients.close()
    print("AI_ALGORITHM_INTEGRATION_SMOKE_OK")


if __name__ == "__main__":
    run()
