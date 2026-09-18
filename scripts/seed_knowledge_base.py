"""Seed the RAG chatbot knowledge base with GoatSports FAQ/rules content.

The chatbot (`RagChatbotService.answer_query`) can only ground answers in
documents that already exist in the `knowledge_documents` collection. Nothing
in this repo ever populates that collection at runtime (see README.md), so
every query falls back to the "khong tim thay tai lieu lien quan" response
until documents are ingested. This script ingests a baseline set of documents
covering the flows most users ask about.

Run once against the target environment:

    python -m scripts.seed_knowledge_base

Re-running is safe: `RagChatbotService.ingest` derives a stable document id
from the title/version and replaces the chunks for that id.
"""

from __future__ import annotations

from app.application.dto.request.models import KnowledgeDocumentRequest
from app.infrastructure.config.container import rag_chatbot_service

DOCUMENTS: list[KnowledgeDocumentRequest] = [
    KnowledgeDocumentRequest(
        title="Quy trình đặt sân và thanh toán cọc",
        content=(
            "Người chơi tìm sân theo môn thể thao, vị trí, khoảng giá và đánh giá, sau đó chọn khung "
            "giờ còn trống để đặt. Khi bấm đặt sân, hệ thống khóa tạm khung giờ đó trong vài phút để "
            "tránh hai người đặt trùng. Người chơi cần thanh toán tiền cọc qua VietQR trong thời gian "
            "khóa; sau khi chủ sân xác nhận đã nhận được tiền, booking mới chuyển sang trạng thái đã "
            "xác nhận và vé đặt sân (kèm mã QR) mới được phát hành. Nếu quá thời gian khóa mà chưa "
            "thanh toán, khung giờ sẽ tự động mở lại cho người khác đặt."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Lịch sử đặt sân và vé QR",
        content=(
            "Người chơi có thể xem lại toàn bộ lịch sử đặt sân của chính mình trong mục 'Lịch sử đặt "
            "sân', gồm trạng thái hiện tại (chờ thanh toán, đã xác nhận, đã hoàn tất, đã hủy) và chi "
            "tiết sân/khung giờ/số tiền. Mỗi booking đã xác nhận có một mã QR riêng, có chữ ký để chống "
            "giả mạo; mã này dùng để check-in tại sân và không thể chỉnh sửa. Người chơi chỉ xem được "
            "booking của chính mình, không xem được booking của người khác."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Check-in tại sân",
        content=(
            "Khi đến sân, người chơi đưa mã QR hoặc mã đặt sân cho chủ sân/nhân viên quét hoặc nhập tay "
            "để check-in. Chỉ chủ sân/nhân viên của đúng venue và đúng khung giờ mới thực hiện check-in "
            "được. Với khách vãng lai chưa đặt trước (walk-in), chủ sân có thể tạo booking tại chỗ và "
            "thu tiền mặt hoặc qua chuyển khoản; hệ thống đảm bảo không thu tiền hai lần cho cùng một "
            "lượt chơi."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Chính sách hủy sân và hoàn tiền",
        content=(
            "Người chơi có thể yêu cầu hủy booking trước giờ chơi theo chính sách hủy của từng sân. Hủy "
            "càng sớm trước giờ chơi thì tỷ lệ hoàn tiền càng cao; hủy sát giờ hoặc không đến (no-show) "
            "có thể không được hoàn cọc. Sau khi yêu cầu hủy được duyệt, tiền sẽ được hoàn qua đúng "
            "phương thức đã thanh toán; nếu hoàn tiền tự động thất bại, yêu cầu sẽ được chuyển sang cho "
            "admin xử lý thủ công và trạng thái booking/thanh toán được đồng bộ khi hoàn tất."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Đánh giá và nhận xét sân",
        content=(
            "Sau khi một lượt đặt sân đã hoàn tất (đã chơi xong), người chơi có thể để lại một đánh giá "
            "kèm số sao và nhận xét cho sân đó; mỗi booking đã hoàn tất chỉ được đánh giá một lần. Điểm "
            "đánh giá trung bình của sân được cập nhật ngay khi có đánh giá mới. Quản trị viên có thể ẩn "
            "hoặc gỡ các đánh giá vi phạm quy định."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="GOAT AI ghép trận (matchmaking) hoạt động như thế nào",
        content=(
            "Tính năng 'AI Ghép trận' giúp tìm đối thủ phù hợp dựa trên môn thể thao, trình độ/ELO hiện "
            "tại, khung giờ rảnh, phong cách chơi (cân bằng, fair-play, cạnh tranh) và khoảng cách tìm "
            "kiếm mà người chơi thiết lập. Sau khi bấm 'Ghép trận', GOAT AI quét những người chơi khác "
            "đang tìm đối thủ, tính điểm phù hợp dựa trên chênh lệch ELO, khung giờ chung, khoảng cách "
            "và phong cách chơi, rồi đề xuất đối thủ có điểm phù hợp cao nhất. Cả hai người chơi đều "
            "phải bấm 'Chấp nhận kèo' thì trận đấu mới được xác nhận; nếu một bên từ chối hoặc hết thời "
            "gian phản hồi, phiên ghép trận sẽ kết thúc an toàn và người chơi có thể tìm lại đối thủ "
            "khác. Sau khi cả hai chấp nhận, hệ thống tiếp tục đề xuất/chọn sân và tiến hành đặt cọc."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Sau khi ghép trận thành công: chọn sân, đặt cọc và trò chuyện",
        content=(
            "Sau khi hai người chơi chấp nhận kèo, trận đấu chuyển sang bước chọn sân rồi đặt cọc; "
            "trong lúc chờ chọn sân/thanh toán, nút 'Xem sân đã đặt' sẽ hiển thị khi đã có sân được "
            "chọn cho trận, và nút 'Mở trò chuyện' sẽ hiển thị khi cuộc trò chuyện giữa hai người chơi "
            "đã được tạo. Người chơi có thể dùng khung chat này để trao đổi lịch chơi, sân bãi trước "
            "khi ra sân. Trận đấu vẫn có thể thao tác (xem sân, trò chuyện, check-in) cho tới khi hoàn "
            "tất; chỉ những phiên xem lại trong mục lịch sử mới ở chế độ chỉ xem."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Kết bạn, nhắn tin và chặn người dùng",
        content=(
            "Người dùng có thể gửi lời mời kết bạn, chấp nhận/từ chối lời mời và trò chuyện 1-1 hoặc "
            "theo nhóm với bạn bè/thành viên câu lạc bộ theo thời gian thực. Tin nhắn hiển thị các "
            "trạng thái đã gửi, đã đến máy và đã xem. Khi chặn một người dùng, hai bên sẽ không thể kết "
            "bạn, nhắn tin hoặc theo dõi nhau nữa cho đến khi bỏ chặn."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Câu lạc bộ (CLB): tạo, thành viên và lệ phí",
        content=(
            "Người dùng có thể tạo câu lạc bộ thể thao, mời hoặc duyệt thành viên tham gia tùy theo chế "
            "độ riêng tư của CLB (công khai hoặc cần duyệt). Quản trị viên CLB có thể tạo hoạt động/buổi "
            "tập, thu lệ phí thành viên và theo dõi lịch sử đóng phí. Thành viên rời CLB hoặc bị xóa "
            "khỏi CLB sẽ tự động rời khỏi nhóm chat của CLB đó."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Tổ chức giải đấu: đăng ký, lịch thi đấu và bảng xếp hạng",
        content=(
            "Người tổ chức (organizer) có thể tạo giải đấu, quy định thể lệ, đặt sân thi đấu và mở đăng "
            "ký cho người chơi/đội. Người chơi đăng ký tham gia, xếp đội hình (lineup) và đóng lệ phí "
            "giải đấu nếu có. Sau khi đủ điều kiện, hệ thống tạo lịch thi đấu (fixture) theo thể thức đã "
            "chọn (loại trực tiếp hoặc vòng tròn) và cập nhật bảng xếp hạng theo kết quả từng trận."
        ),
        sport_type=None,
    ),
    KnowledgeDocumentRequest(
        title="Trợ lý GOAT AI có thể giúp gì",
        content=(
            "Trợ lý GOAT AI (biểu tượng chat ở góc màn hình) trả lời các câu hỏi thường gặp về cách đặt "
            "sân, thanh toán, hủy/hoàn tiền, check-in, ghép trận AI và hoạt động của câu lạc bộ/giải "
            "đấu trên nền tảng GoatSports, đồng thời có thể gợi ý sân phù hợp. Trợ lý chỉ trả lời dựa "
            "trên tài liệu nội bộ của GoatSports; với các câu hỏi ngoài phạm vi trên, người dùng nên "
            "liên hệ đội ngũ hỗ trợ."
        ),
        sport_type=None,
    ),
]


def main() -> None:
    total_chunks = 0
    for document in DOCUMENTS:
        result = rag_chatbot_service.ingest(document)
        total_chunks += result.chunks_indexed
        print(f"Ingested '{document.title}' -> {result.chunks_indexed} chunk(s)")
    print(f"Done. {len(DOCUMENTS)} document(s), {total_chunks} chunk(s) indexed.")


if __name__ == "__main__":
    main()
