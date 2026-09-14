# GoatSports AI Service

FastAPI microservice cung cấp bảy năng lực AI của GoatSports. MongoDB lưu tài liệu,
embedding, hành vi và kết quả dự báo; Redis quản lý hàng chờ ghép trận có TTL.

## Thuật toán đang dùng

1. **Ghép trận:** lọc cứng theo môn, ngày và phần thời gian giao nhau; sau đó tính
   `55% ELO + 30% khoảng cách Haversine + 15% độ phủ thời gian`. Hai bên phải chấp
   thuận trước khi tạo conversation và đề xuất sân/người đặt cọc.
2. **Gợi ý sân:** cosine similarity giữa vector thói quen và thuộc tính sân, kết hợp
   `45% thói quen + 30% khoảng cách + 15% rating + 10% mức quay lại`.
3. **Tìm kiếm ngữ nghĩa:** hybrid ranking `40% TF-IDF + 60% embedding cosine`.
   Embedding được lưu trong collection `embedding_records`; bộ chuẩn hóa tiếng Việt
   xử lý các cách gọi tương đương như “đá banh” và “bóng đá”. Khi bật MongoDB Vector
   Search, service dùng `$vectorSearch`; MongoDB local tự fallback sang cosine scan.
4. **Bảng tin cá nhân hóa:** user-user collaborative filtering từ VIEW, LIKE,
   COMMENT; BOOKING dùng cho hồ sơ thói quen sân. Điểm có time-decay 90 ngày và
   fallback popularity khi người dùng mới chưa đủ hàng xóm tương tự.
5. **Chatbot RAG:** chunk tài liệu, hybrid retrieval, giới hạn context rồi gọi API
   LLM tương thích OpenAI. Prompt chỉ cho phép trả lời từ context và phản hồi có
   nguồn. Nếu LLM chưa cấu hình hoặc lỗi, hệ thống trả lời extractive có căn cứ.
6. **Dự báo nhu cầu:** weighted seasonal average theo thứ/giờ, ưu tiên dữ liệu gần
   đây và trả về khoảng dự báo 95% đã chặn trong miền tỷ lệ lấp đầy `[0, 1]`.
7. **Xác minh người đại diện:** PaddleOCR đọc CCCD Việt Nam, YuNet phát hiện mặt,
   SFace đối chiếu khuôn mặt và MiniFASNet ONNX kiểm tra passive liveness. Kết quả
   chỉ lưu trong Redis theo TTL; khung hình camera không được lưu thành hồ sơ.

## API chính

- `POST /api/v1/ai/matchmaking/queue`
- `GET /api/v1/ai/recommendations/venues`
- `POST /api/v1/ai/personalization/interactions`
- `GET /api/v1/ai/personalization/feed`
- `POST /api/v1/ai/semantic-search/documents`
- `GET /api/v1/ai/semantic-search`
- `POST /api/v1/ai/chatbot/knowledge`
- `POST /api/v1/ai/chatbot/query`
- `POST /api/v1/ai/demand-forecast/observations`
- `POST|GET /api/v1/ai/demand-forecast`
- `GET /api/v1/ai/capabilities`
- `POST /api/v1/ai/identity-verifications`
- `GET /api/v1/ai/identity-verifications/{verificationId}?subjectId={userId}`

Các endpoint ghi nhận tương tác và lập chỉ mục là contract để social-service,
venue-service và club-service đồng bộ dữ liệu thật sang AI service. Không có dữ
liệu mẫu được tạo trong runtime.

## Cấu hình LLM

Đặt `LLM_BASE_URL`, `LLM_MODEL` và (nếu nhà cung cấp yêu cầu) `LLM_API_KEY`.
`LLM_BASE_URL` trỏ đến base URL tương thích OpenAI và service gọi
`POST {LLM_BASE_URL}/chat/completions`. Giới hạn truy xuất được điều chỉnh bằng
`RAG_MAX_CONTEXT_CHUNKS` và `RAG_MIN_RELEVANCE_SCORE`.

Để dùng ANN vector search, chạy job `mongodb-search-index-init` trong
`infrastructure/mongodb-search`, sau đó đặt `MONGODB_VECTOR_SEARCH_ENABLED=true`.
Nếu chưa có index, service ghi warning và fallback an toàn sang cosine search trong ứng dụng.

Với stack self-managed, đặt `AI_MONGODB_URI` để tách database AI khỏi URI MongoDB
dùng chung của các service khác. Khi `mongot` lỗi tạm thời, repository dùng local
ranking và chỉ thử lại `$vectorSearch` sau `MONGODB_VECTOR_SEARCH_RETRY_SECONDS`
(mặc định 60 giây). Cấu hình local và production không Kubernetes nằm tại
`infrastructure/mongodb-search`.

## Kiểm tra

```bash
python scripts/check_architecture.py
ruff check app tests scripts
python -m unittest discover -s tests -v
python -m scripts.smoke_self_hosted_vector_search
```

`tests/integration_smoke.py` chỉ dành cho MongoDB/Redis dùng một lần, không chạy
trên cơ sở dữ liệu production.

## Cấu trúc code

- `app/domain`: enum và entity/document theo class diagram, không phụ thuộc lớp ngoài.
- `app/application`: request/response DTO, port và các use case/thuật toán AI.
- `app/infrastructure`: cấu hình, MongoDB/Redis repository, HTTP và LLM adapter.
- `app/presentation`: FastAPI controller và GlobalExceptionHandler.

Dependency được lắp ráp tập trung tại `app/infrastructure/config/container.py`.
Workflow `.github/workflows/deploy.yml` chạy architecture check, Ruff, unit test,
sau đó build và push image `goatsports-ai-service` lên Docker Hub khi push `main`.

## Run code
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe scripts\download_vision_models.py

.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 0.0.0.0 `
  --port 7081 `
  --reload