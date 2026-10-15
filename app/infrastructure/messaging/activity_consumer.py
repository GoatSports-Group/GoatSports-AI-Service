"""Nghe `user-activity-topic` và chuyển từng sự kiện cho ActivityIngestService.

Giao ít nhất một lần: chỉ commit offset sau khi đã ghi xong. Ghi lỗi (vd. Mongo mất kết nối) thì quay lại đúng bản tin
đó và thử lại, không nhảy qua. Bản tin hỏng định dạng thì bỏ qua (ingest service đã ghi log), để không kẹt cả hàng đợi.
Ghi trùng vô hại vì mỗi hành vi có mã cố định.
"""

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_SECONDS = 5.0


class ActivityEventConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str,
                 handler: Callable[[dict[str, Any]], None]) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.handler = handler
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="activity-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        from confluent_kafka import Consumer, KafkaError, TopicPartition

        consumer = Consumer({
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "enable.auto.commit": False,
            # Lần đầu nhóm này đọc topic: lấy từ đầu, để sự kiện nạp bù phát trước khi ai-service lên không bị mất.
            "auto.offset.reset": "earliest",
        })
        consumer.subscribe([self.topic])
        logger.info("Listening for activity events on %s", self.topic)
        try:
            while not self._stop.is_set():
                message = consumer.poll(1.0)
                if message is None:
                    continue
                if message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        logger.warning("Activity consumer error: %s", message.error())
                    continue
                try:
                    event = json.loads(message.value())
                except (TypeError, ValueError):
                    logger.warning("Skipping non-JSON activity event at offset %s", message.offset())
                    consumer.commit(message=message, asynchronous=False)
                    continue
                try:
                    self.handler(event)
                except Exception:  # noqa: BLE001 - lỗi hạ tầng: thử lại đúng bản tin này, không mất hành vi
                    logger.exception("Could not store activity event, retrying in %ss", _RETRY_SECONDS)
                    consumer.seek(TopicPartition(message.topic(), message.partition(), message.offset()))
                    self._stop.wait(_RETRY_SECONDS)
                    continue
                consumer.commit(message=message, asynchronous=False)
        finally:
            consumer.close()
