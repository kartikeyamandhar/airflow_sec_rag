"""The logger emits structured key-value events."""

import structlog

from app.logging import configure_logging, get_logger


def test_logger_emits_structured_event() -> None:
    configure_logging()
    logger = get_logger("test")
    with structlog.testing.capture_logs() as logs:
        logger.info("acquired_filing", accession="0000320193-23-000106", count=3)
    assert len(logs) == 1
    event = logs[0]
    assert event["event"] == "acquired_filing"
    assert event["accession"] == "0000320193-23-000106"
    assert event["count"] == 3
    assert event["log_level"] == "info"
