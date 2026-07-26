# LIT-DISC Structured Logger
# Reference: Architecture Specification §13

import logging
import json
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "pipeline"):
            log_entry["pipeline"] = record.pipeline
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        return json.dumps(log_entry)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    return logger
