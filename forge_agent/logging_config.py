from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    """Make application logs visible without duplicating Uvicorn logs."""
    application_logger = logging.getLogger("forge_agent")
    application_logger.setLevel(logging.INFO)

    if application_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    )
    application_logger.addHandler(handler)
    application_logger.propagate = False
