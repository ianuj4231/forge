from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI

from forge_agent.config import load_settings
from forge_agent.data import ClinicData
from forge_agent.graph import build_graph
from forge_agent.llm import LLMFactory
from forge_agent.logging_config import configure_logging
from forge_agent.persistence import ExecutionStore
from forge_agent.schemas import MessageInput, ProcessResponse


logger = logging.getLogger(__name__)
configure_logging()


@lru_cache(maxsize=1)
def get_processor():
    settings = load_settings()
    data = ClinicData(settings.data_dir)
    store = ExecutionStore(settings.execution_db)
    return build_graph(settings, data, LLMFactory(settings), store)


settings = load_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
def process(message: MessageInput) -> ProcessResponse:
    logger.info("process request received message_id=%s channel=%s", message.id, message.channel)
    state = get_processor()(message)
    logger.info(
        "process request returning execution_id=%s message_id=%s",
        state["execution_id"],
        message.id,
    )
    return ProcessResponse(execution_id=state["execution_id"], **state["result"])
