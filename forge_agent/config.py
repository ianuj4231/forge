from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    execution_db: Path
    checkpoint_db: Path
    provider: str
    model: str
    base_url: str
    temperature: float
    timeout_seconds: float
    retry_attempts: int
    retry_initial_backoff_seconds: float
    retry_max_backoff_seconds: float
    api_key: str | None
    fallback_without_key: bool
    api_title: str
    api_version: str


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    config_file = Path(config_path).resolve()
    root = config_file.parent
    load_dotenv(root / ".env")
    raw: dict[str, Any] = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    llm = raw.get("llm", {})
    api = raw.get("api", {})
    data_dir = root / raw.get("data_dir", ".")
    return Settings(
        root=root,
        data_dir=data_dir.resolve(),
        execution_db=(root / raw.get("execution_db", "executions.db")).resolve(),
        checkpoint_db=(root / raw.get("checkpoint_db", "checkpoints.db")).resolve(),
        provider=os.getenv("LLM_PROVIDER", llm.get("provider", "openrouter")),
        model=os.getenv("OPENROUTER_MODEL", llm.get("model", "openai/gpt-4o-mini")),
        base_url=os.getenv("OPENROUTER_BASE_URL", llm.get("base_url", "https://openrouter.ai/api/v1")),
        temperature=float(llm.get("temperature", 0)),
        timeout_seconds=float(llm.get("timeout_seconds", 30)),
        retry_attempts=int(llm.get("retry_attempts", 6)),
        retry_initial_backoff_seconds=float(llm.get("retry_initial_backoff_seconds", 1)),
        retry_max_backoff_seconds=float(llm.get("retry_max_backoff_seconds", 30)),
        api_key=os.getenv("OPENROUTER_API_KEY") or None,
        fallback_without_key=bool(llm.get("fallback_without_key", True)),
        api_title=api.get("title", "Forge Front-Desk Agent"),
        api_version=api.get("version", "0.1.0"),
    )
