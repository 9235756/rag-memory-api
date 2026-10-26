"""Загрузка настроек из local.env и определение режима работы (REAL / MOCK).

Все ключи читаются только из local.env. Переменные, уже заданные в окружении
(например, через monkeypatch в тестах), имеют приоритет над файлом.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / "local.env"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# override=False: переменные окружения важнее файла (нужно для тестов и CI)
load_dotenv(ENV_FILE, override=False)


class ConfigError(RuntimeError):
    """Ошибка конфигурации: например, MOCK_MODE=false, а ключ не задан."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip() or default


@dataclass(frozen=True)
class Settings:
    mock_mode: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    embedding_dim: int
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    namespace_kb: str
    namespace_memory: str
    telegram_bot_token: str
    cat_fact_api_url: str
    http_timeout: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float


def get_settings() -> Settings:
    """Читает настройки заново при каждом вызове — так тесты могут менять env."""
    mode = _str("MOCK_MODE", "auto").lower()
    if mode not in {"auto", "true", "false"}:
        raise ConfigError(f"MOCK_MODE должен быть auto/true/false, получено: {mode!r}")
    return Settings(
        mock_mode=mode,
        openai_api_key=_str("OPENAI_API_KEY"),
        openai_base_url=_str("OPENAI_BASE_URL"),
        openai_model=_str("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=_str("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=_int("EMBEDDING_DIM", 1536),
        pinecone_api_key=_str("PINECONE_API_KEY"),
        pinecone_index_name=_str("PINECONE_INDEX_NAME", "rag-memory"),
        pinecone_cloud=_str("PINECONE_CLOUD", "aws"),
        pinecone_region=_str("PINECONE_REGION", "us-east-1"),
        namespace_kb=_str("PINECONE_NAMESPACE_KB", "knowledge"),
        namespace_memory=_str("PINECONE_NAMESPACE_MEMORY", "user-memory"),
        telegram_bot_token=_str("TELEGRAM_BOT_TOKEN"),
        cat_fact_api_url=_str("CAT_FACT_API_URL", "https://catfact.ninja/fact"),
        http_timeout=_float("HTTP_TIMEOUT", 10.0),
        chunk_size=_int("CHUNK_SIZE", 800),
        chunk_overlap=_int("CHUNK_OVERLAP", 100),
        top_k=_int("TOP_K", 4),
        score_threshold=_float("SCORE_THRESHOLD", 0.3),
    )


_KEY_BY_SERVICE = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "pinecone": ("pinecone_api_key", "PINECONE_API_KEY"),
    "telegram": ("telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
}


def is_mock(service: str, settings: Settings | None = None) -> bool:
    """Нужно ли эмулировать сервис.

    service: "openai" | "pinecone" | "telegram" — сервисы с ключом;
             "http" — внешние HTTP-запросы без ключа (Cat API, загрузка URL):
             эмулируются только при MOCK_MODE=true (полный оффлайн).
    """
    s = settings or get_settings()
    if service == "http":
        return s.mock_mode == "true"
    if service not in _KEY_BY_SERVICE:
        raise ValueError(f"Неизвестный сервис: {service}")
    if s.mock_mode == "true":
        return True
    attr, env_name = _KEY_BY_SERVICE[service]
    has_key = bool(getattr(s, attr))
    if s.mock_mode == "false":
        if not has_key:
            raise ConfigError(
                f"MOCK_MODE=false, но {env_name} не задан в local.env. "
                f"Впишите ключ или переключите MOCK_MODE=auto."
            )
        return False
    return not has_key  # auto


def allow_fallback(settings: Settings | None = None) -> bool:
    """Можно ли подменять недоступный сервис эмуляцией (всё, кроме MOCK_MODE=false)."""
    s = settings or get_settings()
    return s.mock_mode != "false"


def mask_secret(value: str) -> str:
    """sk-proj-abcdef...1234 → sk-...1234. Сам ключ никогда не печатаем."""
    if not value:
        return "(не задан)"
    prefix = value.split("-")[0] + "-" if "-" in value[:8] else value[:2]
    return f"{prefix}...{value[-4:]}"


def mode_label(service: str, settings: Settings | None = None) -> str:
    try:
        return "MOCK" if is_mock(service, settings) else "REAL"
    except ConfigError:
        return "ERROR"


def print_modes(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    rows = [
        ("OpenAI", mode_label("openai", s), mask_secret(s.openai_api_key)),
        ("Pinecone", mode_label("pinecone", s), mask_secret(s.pinecone_api_key)),
        ("Telegram", mode_label("telegram", s), mask_secret(s.telegram_bot_token)),
        ("HTTP (URL, Cat API)", mode_label("http", s), "ключ не нужен"),
    ]
    print(f"┌─ Режимы работы (MOCK_MODE={s.mock_mode}) " + "─" * 20)
    for name, mode, key in rows:
        print(f"│ {name:<20} {mode:<5}  ключ: {key}")
    print("└" + "─" * 50)
