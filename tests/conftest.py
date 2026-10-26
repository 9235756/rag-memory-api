"""Общие фикстуры: все тесты работают оффлайн в MOCK_MODE=true."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    for key in ("OPENAI_API_KEY", "PINECONE_API_KEY", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.3")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")


@pytest.fixture
def store(tmp_path):
    from vector_store import VectorStoreManager

    # отдельный JSON на каждый тест — тесты не видят данные друг друга и демо
    return VectorStoreManager(mock_path=tmp_path / "store.json")
