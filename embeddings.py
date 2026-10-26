"""Эмбеддинги: OpenAI в реальном режиме, детерминированные хеш-векторы в моке."""
from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from config import Settings, get_settings, is_mock

# Короткий список стоп-слов: без них запрос «Что такое Земля?» сводится к сути
STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "о", "об", "а", "но", "к", "ко", "у", "из",
    "за", "от", "до", "для", "что", "такое", "как", "это", "этот", "эта", "то", "ли",
    "же", "бы", "не", "какой", "какая", "какое", "какие", "каков", "какова",
    "сколько", "где", "когда", "почему", "зачем", "расскажи", "скажи",
    "я", "мне", "меня", "мой", "моя", "ты", "вы", "он", "она", "они",
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what", "how",
}

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_RU_ENDINGS = re.compile(
    r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ией|ов|ев|ей|ий|ый|ой|ая|яя|ое|ее|ые|ие"
    r"|ом|ем|ах|ях|ам|ям|ую|юю|а|я|о|е|ы|и|у|ю|ь)$"
)


def tokenize(text: str) -> list[str]:
    """Токены → грубые основы: «Земля», «Земли», «Землю» → «земл»."""
    stems = []
    for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        stem = _RU_ENDINGS.sub("", tok)
        stems.append(stem if len(stem) >= 3 else tok)
    return stems


class MockEmbeddings(Embeddings):
    """Bag-of-words hashing + L2-нормализация.

    Каждая основа слова хешируется (md5, стабильно между запусками) в одну из
    `dim` позиций со знаком ±1. Тексты с общими словами дают близкие векторы,
    поэтому поиск в мок-режиме осмысленный, хоть и без «понимания» синонимов.
    Вес бинарный (слово есть / нет): иначе чанк, где «Земля» повторена 10 раз,
    обгоняет чанк, где есть и «возраст», и «Земля».
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for stem in set(tokenize(text)):
            digest = hashlib.md5(stem.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # Пустой текст: не нулевой вектор (Pinecone не принимает нулевые)
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if is_mock("openai", s):
        return MockEmbeddings(dim=s.embedding_dim)
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": s.embedding_model, "api_key": s.openai_api_key}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    # text-embedding-3-* умеют укорачивать вектор под размерность индекса
    if s.embedding_model.startswith("text-embedding-3"):
        kwargs["dimensions"] = s.embedding_dim
    return OpenAIEmbeddings(**kwargs)
