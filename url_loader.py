"""ЗАДАНИЕ 2. URL → HTML → текст → чанки → эмбеддинги → векторная база."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, Settings, allow_fallback, get_settings, is_mock

SAMPLE_PAGE = DATA_DIR / "sample_page.html"
USER_AGENT = "rag-memory-api/1.0 (educational RAG homework; python-requests)"
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript"]
# Служебные блоки Википедии и похожих сайтов: сноски [1], «править», навигация
NOISE_SELECTORS = [".mw-editsection", "sup.reference", ".navbox", ".reflist",
                   ".mw-jump-link", ".noprint", "#toc", ".toc"]


class UrlLoadError(RuntimeError):
    pass


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise UrlLoadError(f"Нужна ссылка вида http(s)://…, получено: {url!r}")
    return url.strip()


def _load_sample_page(reason: str) -> tuple[str, bool]:
    print(f"[MOCK] используется оффлайн-страница {SAMPLE_PAGE.name} ({reason})")
    return SAMPLE_PAGE.read_text(encoding="utf-8"), True


def fetch_url(url: str, settings: Settings | None = None) -> tuple[str, bool]:
    """Возвращает (html, is_offline_page)."""
    s = settings or get_settings()
    url = validate_url(url)
    if is_mock("http", s):
        return _load_sample_page("MOCK_MODE=true — сеть не используется")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=s.http_timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            raise UrlLoadError(f"Ожидался text/html, получено: {ctype or 'нет заголовка'}")
        # requests по умолчанию ставит ISO-8859-1, если charset не указан
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding
        return resp.text, False
    except (requests.RequestException, UrlLoadError) as exc:
        if allow_fallback(s):
            return _load_sample_page(f"URL недоступен: {type(exc).__name__}")
        raise UrlLoadError(f"Не удалось загрузить {url}: {exc}") from exc


def parse_html(html: str) -> tuple[str, str]:
    """Возвращает (title, чистый текст)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    # Абзацы склеиваем через перевод строки — сплиттер режет по ним в первую очередь
    blocks = [re.sub(r"\s+", " ", b.get_text(" ", strip=True))
              for b in root.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote", "dd"])]
    blocks = [b for b in blocks if len(b) > 1]
    text = "\n\n".join(blocks) if blocks else re.sub(r"\s+", " ", root.get_text(" ", strip=True))
    text = re.sub(r"\[\d+\]", "", text)  # остатки сносок
    return title or "Без заголовка", text.strip()


def split_text(text: str, settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size,
                                              chunk_overlap=s.chunk_overlap)
    return splitter.split_text(text)


def doc_id(url: str, i: int) -> str:
    """Детерминированный id: повторная загрузка той же страницы = upsert без дублей."""
    return f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}-{i}"


def ingest_url(url: str, store, html: str | None = None) -> dict:
    """Полный конвейер. store — VectorStoreManager. html можно передать готовым (тесты)."""
    s = store.settings
    print(f"Загрузка → {url}")
    offline = False
    if html is None:
        html, offline = fetch_url(url, s)
    title, text = parse_html(html)
    print(f"Парсинг → «{title}», {len(text)} символов")
    chunks = split_text(text, s)
    if not chunks:
        raise UrlLoadError("После парсинга страница пустая — сохранять нечего")
    print(f"Чанков: {len(chunks)}")

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    docs = [Document(page_content=chunk,
                     metadata={"source": url, "title": title, "chunk_index": i,
                               "total_chunks": len(chunks), "ingested_at": ingested_at})
            for i, chunk in enumerate(chunks)]
    ids = [doc_id(url, i) for i in range(len(chunks))]
    print(f"Эмбеддинги → {type(store.embeddings).__name__}")
    store.add_documents(docs, namespace=s.namespace_kb, ids=ids)
    target = "Pinecone" if store.mode == "REAL" else "Mock"
    print(f"Сохранено в {target} (namespace={s.namespace_kb}, векторов: {len(ids)})")

    dim = len(store.embeddings.embed_query("размерность"))
    return {"url": url, "title": title, "chars": len(text), "chunks": len(chunks),
            "embedding_dim": dim, "namespace": s.namespace_kb, "store": target,
            "offline_page": offline, "ids": ids, "sample_chunks": chunks[:2]}
