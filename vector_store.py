"""ЗАДАНИЕ 1. Векторное хранилище: Pinecone (REAL) или InMemoryVectorStore (MOCK)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import DATA_DIR, Settings, get_settings, is_mock
from embeddings import MockEmbeddings, get_embeddings

try:  # официальная интеграция LangChain ↔ Pinecone
    from langchain_pinecone import PineconeVectorStore
except ImportError:  # pragma: no cover — fallback на прямую работу с клиентом
    PineconeVectorStore = None

MOCK_STORE_PATH = DATA_DIR / "mock_store.json"


class VectorStoreError(RuntimeError):
    pass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _match_filter(metadata: dict, flt: dict | None) -> bool:
    """Подмножество синтаксиса фильтров Pinecone: {"k": v} и {"k": {"$eq": v}}."""
    if not flt:
        return True
    for key, cond in flt.items():
        expected = cond.get("$eq") if isinstance(cond, dict) else cond
        if metadata.get(key) != expected:
            return False
    return True


class InMemoryVectorStore:
    """Эмуляция Pinecone: косинусное сходство на чистом Python + JSON на диске."""

    def __init__(self, embeddings: Embeddings, path: Path | None = MOCK_STORE_PATH):
        self.embeddings = embeddings
        self.path = path
        # {namespace: {id: {"text", "metadata", "values"}}}
        self.data: dict[str, dict[str, dict[str, Any]]] = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def upsert(self, docs: list[Document], ids: list[str], namespace: str) -> list[str]:
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        bucket = self.data.setdefault(namespace, {})
        for doc, doc_id, values in zip(docs, ids, vectors):
            bucket[doc_id] = {"text": doc.page_content, "metadata": dict(doc.metadata),
                              "values": values}
        self._save()
        return ids

    def query(self, query: str, k: int, namespace: str,
              flt: dict | None = None) -> list[tuple[Document, float]]:
        qv = self.embeddings.embed_query(query)
        scored = []
        for doc_id, item in self.data.get(namespace, {}).items():
            if not _match_filter(item["metadata"], flt):
                continue
            doc = Document(page_content=item["text"], metadata=item["metadata"], id=doc_id)
            scored.append((doc, _cosine(qv, item["values"])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def delete_namespace(self, namespace: str) -> None:
        self.data.pop(namespace, None)
        self._save()

    def stats(self) -> dict:
        namespaces = {ns: {"vector_count": len(items)} for ns, items in self.data.items()}
        return {"namespaces": namespaces,
                "total_vector_count": sum(n["vector_count"] for n in namespaces.values())}


class _DirectPineconeStore:
    """Fallback без langchain_pinecone: index.upsert / index.query напрямую."""

    def __init__(self, index, embeddings: Embeddings, namespace: str):
        self.index, self.embeddings, self.namespace = index, embeddings, namespace

    def add_documents(self, docs: list[Document], ids: list[str]) -> list[str]:
        values = self.embeddings.embed_documents([d.page_content for d in docs])
        vectors = [{"id": i, "values": v, "metadata": {**d.metadata, "text": d.page_content}}
                   for d, i, v in zip(docs, ids, values)]
        for start in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[start:start + 100], namespace=self.namespace)
        return ids

    def similarity_search_with_score(self, query: str, k: int, filter: dict | None = None):
        res = self.index.query(vector=self.embeddings.embed_query(query), top_k=k,
                               namespace=self.namespace, include_metadata=True, filter=filter)
        out = []
        for m in res.get("matches", []):
            meta = dict(m.get("metadata") or {})
            text = meta.pop("text", "")
            out.append((Document(page_content=text, metadata=meta, id=m["id"]), m["score"]))
        return out


class VectorStoreManager:
    """Единый интерфейс к хранилищу; остальной код не знает, Pinecone это или мок."""

    def __init__(self, settings: Settings | None = None,
                 embeddings: Embeddings | None = None,
                 mock_path: Path | None = MOCK_STORE_PATH):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or get_embeddings(self.settings)
        self.mock = is_mock("pinecone", self.settings)
        self.mode = "MOCK" if self.mock else "REAL"
        self._mock_path = mock_path
        self._pc = None
        self._index = None
        self._stores: dict[str, Any] = {}
        self._memory: InMemoryVectorStore | None = None

    # ---------- подключение ----------
    def ensure_index(self) -> None:
        """Создаёт индекс при необходимости и проверяет его размерность."""
        if self.mock:
            if self._memory is None:
                self._memory = InMemoryVectorStore(self.embeddings, self._mock_path)
            return
        if self._index is not None:
            return
        from pinecone import Pinecone, ServerlessSpec

        s = self.settings
        self._pc = Pinecone(api_key=s.pinecone_api_key)
        if s.pinecone_index_name not in self._pc.list_indexes().names():
            print(f"[Pinecone] Индекс {s.pinecone_index_name!r} не найден — создаю "
                  f"(dim={s.embedding_dim}, cosine, {s.pinecone_cloud}/{s.pinecone_region})")
            self._pc.create_index(
                name=s.pinecone_index_name, dimension=s.embedding_dim, metric="cosine",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
            deadline = time.time() + 120
            while not self._pc.describe_index(s.pinecone_index_name).status["ready"]:
                if time.time() > deadline:
                    raise VectorStoreError("Индекс не перешёл в состояние ready за 120 с")
                time.sleep(2)
        desc = self._pc.describe_index(s.pinecone_index_name)
        if int(desc.dimension) != s.embedding_dim:
            raise VectorStoreError(
                f"Размерность индекса {s.pinecone_index_name!r} = {desc.dimension}, "
                f"а EMBEDDING_DIM = {s.embedding_dim}. Удалите индекс или укажите "
                f"другой PINECONE_INDEX_NAME / EMBEDDING_DIM."
            )
        self._index = self._pc.Index(s.pinecone_index_name)

    def _store(self, namespace: str):
        if namespace not in self._stores:
            if PineconeVectorStore is not None:
                self._stores[namespace] = PineconeVectorStore(
                    index=self._index, embedding=self.embeddings, namespace=namespace)
            else:
                self._stores[namespace] = _DirectPineconeStore(
                    self._index, self.embeddings, namespace)
        return self._stores[namespace]

    # ---------- запись / поиск ----------
    def add_documents(self, docs: list[Document], namespace: str | None = None,
                      ids: list[str] | None = None) -> list[str]:
        self.ensure_index()
        namespace = namespace or self.settings.namespace_kb
        ids = ids or [d.id or f"doc-{abs(hash(d.page_content))}" for d in docs]
        if self.mock:
            return self._memory.upsert(docs, ids, namespace)
        return self._store(namespace).add_documents(docs, ids=ids)

    def _calibrate(self, score: float) -> float:
        """Хеш-векторы дают cos в разы меньше, чем OpenAI-эмбеддинги: короткий
        запрос и длинный чанк делят 1–3 слова. √cos сохраняет порядок результатов
        и переводит шкалу в сопоставимую, чтобы SCORE_THRESHOLD работал одинаково."""
        if isinstance(self.embeddings, MockEmbeddings):
            return math.sqrt(max(score, 0.0))
        return score

    def similarity_search(self, query: str, k: int | None = None,
                          namespace: str | None = None, filter: dict | None = None,
                          apply_threshold: bool = True) -> list[tuple[Document, float]]:
        self.ensure_index()
        k = k or self.settings.top_k
        namespace = namespace or self.settings.namespace_kb
        if self.mock:
            raw = self._memory.query(query, k, namespace, filter)
        else:
            raw = self._store(namespace).similarity_search_with_score(query, k=k, filter=filter)
        results = [(doc, round(self._calibrate(score), 4)) for doc, score in raw]
        if apply_threshold:
            results = [(d, sc) for d, sc in results if sc >= self.settings.score_threshold]
        return results

    def delete_namespace(self, namespace: str) -> None:
        self.ensure_index()
        if self.mock:
            self._memory.delete_namespace(namespace)
            return
        try:
            self._index.delete(delete_all=True, namespace=namespace)
        except Exception:  # namespace ещё не существует — это не ошибка
            pass

    # ---------- диагностика ----------
    def list_indexes(self) -> list[dict]:
        self.ensure_index()
        s = self.settings
        if self.mock:
            return [{"name": s.pinecone_index_name, "dimension": s.embedding_dim,
                     "metric": "cosine", "host": "mock://local (data/mock_store.json)",
                     "spec": {"serverless": {"cloud": s.pinecone_cloud, "region": s.pinecone_region}},
                     "status": {"ready": True, "state": "Ready"}}]
        out = []
        for idx in self._pc.list_indexes():
            d = idx.to_dict() if hasattr(idx, "to_dict") else dict(idx)
            out.append({k: d.get(k) for k in ("name", "dimension", "metric", "host", "spec", "status")})
        return out

    def stats(self) -> dict:
        self.ensure_index()
        if self.mock:
            return self._memory.stats()
        raw = self._index.describe_index_stats()
        raw = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        namespaces = {ns: {"vector_count": v.get("vector_count", 0)}
                      for ns, v in (raw.get("namespaces") or {}).items()}
        return {"namespaces": namespaces,
                "total_vector_count": raw.get("total_vector_count", 0),
                "dimension": raw.get("dimension")}

    def health_check(self) -> dict:
        started = time.perf_counter()
        try:
            self.ensure_index()
            stats = self.stats()
            return {"ok": True, "mode": self.mode, "index": self.settings.pinecone_index_name,
                    "dimension": self.settings.embedding_dim,
                    "total_vectors": stats.get("total_vector_count", 0),
                    "latency_ms": round((time.perf_counter() - started) * 1000)}
        except Exception as exc:
            return {"ok": False, "mode": self.mode, "error": f"{type(exc).__name__}: {exc}"}
