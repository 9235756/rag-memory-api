"""ЗАДАНИЕ 1 (проверка): подключение к Pinecone, запись 3 документов, тестовый запрос."""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from config import get_settings, print_modes  # noqa: E402
from vector_store import VectorStoreManager  # noqa: E402

NAMESPACE = "healthcheck"
TEST_QUERY = "Что такое векторная база данных?"
TEST_DOCS = [
    ("hc-pinecone", "Pinecone — управляемая векторная база данных. Она хранит эмбеддинги "
                    "и быстро находит ближайшие векторы по косинусному сходству."),
    ("hc-rag", "RAG (Retrieval-Augmented Generation) — подход, при котором языковая модель "
               "перед ответом ищет релевантные фрагменты в базе знаний и опирается на них."),
    ("hc-telegram", "Telegram-бот принимает сообщения пользователя через Bot API и отвечает "
                    "ему в чате; для Python есть библиотека pyTelegramBotAPI."),
]


def run_check(store: VectorStoreManager | None = None) -> bool:
    settings = get_settings()
    print_modes(settings)
    store = store or VectorStoreManager(settings)
    try:
        print(f"\n1) Подключение к Pinecone (режим: {store.mode})")
        store.ensure_index()
        print("   list_indexes():")
        for idx in store.list_indexes():
            print("   " + json.dumps(idx, ensure_ascii=False, default=str))
        print("   describe_index_stats(): "
              + json.dumps(store.stats(), ensure_ascii=False, default=str))

        print(f"\n2) Запись {len(TEST_DOCS)} тестовых документов в namespace «{NAMESPACE}»")
        docs = [Document(page_content=text, metadata={"source": "healthcheck", "title": doc_id})
                for doc_id, text in TEST_DOCS]
        store.add_documents(docs, namespace=NAMESPACE, ids=[d for d, _ in TEST_DOCS])
        for doc_id, text in TEST_DOCS:
            print(f"   upsert {doc_id}: {text[:60]}…")

        print(f"\n3) Тестовый запрос: «{TEST_QUERY}»")
        # Pinecone serverless индексирует запись с небольшой задержкой — повторяем
        results = []
        for attempt in range(10):
            results = store.similarity_search(TEST_QUERY, k=3, namespace=NAMESPACE,
                                              apply_threshold=False)
            if results or store.mock:
                break
            time.sleep(2)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"   #{rank} score={score:.4f}  [{doc.metadata.get('title')}] "
                  f"{doc.page_content[:70]}…")

        if not results:
            raise RuntimeError("тестовый запрос не вернул ни одного результата")
        top = results[0][0].metadata.get("title")
        if top != "hc-pinecone":
            raise RuntimeError(f"первым ожидался документ про Pinecone, получен {top}")
        print(f"\n✅ Интеграция с Pinecone работает (режим: {store.mode})")
        return True
    except Exception as exc:
        print(f"\n❌ Интеграция с Pinecone не работает (режим: {store.mode}): "
              f"{type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
