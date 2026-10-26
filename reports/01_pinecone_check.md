Режим: MOCK (эмуляция, ключи не заданы)

> Сервисы: pinecone=MOCK, openai=MOCK

# Задание 1. Подключение к Pinecone и тестовый запрос

_Сгенерировано: 2026-10-26 22:41:02_

Команда: `python scripts/check_pinecone.py`

```text
┌─ Режимы работы (MOCK_MODE=auto) ────────────────────
│ OpenAI               MOCK   ключ: (не задан)
│ Pinecone             MOCK   ключ: (не задан)
│ Telegram             MOCK   ключ: (не задан)
│ HTTP (URL, Cat API)  REAL   ключ: ключ не нужен
└──────────────────────────────────────────────────

1) Подключение к Pinecone (режим: MOCK)
   list_indexes():
   {"name": "rag-memory", "dimension": 1536, "metric": "cosine", "host": "mock://local (data/mock_store.json)", "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}, "status": {"ready": true, "state": "Ready"}}
   describe_index_stats(): {"namespaces": {"healthcheck": {"vector_count": 3}, "knowledge": {"vector_count": 121}, "user-memory": {"vector_count": 4}}, "total_vector_count": 128}

2) Запись 3 тестовых документов в namespace «healthcheck»
   upsert hc-pinecone: Pinecone — управляемая векторная база данных. Она хранит эмб…
   upsert hc-rag: RAG (Retrieval-Augmented Generation) — подход, при котором я…
   upsert hc-telegram: Telegram-бот принимает сообщения пользователя через Bot API …

3) Тестовый запрос: «Что такое векторная база данных?»
   #1 score=0.6931  [hc-pinecone] Pinecone — управляемая векторная база данных. Она хранит эмбеддинги и …
   #2 score=0.3689  [hc-rag] RAG (Retrieval-Augmented Generation) — подход, при котором языковая мо…
   #3 score=0.0000  [hc-telegram] Telegram-бот принимает сообщения пользователя через Bot API и отвечает…

✅ Интеграция с Pinecone работает (режим: MOCK)
```
