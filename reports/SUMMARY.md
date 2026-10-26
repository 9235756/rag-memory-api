# VPg04 — сводка прогона

Дата/время запуска: **2026-10-26 22:41:02**

Режимы сервисов: openai=MOCK, pinecone=MOCK, telegram=MOCK, http=REAL (MOCK_MODE=auto)

| Задание | Статус | Режим | Файл-доказательство |
|---|---|---|---|
| 1. Pinecone: подключение и тестовый запрос | ✅ | MOCK | [01_pinecone_check.md](01_pinecone_check.md) |
| 2. URL → чанки → эмбеддинги → база + поиск | ✅ | MOCK | [02_url_ingest.md](02_url_ingest.md) |
| 3. API-инструмент (catfact.ninja) | ✅ | REAL | [03_api_tool.md](03_api_tool.md) |
| Telegram-бот (эмуляция диалога) | ✅ | MOCK | [04_telegram_dialog.md](04_telegram_dialog.md) |

MOCK — сервис эмулируется, потому что ключ в `local.env` не задан. Впишите ключи и
перезапустите `python scripts/run_demo.py` — отчёты пересоздадутся в режиме REAL.
