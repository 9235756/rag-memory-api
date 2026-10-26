# VPg04 — Интеграция памяти с API: RAG-агент + Pinecone + URL + API

Telegram-бот с RAG-агентом. Агент хранит базу знаний и память о пользователе в векторной базе **Pinecone**, загружает веб-страницы по ссылке и ходит во внешний API (случайный факт о котах). Без ключей проект целиком работает в **режиме эмуляции (MOCK)**, поэтому сдать задание можно и без регистрации в сервисах.

| Задание | Где реализовано | Как проверить |
|---|---|---|
| 1. RAG-агент + Pinecone, тестовый запрос | `vector_store.py`, `scripts/check_pinecone.py` | `python scripts/check_pinecone.py` |
| 2. URL → HTML → чанки → эмбеддинги → база | `url_loader.py` | `/add_url <ссылка>`, затем `/search <запрос>` |
| 3. API-инструмент вместо заглушки | `api_tools.py` (`create_api_request_tool`) | `/catfact` или «расскажи факт о кошках» |

## Архитектура

```
                         ┌──────────────────────────────────────────────┐
 User ──► Telegram-бот ──►                RAGAgent                      │
          (telegram_bot) │  1. memory.py: «меня зовут…» → факты         │
 User ──► Консоль (cli)  │  2. маршрутизация: коты? ──► Cat API          │
                         │  3. иначе RAG: поиск → промпт → LLM           │
                         └───┬──────────────┬──────────────┬────────────┘
                             │              │              │
                ┌────────────▼───┐  ┌───────▼───────┐  ┌───▼──────────────┐
                │ Pinecone       │  │ URL loader    │  │ Cat API          │
                │ knowledge      │◄─┤ requests + BS4│  │ catfact.ninja    │
                │ user-memory    │  │ text splitter │  │ GET + ретраи     │
                └───────▲────────┘  └───────────────┘  └──────────────────┘
                        │ эмбеддинги (OpenAI / MockEmbeddings)
                ┌───────┴────────┐
                │ LLM            │  ChatOpenAI / MockLLM
                └────────────────┘
```

| Модуль | Назначение |
|---|---|
| `config.py` | Чтение `local.env`, `Settings`, `is_mock(service)`, таблица режимов (ключи маскируются) |
| `embeddings.py` | `OpenAIEmbeddings` или `MockEmbeddings` (хеширование слов + L2-нормализация) |
| `vector_store.py` | `VectorStoreManager`: Pinecone (`PineconeVectorStore`, fallback на голый клиент) или `InMemoryVectorStore` с JSON на диске |
| `url_loader.py` | `fetch_url`, `parse_html`, `split_text`, `ingest_url` |
| `api_tools.py` | `create_api_request_tool()` → LangChain-инструмент `get_random_cat_fact` |
| `memory.py` | Эвристики фактов о пользователе → namespace `user-memory` с `user_id` |
| `llm.py` | `ChatOpenAI` или `MockLLM` (экстрактивный ответ из контекста) |
| `rag_agent.py` | `RAGAgent` и общий обработчик команд `execute_command` |
| `telegram_bot.py` | Бот на `pyTelegramBotAPI`; без токена эмулирует диалог через те же хендлеры |
| `cli.py` | Консольный чат с теми же командами |

## Установка

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp local.env.example local.env
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy local.env.example local.env
```

Нужен Python 3.10+.

## Настройка `local.env`

Все ключи и настройки читаются только из `local.env`. Он в `.gitignore`, в репозиторий попадает только шаблон `local.env.example`.

| Переменная | Где взять / что значит |
|---|---|
| `MOCK_MODE` | `auto` — реальные сервисы, где есть ключ, остальное эмулируется; `true` — всё оффлайн; `false` — только реальные сервисы (ошибка без ключа) |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `OPENAI_BASE_URL` | Необязательно: адрес прокси или совместимого провайдера, например `https://api.proxyapi.ru/openai/v1` |
| `PINECONE_API_KEY` | https://app.pinecone.io → API Keys (бесплатного Starter-плана достаточно) |
| `PINECONE_INDEX_NAME` | Индекс создаётся автоматически: `dimension=EMBEDDING_DIM`, `metric=cosine`, serverless |
| `TELEGRAM_BOT_TOKEN` | Telegram → @BotFather → `/newbot` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SCORE_THRESHOLD` | Параметры чанкинга и поиска |

Сервисы задаются независимо. Например, только `PINECONE_API_KEY` даст настоящий Pinecone при эмулированных эмбеддингах.

## Запуск

```bash
pytest -q                          # 28 тестов, оффлайн
python scripts/check_pinecone.py   # Задание 1: подключение + тестовый запрос
python scripts/run_demo.py         # все задания → reports/
python telegram_bot.py             # бот (без токена — эмуляция диалога)
python cli.py                      # консольный чат
```

Команды бота и консоли:

| Команда | Что делает |
|---|---|
| `/start`, `/help` | Приветствие и список команд |
| `/add_url <ссылка>` | Загрузка → парсинг → чанки → эмбеддинги → Pinecone (namespace `knowledge`) |
| `/search <запрос>` | Семантический поиск, топ-k со score |
| `/catfact` | Случайный факт о котах из https://catfact.ninja/fact |
| `/memory` | Что бот помнит о пользователе |
| `/stats` | Режимы сервисов и число векторов по namespace |
| обычный текст | Вопрос к базе знаний; «меня зовут…», «мне N лет», «я живу в…», «я работаю…», «я люблю…», «моя цель…» запоминаются |

## Режим MOCK

| Сервис | Реальный режим | Эмуляция |
|---|---|---|
| Эмбеддинги | `OpenAIEmbeddings` | `MockEmbeddings`: основы слов хешируются в вектор длины `EMBEDDING_DIM`, тексты с общими словами оказываются рядом |
| Pinecone | serverless-индекс | `InMemoryVectorStore`: косинусное сходство на чистом Python, сохраняется в `data/mock_store.json` |
| LLM | `ChatOpenAI` | `MockLLM`: «По данным базы знаний: …» и 2–3 самых релевантных предложения из контекста |
| Загрузка URL | `requests` | При недоступности сайта или `MOCK_MODE=true` берётся `data/sample_page.html` с пометкой `[MOCK]` в логе |
| Cat API | GET-запрос (ключ не нужен) | При сбое сети или `MOCK_MODE=true` выдаётся один из 10 локальных фактов, `source: "mock"` |
| Telegram | `infinity_polling` | Сценарий диалога через те же хендлеры с эмулированными `message` |

Хеш-векторы дают косинус заметно меньше, чем OpenAI-эмбеддинги: короткий запрос и длинный чанк делят всего пару слов. Поэтому в мок-режиме score приводится к сопоставимой шкале как `√cos`. Порядок результатов от этого не меняется, а `SCORE_THRESHOLD` работает одинаково в обоих режимах.

## Результаты

Отчёты последнего прогона `python scripts/run_demo.py` лежат в [`reports/`](reports/):

- [`SUMMARY.md`](reports/SUMMARY.md) — сводная таблица: статус, режим, файл-доказательство
- [`01_pinecone_check.md`](reports/01_pinecone_check.md) — `list_indexes`, статистика, тестовый запрос с score
- [`02_url_ingest.md`](reports/02_url_ingest.md) — загрузка статьи «Земля» из Википедии, пример чанков, поиск
- [`03_api_tool.md`](reports/03_api_tool.md) — код «БЫЛО / СТАЛО», три вызова API с latency
- [`04_telegram_dialog.md`](reports/04_telegram_dialog.md) — диалог с ботом по всем командам

Каждый отчёт начинается со строки режима. Чтобы получить отчёты в режиме REAL, впишите ключи в `local.env` и запустите демо ещё раз.

## Частые ошибки

| Симптом | Причина и решение |
|---|---|
| `Exception: The official Pinecone python package has been renamed…` | Установлен старый `pinecone-client`. Выполните `pip uninstall -y pinecone-client && pip install --upgrade pinecone` |
| `ImportError` у `langchain.vectorstores`, `langchain.embeddings`, `langchain.chat_models` | Это устаревшие пути. Используйте `langchain_pinecone`, `langchain_openai`, `langchain_text_splitters`, `langchain_core` |
| `AttributeError: init` у `pinecone.init(...)` | Старый API. Теперь `from pinecone import Pinecone; pc = Pinecone(api_key=...)` |
| `Размерность индекса … = 768, а EMBEDDING_DIM = 1536` | Индекс создан под другую модель. Удалите его в консоли Pinecone или задайте новый `PINECONE_INDEX_NAME` |
| Конфликты версий после обновления | `pip install --upgrade -r requirements.txt` |
| Поиск сразу после записи в Pinecone пуст | Serverless-индекс обновляется с задержкой в несколько секунд. `check_pinecone.py` повторяет запрос |
| `MOCK_MODE=false, но … не задан` | Впишите ключ или верните `MOCK_MODE=auto` |
