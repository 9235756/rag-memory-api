"""Прогон всех трёх заданий и сохранение доказательств в reports/."""
from __future__ import annotations

import contextlib
import inspect
import io
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api_tools  # noqa: E402
from config import REPORTS_DIR, get_settings, mode_label  # noqa: E402
from rag_agent import RAGAgent, execute_command  # noqa: E402
from scripts.check_pinecone import run_check  # noqa: E402
from telegram_bot import DEMO_SCENARIO, run_emulation  # noqa: E402

DEMO_URL = "https://ru.wikipedia.org/wiki/Земля"
DEMO_QUERY = "Что такое Земля?"


class Tee(io.StringIO):
    """Пишет и в консоль, и в буфер — вывод виден сразу и попадает в отчёт."""

    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


@contextlib.contextmanager
def capture():
    buf = Tee()
    with contextlib.redirect_stdout(buf):
        yield buf


def mode_line(*services: str) -> str:
    labels = {svc: mode_label(svc) for svc in services}
    if all(v == "REAL" for v in labels.values()):
        head = "Режим: REAL"
    elif all(v == "MOCK" for v in labels.values()):
        head = "Режим: MOCK (эмуляция, ключи не заданы)"
    else:
        head = "Режим: MIXED (часть сервисов эмулируется — ключи не заданы)"
    detail = ", ".join(f"{k}={v}" for k, v in labels.items())
    return f"{head}\n\n> Сервисы: {detail}"


def write_report(name: str, title: str, mode: str, body: str) -> Path:
    path = REPORTS_DIR / name
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(f"{mode}\n\n# {title}\n\n_Сгенерировано: {stamp}_\n\n{body}\n",
                    encoding="utf-8")
    return path


def fence(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


# ---------------------------------------------------------------- задания
def task1() -> tuple[bool, str]:
    with capture() as out:
        ok = run_check()
    write_report("01_pinecone_check.md", "Задание 1. Подключение к Pinecone и тестовый запрос",
                 mode_line("pinecone", "openai"),
                 "Команда: `python scripts/check_pinecone.py`\n\n" + fence(out.getvalue()))
    return ok, mode_label("pinecone")


def task2(agent: RAGAgent) -> tuple[bool, str]:
    with capture() as out:
        print(f"$ /add_url {DEMO_URL}")
        report = agent.add_url(DEMO_URL)
        print(f"\n$ /search {DEMO_QUERY}")
        results = agent.search(DEMO_QUERY)
        for i, (doc, score) in enumerate(results, 1):
            print(f"#{i} score={score} chunk={doc.metadata.get('chunk_index')}: "
                  f"{doc.page_content[:150].replace(chr(10), ' ')}…")
    samples = "\n\n".join(f"**Чанк {i}** ({len(c)} символов):\n\n" + fence(c)
                          for i, c in enumerate(report["sample_chunks"]))
    rows = "\n".join(f"| {i} | {score} | {doc.metadata.get('chunk_index')} | "
                     f"{doc.page_content[:120].replace(chr(10), ' ').replace('|', '/')}… |"
                     for i, (doc, score) in enumerate(results, 1))
    source = ("оффлайн-страница `data/sample_page.html` (сайт недоступен или MOCK_MODE=true)"
              if report["offline_page"] else "живая страница из интернета")
    body = f"""## Загрузка: `/add_url {DEMO_URL}`

| Параметр | Значение |
|---|---|
| Источник | {source} |
| Заголовок | {report['title']} |
| Символов после парсинга | {report['chars']} |
| Чанков (CHUNK_SIZE={agent.settings.chunk_size}, overlap={agent.settings.chunk_overlap}) | {report['chunks']} |
| Размерность эмбеддингов | {report['embedding_dim']} |
| Хранилище / namespace | {report['store']} / `{report['namespace']}` |
| Пример id (sha256(url)[:16]-i) | `{report['ids'][0]}` |

## Пример двух чанков

{samples}

## Поиск: `/search {DEMO_QUERY}`

| # | score | чанк | текст |
|---|---|---|---|
{rows or '| — | — | — | ничего не найдено |'}

## Лог конвейера

{fence(out.getvalue())}"""
    write_report("02_url_ingest.md", "Задание 2. Обработка URL → чанки → эмбеддинги → векторная база",
                 mode_line("pinecone", "openai", "http"), body)
    ok = report["chunks"] > 0 and bool(results)
    return ok, mode_label("pinecone")


def task3(agent: RAGAgent) -> tuple[bool, str]:
    src = inspect.getsource(api_tools)
    was = src.split("# БЫЛО:", 1)[1].split("\n\n", 1)[0].strip()
    was = "\n".join(line.removeprefix("# ") for line in was.splitlines())
    now = inspect.getsource(api_tools.create_api_request_tool)
    calls, sources = [], []
    with capture() as out:
        for i in range(1, 4):
            print(f"--- вызов {i}: get_random_cat_fact.invoke({{}})")
            res = agent.cat_fact()
            print(res)
            calls.append(res)
            sources.append(res.get("source"))
    rows = "\n".join(
        f"| {i} | {'✅' if r.get('ok') else '❌'} | {r.get('latency_ms', '—')} | {r.get('source')} | "
        f"{(r.get('fact') or r.get('error', '')).replace('|', '/')} |"
        for i, r in enumerate(calls, 1))
    real = all(s != "mock" for s in sources)
    body = f"""## БЫЛО (заглушка)

{fence(was, 'python')}

## СТАЛО (реальный GET-запрос к открытому API)

{fence(now, 'python')}

Полная реализация — `api_tools.py`: `call_cat_api()` (GET, таймаут, 2 попытки с backoff,
проверка статуса и JSON-схемы `fact: str, length: int`, обработка `Timeout`, `ConnectionError`,
`HTTPError`, невалидного JSON) и `get_cat_fact()` (эмуляция в MOCK_MODE=true или при сбое сети).

## Три вызова инструмента

| # | ok | latency, мс | источник | факт |
|---|---|---|---|---|
{rows}

## Лог (видно обращение агента к API)

{fence(out.getvalue())}"""
    mode = ("Режим: REAL\n\n> Cat API не требует ключа — запросы шли в https://catfact.ninja/fact"
            if real else "Режим: MOCK (эмуляция, сеть недоступна или MOCK_MODE=true)")
    write_report("03_api_tool.md", "Задание 3. API-инструмент: случайный факт о котах", mode, body)
    return all(r.get("ok") for r in calls), "REAL" if real else "MOCK"


def task4(agent: RAGAgent) -> tuple[bool, str]:
    with capture() as out:
        print("[MOCK] Telegram-токен не задан — запускаю эмуляцию диалога\n"
              if mode_label("telegram") == "MOCK" else
              "Токен задан, но для отчёта диалог прогоняется через эмулятор с теми же хендлерами\n")
        bot = run_emulation(agent, DEMO_SCENARIO)
    lines = []
    for who, text in bot.transcript:
        prefix = "👤 **User:**" if who == "user" else "🤖 **Bot:**"
        quoted = "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
        lines.append(f"{prefix}\n\n{quoted}\n")
    replies = [t for w, t in bot.transcript if w == "bot"]
    ok = len(replies) == len(DEMO_SCENARIO) and not any(t.startswith("⚠️") for t in replies)
    body = ("Сценарий прогоняется через `register_handlers()` — те же хендлеры, что у настоящего "
            "`telebot.TeleBot`, но с эмулированными объектами `message`.\n\n"
            "## Диалог\n\n" + "\n".join(lines) + "\n## Сырой вывод\n\n" + fence(out.getvalue()))
    write_report("04_telegram_dialog.md", "Telegram-бот: диалог со всеми командами",
                 mode_line("telegram", "pinecone", "openai"), body)
    return ok, mode_label("telegram")


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    settings = get_settings()
    agent = RAGAgent(settings)

    results = []
    for title, func, args, file in [
        ("1. Pinecone: подключение и тестовый запрос", task1, (), "01_pinecone_check.md"),
        ("2. URL → чанки → эмбеддинги → база + поиск", task2, (agent,), "02_url_ingest.md"),
        ("3. API-инструмент (catfact.ninja)", task3, (agent,), "03_api_tool.md"),
        ("Telegram-бот (эмуляция диалога)", task4, (agent,), "04_telegram_dialog.md"),
    ]:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        try:
            ok, mode = func(*args)
        except Exception as exc:
            print(f"❌ {type(exc).__name__}: {exc}")
            ok, mode = False, "ERROR"
        results.append((title, ok, mode, file))

    table = "\n".join(f"| {t} | {'✅' if ok else '❌'} | {m} | [{f}]({f}) |"
                      for t, ok, m, f in results)
    modes = ", ".join(f"{s}={mode_label(s)}" for s in ("openai", "pinecone", "telegram", "http"))
    summary = f"""# VPg04 — сводка прогона

Дата/время запуска: **{started}**

Режимы сервисов: {modes} (MOCK_MODE={settings.mock_mode})

| Задание | Статус | Режим | Файл-доказательство |
|---|---|---|---|
{table}

MOCK — сервис эмулируется, потому что ключ в `local.env` не задан. Впишите ключи и
перезапустите `python scripts/run_demo.py` — отчёты пересоздадутся в режиме REAL.
"""
    (REPORTS_DIR / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\n{summary}")
    return 0 if all(ok for _, ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
