"""RAG-агент: память о пользователе, маршрутизация (Cat API / RAG), ответ с источниками."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from api_tools import create_api_request_tool
from config import Settings, get_settings, mode_label
from llm import CONTEXT_MARK, NO_DATA, QUESTION_MARK, get_llm
from memory import UserMemory
from url_loader import UrlLoadError, ingest_url
from vector_store import VectorStoreManager

# Явный список словоформ: «кот\w*» поймал бы «который», «котёл» и т. п.
CAT_RE = re.compile(
    r"\b(кот|кота|коту|котом|коте|коты|котов|котам|котами|котах|котэ|котик\w*|котен\w*|котён\w*"
    r"|кошк\w*|кошек|кошач\w*|cats?|cat\s*facts?|kitt(?:y|en)s?)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "Ты — ассистент с базой знаний. Отвечай по-русски, кратко и только на основе "
    "КОНТЕКСТА и фактов о пользователе. Если в контексте нет ответа, так и скажи: "
    f"«{NO_DATA}». Не выдумывай."
)

HELP_TEXT = """Команды:
/add_url <ссылка> — загрузить страницу в базу знаний
/search <запрос> — семантический поиск по базе (со score)
/catfact — случайный факт о котах (открытый API)
/memory — что я помню о вас
/stats — статистика хранилища и режимы работы
/help — эта справка
Обычный текст — вопрос к базе знаний. Напишите «меня зовут …», «я живу в …» — запомню."""


@dataclass
class AgentReply:
    text: str
    route: str  # "rag" | "cat_api" | "memory"
    sources: list[str] = field(default_factory=list)
    saved_facts: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


class RAGAgent:
    def __init__(self, settings: Settings | None = None, store: VectorStoreManager | None = None,
                 llm=None):
        self.settings = settings or get_settings()
        self.store = store or VectorStoreManager(self.settings)
        self.memory = UserMemory(self.store)
        self.llm = llm or get_llm(self.settings)
        self.cat_tool = create_api_request_tool()

    # ---------- публичные методы ----------
    def add_url(self, url: str) -> dict:
        return ingest_url(url, self.store)

    def search(self, query: str, k: int | None = None):
        return self.store.similarity_search(query, k=k, namespace=self.settings.namespace_kb)

    def remember(self, user_id, text: str) -> list[dict]:
        return self.memory.remember(user_id, text)

    def memory_facts(self, user_id) -> list[str]:
        return self.memory.list_facts(user_id)

    def cat_fact(self) -> dict:
        return self.cat_tool.invoke({})

    def stats(self) -> dict:
        return {"modes": {svc: mode_label(svc, self.settings)
                          for svc in ("openai", "pinecone", "telegram", "http")},
                "index": self.settings.pinecone_index_name,
                "store": self.store.stats()}

    @staticmethod
    def is_cat_question(text: str) -> bool:
        return bool(CAT_RE.search(text))

    def answer(self, user_id, text: str) -> AgentReply:
        # 1. память: вытаскиваем факты о пользователе из любого сообщения
        saved = [f["text"] for f in self.remember(user_id, text)]

        # 2. маршрутизация
        if self.is_cat_question(text):
            return self.answer_cat(saved)
        if saved and "?" not in text:
            return AgentReply(text="Запомнил: " + "; ".join(saved) + ".", route="memory",
                              saved_facts=saved)
        return self._answer_rag(user_id, text, saved)

    # ---------- внутреннее ----------
    def answer_cat(self, saved: list[str] | None = None) -> AgentReply:
        res = self.cat_fact()
        if not res.get("ok"):
            return AgentReply(text=f"Не удалось получить факт о котах: {res.get('error')}",
                              route="cat_api", saved_facts=saved or [])
        src = "эмуляция (mock)" if res["source"] == "mock" else f"{res['source']}, {res['latency_ms']} мс"
        return AgentReply(text=f"🐱 Факт о котах: {res['fact']}\n\nИсточник: {src}",
                          route="cat_api", sources=[res["source"]], saved_facts=saved or [])

    def _answer_rag(self, user_id, question: str, saved: list[str]) -> AgentReply:
        kb_hits = self.search(question)
        mem_hits = self.memory.recall(user_id, question)
        if not kb_hits and not mem_hits:
            return AgentReply(text=f"{NO_DATA} Добавьте источник: /add_url <ссылка>",
                              route="rag", saved_facts=saved)

        parts = []
        for i, (doc, score) in enumerate(kb_hits, 1):
            parts.append(f"[{i}] {doc.metadata.get('title', '')} (score {score})\n{doc.page_content}")
        facts = "\n".join(f"- {doc.page_content}." for doc, _ in mem_hits)
        prompt = (f"ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n{facts or '- нет'}\n\n"
                  f"{CONTEXT_MARK}\n" + "\n\n".join(parts) + "\n"
                  + "\n".join(f"{doc.page_content}." for doc, _ in mem_hits)
                  + f"\n\n{QUESTION_MARK} {question}")
        reply = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        text = reply.content if hasattr(reply, "content") else str(reply)

        sources = []
        for doc, _ in kb_hits:
            label = f"{doc.metadata.get('title', 'без названия')} — {doc.metadata.get('source', '')}"
            if label not in sources:
                sources.append(label)
        if mem_hits:
            sources.append("память пользователя")
        if sources and not text.startswith(NO_DATA):
            text += "\n\nИсточники:\n" + "\n".join(f"• {s}" for s in sources)
        return AgentReply(text=text, route="rag", sources=sources, saved_facts=saved)


def _format_search(results) -> str:
    if not results:
        return "Ничего не найдено (выше порога SCORE_THRESHOLD)."
    lines = []
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc.page_content[:220].replace("\n", " ")
        lines.append(f"{i}. score={score} · {doc.metadata.get('title', '')} "
                     f"(чанк {doc.metadata.get('chunk_index', '?')})\n   {snippet}…")
    return "\n".join(lines)


def execute_command(agent: RAGAgent, user_id, text: str, first_name: str = "") -> str:
    """Общая логика команд для Telegram-бота и консоли."""
    text = (text or "").strip()
    command, _, arg = text.partition(" ")
    command = command.split("@")[0].lower()  # /help@MyBot → /help
    arg = arg.strip()

    if command == "/start":
        hello = f"Привет, {first_name}! " if first_name else "Привет! "
        return hello + "Я RAG-агент с памятью на Pinecone.\n\n" + HELP_TEXT
    if command == "/help":
        return HELP_TEXT
    if command == "/add_url":
        if not arg:
            return "Укажите ссылку: /add_url https://…"
        try:
            r = agent.add_url(arg)
        except UrlLoadError as exc:
            return f"❌ {exc}"
        note = "\n⚠️ Загружена оффлайн-страница data/sample_page.html (MOCK: сайт недоступен или MOCK_MODE=true)." if r["offline_page"] else ""
        return (f"✅ Загружено: «{r['title']}»\nСимволов: {r['chars']}, чанков: {r['chunks']}, "
                f"размерность эмбеддингов: {r['embedding_dim']}\n"
                f"Сохранено в {r['store']} (namespace={r['namespace']}){note}")
    if command == "/search":
        if not arg:
            return "Укажите запрос: /search <текст>"
        return _format_search(agent.search(arg))
    if command == "/catfact":
        return agent.answer_cat().text
    if command == "/memory":
        facts = agent.memory_facts(user_id)
        if not facts:
            return "Пока ничего о вас не знаю. Напишите, например: «Меня зовут Иван»."
        return "Что я о вас помню:\n" + "\n".join(f"• {f}" for f in facts)
    if command == "/stats":
        st = agent.stats()
        modes = ", ".join(f"{k}: {v}" for k, v in st["modes"].items())
        ns = "\n".join(f"• {name}: {v['vector_count']}"
                       for name, v in st["store"]["namespaces"].items()) or "• пусто"
        return (f"Индекс: {st['index']}\nРежимы — {modes}\n"
                f"Векторов всего: {st['store']['total_vector_count']}\nПо namespace:\n{ns}")
    if command.startswith("/"):
        return f"Неизвестная команда {command}. /help — список команд."
    return agent.answer(user_id, text).text
