"""Telegram-бот на pyTelegramBotAPI. Без токена — эмуляция диалога через те же хендлеры."""
from __future__ import annotations

import itertools
from types import SimpleNamespace

from config import get_settings, is_mock, print_modes
from rag_agent import RAGAgent, execute_command

TG_LIMIT = 4000  # лимит Telegram — 4096 символов на сообщение

DEMO_SCENARIO = [
    "/start",
    "/add_url https://ru.wikipedia.org/wiki/Земля",
    "/search Что такое Земля?",
    "Меня зовут Иван, я живу в Казани и я люблю астрономию",
    "Какой возраст Земли?",
    "/catfact",
    "Расскажи факт о кошках",
    "/memory",
    "/stats",
]


def _chunks(text: str, size: int = TG_LIMIT):
    for start in range(0, len(text), size):
        yield text[start:start + size]


def register_handlers(bot, agent: RAGAgent) -> None:
    """Один набор хендлеров и для настоящего TeleBot, и для эмулятора."""

    def reply(message, text: str) -> None:
        for part in _chunks(text or "…"):
            bot.reply_to(message, part)

    def run(message) -> None:
        try:
            user = message.from_user
            text = execute_command(agent, user.id, message.text or "", user.first_name or "")
        except Exception as exc:  # бот не должен падать из-за одного сообщения
            text = f"⚠️ Ошибка обработки: {type(exc).__name__}: {exc}"
        reply(message, text)

    @bot.message_handler(commands=["start", "help", "add_url", "search", "catfact", "memory", "stats"])
    def on_command(message):
        if message.text.split()[0].split("@")[0] == "/add_url" and " " in message.text:
            bot.send_chat_action(message.chat.id, "typing")
        run(message)

    @bot.message_handler(content_types=["text"])
    def on_text(message):
        bot.send_chat_action(message.chat.id, "typing")
        run(message)


class EmulatedBot:
    """Минимальная замена telebot.TeleBot: те же декораторы, ответы печатаются в консоль."""

    def __init__(self):
        self.handlers: list[tuple[dict, callable]] = []
        self.transcript: list[tuple[str, str]] = []
        self._ids = itertools.count(1)

    def message_handler(self, commands=None, content_types=None, **_):
        def decorator(func):
            self.handlers.append(({"commands": commands, "content_types": content_types}, func))
            return func
        return decorator

    def reply_to(self, message, text: str):
        self.transcript.append(("bot", text))
        print(f"🤖 Bot: {text}\n")

    def send_message(self, chat_id, text: str, **_):
        self.reply_to(None, text)

    def send_chat_action(self, *_args, **_kwargs):
        pass

    def process(self, text: str, user_id: int = 100500, first_name: str = "Иван") -> None:
        message = SimpleNamespace(
            message_id=next(self._ids), text=text, content_type="text",
            from_user=SimpleNamespace(id=user_id, first_name=first_name),
            chat=SimpleNamespace(id=user_id, type="private"),
        )
        self.transcript.append(("user", text))
        print(f"👤 User: {text}")
        command = text.split()[0].lstrip("/").split("@")[0] if text.startswith("/") else None
        for flt, func in self.handlers:  # порядок как в telebot: первый подходящий
            if flt["commands"] and command in flt["commands"]:
                return func(message)
            if not flt["commands"] and "text" in (flt["content_types"] or ["text"]):
                return func(message)


def run_emulation(agent: RAGAgent | None = None, scenario: list[str] | None = None) -> EmulatedBot:
    agent = agent or RAGAgent()
    bot = EmulatedBot()
    register_handlers(bot, agent)
    for text in scenario or DEMO_SCENARIO:
        bot.process(text)
    return bot


def main() -> None:
    settings = get_settings()
    print_modes(settings)
    agent = RAGAgent(settings)
    if is_mock("telegram", settings):
        print("[MOCK] Telegram-токен не задан — запускаю эмуляцию диалога\n")
        run_emulation(agent)
        return

    import telebot

    bot = telebot.TeleBot(settings.telegram_bot_token)
    register_handlers(bot, agent)
    me = bot.get_me()
    print(f"Бот @{me.username} запущен. Ctrl+C — остановка.")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
