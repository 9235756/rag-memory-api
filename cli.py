"""Консольный чат с теми же командами, что и Telegram-бот."""
from __future__ import annotations

from config import print_modes
from rag_agent import HELP_TEXT, RAGAgent, execute_command

CLI_USER_ID = "cli-user"


def main() -> None:
    print_modes()
    agent = RAGAgent()
    print("\nКонсольный режим RAG-агента. Выход: /exit или Ctrl+C\n")
    print(HELP_TEXT + "\n")
    while True:
        try:
            text = input("👤 Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи!")
            break
        if not text:
            continue
        if text.lower() in {"/exit", "/quit", "выход"}:
            print("До встречи!")
            break
        try:
            reply = execute_command(agent, CLI_USER_ID, text)
        except Exception as exc:
            reply = f"⚠️ Ошибка: {type(exc).__name__}: {exc}"
        print(f"🤖 Бот: {reply}\n")


if __name__ == "__main__":
    main()
