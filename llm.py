"""LLM: ChatOpenAI в реальном режиме, экстрактивный MockLLM в эмуляции."""
from __future__ import annotations

import math
import re

from langchain_core.messages import AIMessage

from config import Settings, get_settings, is_mock
from embeddings import tokenize

CONTEXT_MARK = "КОНТЕКСТ:"
QUESTION_MARK = "ВОПРОС:"
NO_DATA = "В базе знаний нет данных по этому вопросу."


class MockLLM:
    """Без сети и ключей: выбирает 2–3 предложения из контекста, ближе всего к вопросу."""

    model_name = "mock-extractive"

    def invoke(self, messages) -> AIMessage:
        prompt = messages[-1].content if isinstance(messages, list) else str(messages)
        context, question = self._split(prompt)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", context) if len(s.strip()) > 15]
        q_terms = set(tokenize(question))
        sent_terms = [set(tokenize(sent)) for sent in sentences]
        # IDF по предложениям контекста: слово «Земля» есть почти везде и весит мало,
        # а «возраст» встречается редко — именно оно отличает нужное предложение
        n = len(sentences)
        idf = {t: math.log(1 + n / (1 + sum(t in st for st in sent_terms))) for t in q_terms}
        scored = []
        for pos, (sent, terms) in enumerate(zip(sentences, sent_terms)):
            weight = sum(idf[t] for t in q_terms & terms)
            if weight:
                scored.append((round(weight, 6), -pos, sent))
        if not scored:
            return AIMessage(content=NO_DATA)
        best = sorted(scored, reverse=True)[:3]
        best.sort(key=lambda item: -item[1])  # в исходном порядке текста
        return AIMessage(content="По данным базы знаний: " + " ".join(s for _, _, s in best))

    @staticmethod
    def _split(prompt: str) -> tuple[str, str]:
        if CONTEXT_MARK in prompt and QUESTION_MARK in prompt:
            ctx = prompt.split(CONTEXT_MARK, 1)[1].split(QUESTION_MARK, 1)[0]
            question = prompt.rsplit(QUESTION_MARK, 1)[1]
            ctx = re.sub(r"^\[\d+\][^\n]*\n", "", ctx, flags=re.M)  # заголовки фрагментов
            return ctx, question
        return "", prompt


def get_llm(settings: Settings | None = None):
    s = settings or get_settings()
    if is_mock("openai", s):
        return MockLLM()
    from langchain_openai import ChatOpenAI

    kwargs = {"model": s.openai_model, "temperature": 0.2, "api_key": s.openai_api_key}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return ChatOpenAI(**kwargs)
