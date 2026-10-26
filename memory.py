"""Долговременная память о пользователе: эвристики → эмбеддинги → namespace user-memory."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from langchain_core.documents import Document

_END = r"(?=[.,!?;]|\s+и\s+|$)"  # конец факта: знак препинания, союз «и» или конец строки

# (тип, регулярка, шаблон текста факта, один ли такой факт у пользователя)
FACT_PATTERNS = [
    ("name", re.compile(r"\bменя\s+зовут\s+([A-Za-zА-Яа-яЁё\-]+)", re.I),
     "Пользователя зовут {0}", True),
    ("age", re.compile(r"\bмне\s+(\d{1,3})\s+(?:год|года|лет)\b", re.I),
     "Пользователю {0} лет", True),
    ("city", re.compile(r"\bя\s+живу\s+(?:в|во)\s+(.+?)" + _END, re.I),
     "Пользователь живёт в {0}", True),
    ("work", re.compile(r"\bя\s+работаю\s+(.+?)" + _END, re.I),
     "Пользователь работает {0}", True),
    ("likes", re.compile(r"\bя\s+(?:люблю|предпочитаю|обожаю)\s+(.+?)" + _END, re.I),
     "Пользователь любит {0}", False),
    ("goal", re.compile(r"\bмоя\s+цель\s*[—–:\-]?\s*(.+?)(?=[.!?]|$)", re.I),
     "Цель пользователя: {0}", True),
]


def extract_facts(text: str) -> list[dict]:
    facts = []
    for kind, pattern, template, single in FACT_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).strip(" .,!?«»\"'")
            if not value:
                continue
            if kind == "name":
                value = value.capitalize()
            facts.append({"type": kind, "value": value, "single": single,
                          "text": template.format(value)})
            if single:
                break
    return facts


class UserMemory:
    def __init__(self, store):
        self.store = store
        self.namespace = store.settings.namespace_memory

    @staticmethod
    def _fact_id(user_id: str, fact: dict) -> str:
        # Однозначные факты (имя, возраст, город…) перезаписываются; «люблю» — копятся
        if fact["single"]:
            return f"{user_id}-{fact['type']}"
        digest = hashlib.sha256(fact["value"].lower().encode("utf-8")).hexdigest()[:8]
        return f"{user_id}-{fact['type']}-{digest}"

    def remember(self, user_id, text: str) -> list[dict]:
        user_id = str(user_id)
        facts = extract_facts(text)
        if not facts:
            return []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        docs = [Document(page_content=f["text"],
                         metadata={"user_id": user_id, "fact_type": f["type"],
                                   "value": f["value"], "saved_at": now,
                                   "source": "память пользователя"})
                for f in facts]
        ids = [self._fact_id(user_id, f) for f in facts]
        self.store.add_documents(docs, namespace=self.namespace, ids=ids)
        return facts

    def recall(self, user_id, query: str, k: int = 3) -> list[tuple[Document, float]]:
        return self.store.similarity_search(query, k=k, namespace=self.namespace,
                                            filter={"user_id": str(user_id)})

    def list_facts(self, user_id, limit: int = 50) -> list[str]:
        hits = self.store.similarity_search("пользователь", k=limit, namespace=self.namespace,
                                            filter={"user_id": str(user_id)},
                                            apply_threshold=False)
        return sorted({doc.page_content for doc, _ in hits})
