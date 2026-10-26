"""ЗАДАНИЕ 3. API-инструмент агента: случайный факт о котах (https://catfact.ninja/fact)."""
from __future__ import annotations

import random
import time

import requests
from langchain_core.tools import tool

from config import Settings, allow_fallback, get_settings, is_mock

# БЫЛО:
# def create_api_request_tool():
#     return "API tool not implemented"

MOCK_CAT_FACTS = [
    "Cats sleep for about 70% of their lives.",
    "A group of cats is called a clowder.",
    "Cats have five toes on their front paws, but only four on the back ones.",
    "A cat's purr vibrates at a frequency of 25 to 150 hertz.",
    "Cats can rotate their ears 180 degrees.",
    "The oldest known pet cat was found in a 9,500-year-old grave on Cyprus.",
    "A cat's nose print is unique, much like a human's fingerprint.",
    "Cats cannot taste sweetness.",
    "Adult cats meow mostly to communicate with humans, not with other cats.",
    "A cat can jump up to six times its own length.",
]

RETRY_ATTEMPTS = 2
RETRY_BACKOFF_S = 0.5


def mock_cat_fact(reason: str = "MOCK_MODE") -> dict:
    fact = random.choice(MOCK_CAT_FACTS)
    print(f"[API] MOCK catfact ({reason})")
    return {"ok": True, "fact": fact, "length": len(fact), "source": "mock",
            "latency_ms": 0, "note": reason}


def _validate(payload) -> tuple[str, int]:
    """Схема ответа: {"fact": str, "length": int}."""
    if not isinstance(payload, dict):
        raise ValueError("ответ API — не JSON-объект")
    fact, length = payload.get("fact"), payload.get("length")
    if not isinstance(fact, str) or not fact.strip():
        raise ValueError("в ответе нет строкового поля 'fact'")
    if not isinstance(length, int):
        raise ValueError("в ответе нет целого поля 'length'")
    return fact.strip(), length


def call_cat_api(settings: Settings | None = None,
                 attempts: int = RETRY_ATTEMPTS, backoff: float = RETRY_BACKOFF_S) -> dict:
    """Реальный GET с ретраями. Никогда не бросает исключения — только {ok: False, error}."""
    s = settings or get_settings()
    url = s.cat_fact_api_url
    last_error = "неизвестная ошибка"
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            resp = requests.get(url, timeout=s.http_timeout,
                                headers={"Accept": "application/json"})
            latency = round((time.perf_counter() - started) * 1000)
            print(f"[API] GET {url} → {resp.status_code} ({latency} ms)")
            resp.raise_for_status()
            fact, length = _validate(resp.json())
            return {"ok": True, "fact": fact, "length": length, "source": url,
                    "latency_ms": latency}
        except requests.Timeout:
            last_error = f"таймаут {s.http_timeout} с"
        except requests.ConnectionError as exc:
            last_error = f"нет соединения: {type(exc).__name__}"
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            last_error = f"HTTP {code}"
            if isinstance(code, int) and code < 500 and code != 429:
                break  # клиентская ошибка — повтор не поможет
        except ValueError as exc:  # сюда же попадает невалидный JSON
            return {"ok": False, "error": f"некорректный ответ API: {exc}", "source": url}
        print(f"[API] попытка {attempt}/{attempts} не удалась: {last_error}")
        if attempt < attempts:
            time.sleep(backoff * attempt)
    return {"ok": False, "error": last_error, "source": url}


def get_cat_fact(settings: Settings | None = None) -> dict:
    """Факт с учётом режима: MOCK_MODE=true → эмуляция; сбой сети → эмуляция (если разрешено)."""
    s = settings or get_settings()
    if is_mock("http", s):
        return mock_cat_fact("MOCK_MODE=true")
    result = call_cat_api(s)
    if not result["ok"] and allow_fallback(s):
        return {**mock_cat_fact(f"API недоступен: {result['error']}"),
                "fallback_from_error": result["error"]}
    return result


# СТАЛО:
def create_api_request_tool():
    """Возвращает LangChain-инструмент, который агент вызывает для запроса к API."""

    @tool
    def get_random_cat_fact() -> dict:
        """Получить случайный факт о кошках из открытого API catfact.ninja.
        Используй, когда пользователь просит факт о котах/кошках (cat fact)."""
        return get_cat_fact()

    return get_random_cat_fact
