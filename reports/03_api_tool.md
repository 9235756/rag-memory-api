Режим: REAL

> Cat API не требует ключа — запросы шли в https://catfact.ninja/fact

# Задание 3. API-инструмент: случайный факт о котах

_Сгенерировано: 2026-10-26 22:41:04_

## БЫЛО (заглушка)

```python
def create_api_request_tool():
    return "API tool not implemented"
```

## СТАЛО (реальный GET-запрос к открытому API)

```python
def create_api_request_tool():
    """Возвращает LangChain-инструмент, который агент вызывает для запроса к API."""

    @tool
    def get_random_cat_fact() -> dict:
        """Получить случайный факт о кошках из открытого API catfact.ninja.
        Используй, когда пользователь просит факт о котах/кошках (cat fact)."""
        return get_cat_fact()

    return get_random_cat_fact
```

Полная реализация — `api_tools.py`: `call_cat_api()` (GET, таймаут, 2 попытки с backoff,
проверка статуса и JSON-схемы `fact: str, length: int`, обработка `Timeout`, `ConnectionError`,
`HTTPError`, невалидного JSON) и `get_cat_fact()` (эмуляция в MOCK_MODE=true или при сбое сети).

## Три вызова инструмента

| # | ok | latency, мс | источник | факт |
|---|---|---|---|---|
| 1 | ✅ | 331 | https://catfact.ninja/fact | The Maine Coon is 4 to 5 times larger than the Singapura, the smallest breed of cat. |
| 2 | ✅ | 338 | https://catfact.ninja/fact | The oldest cat on record was Crème Puff from Austin, Texas, who lived from 1967 to August 6, 2005, three days after her 38th birthday. A cat typically can live up to 20 years, which is equivalent to about 96 human years. |
| 3 | ✅ | 310 | https://catfact.ninja/fact | Cats have about 130,000 hairs per square inch (20,155 hairs per square centimeter). |

## Лог (видно обращение агента к API)

```text
--- вызов 1: get_random_cat_fact.invoke({})
[API] GET https://catfact.ninja/fact → 200 (331 ms)
{'ok': True, 'fact': 'The Maine Coon is 4 to 5 times larger than the Singapura, the smallest breed of cat.', 'length': 84, 'source': 'https://catfact.ninja/fact', 'latency_ms': 331}
--- вызов 2: get_random_cat_fact.invoke({})
[API] GET https://catfact.ninja/fact → 200 (338 ms)
{'ok': True, 'fact': 'The oldest cat on record was Crème Puff from Austin, Texas, who lived from 1967 to August 6, 2005, three days after her 38th birthday. A cat typically can live up to 20 years, which is equivalent to about 96 human years.', 'length': 220, 'source': 'https://catfact.ninja/fact', 'latency_ms': 338}
--- вызов 3: get_random_cat_fact.invoke({})
[API] GET https://catfact.ninja/fact → 200 (310 ms)
{'ok': True, 'fact': 'Cats have about 130,000 hairs per square inch (20,155 hairs per square centimeter).', 'length': 83, 'source': 'https://catfact.ninja/fact', 'latency_ms': 310}
```
