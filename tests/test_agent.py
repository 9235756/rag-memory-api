import pytest

from memory import extract_facts
from rag_agent import RAGAgent, execute_command
from url_loader import SAMPLE_PAGE, ingest_url


@pytest.fixture
def agent(store):
    return RAGAgent(store=store)


@pytest.mark.parametrize("text", ["Расскажи факт о кошках", "cat fact please", "Хочу про котов"])
def test_cat_question_goes_to_api(agent, text):
    reply = agent.answer("u1", text)
    assert reply.route == "cat_api"
    assert "Факт о котах" in reply.text


def test_not_cat_question():
    assert not RAGAgent.is_cat_question("Который час? Котёл сломался")


def test_name_saved_and_shown_in_memory(agent):
    reply = agent.answer("42", "Меня зовут Иван")
    assert reply.route == "memory" and "Пользователя зовут Иван" in reply.saved_facts
    assert "Пользователя зовут Иван" in execute_command(agent, "42", "/memory")
    # память изолирована по user_id
    assert "Пользователя зовут Иван" not in execute_command(agent, "other", "/memory")


def test_memory_used_in_answer(agent):
    agent.answer("42", "Меня зовут Иван")
    reply = agent.answer("42", "Как меня зовут?")
    assert "Иван" in reply.text and "память пользователя" in reply.sources


def test_extract_facts_variants():
    facts = {f["type"]: f["value"] for f in extract_facts(
        "Меня зовут анна, мне 30 лет, я живу в Казани и я люблю джаз. Моя цель — выучить Python.")}
    assert facts == {"name": "Анна", "age": "30", "city": "Казани", "likes": "джаз",
                     "goal": "выучить Python"}


def test_rag_answer_with_sources(agent, store):
    ingest_url("https://example.com/earth", store, html=SAMPLE_PAGE.read_text(encoding="utf-8"))
    reply = agent.answer("u1", "Каков возраст Земли?")
    assert reply.route == "rag"
    assert "4,54" in reply.text
    assert "Источники" in reply.text and "https://example.com/earth" in reply.text


def test_no_data_answer(agent):
    reply = agent.answer("u1", "Какая столица Перу?")
    assert "нет данных" in reply.text


def test_commands(agent):
    assert "/add_url" in execute_command(agent, "u", "/help")
    assert "Привет, Иван" in execute_command(agent, "u", "/start", first_name="Иван")
    assert "чанков" in execute_command(agent, "u", "/add_url https://ru.wikipedia.org/wiki/Земля")
    assert "score=" in execute_command(agent, "u", "/search Что такое Земля?")
    assert "Векторов всего" in execute_command(agent, "u", "/stats")
    assert "Неизвестная команда" in execute_command(agent, "u", "/foo")
