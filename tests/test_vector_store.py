from langchain_core.documents import Document

from embeddings import MockEmbeddings
from vector_store import VectorStoreManager

DOCS = [
    Document(page_content="Pinecone — векторная база данных для хранения эмбеддингов.",
             metadata={"title": "pinecone"}),
    Document(page_content="Борщ варят из свёклы, капусты и картофеля.", metadata={"title": "борщ"}),
    Document(page_content="Telegram-бот отвечает пользователю в чате.", metadata={"title": "бот"}),
]
IDS = ["a", "b", "c"]


def test_mock_mode_selected(store):
    assert store.mode == "MOCK"
    assert isinstance(store.embeddings, MockEmbeddings)


def test_add_and_search_returns_relevant_first(store):
    store.add_documents(DOCS, namespace="t", ids=IDS)
    results = store.similarity_search("Что такое векторная база данных?", k=3, namespace="t")
    assert results, "поиск должен что-то найти"
    assert results[0][0].metadata["title"] == "pinecone"
    assert all(score >= store.settings.score_threshold for _, score in results)


def test_upsert_does_not_duplicate(store):
    store.add_documents(DOCS, namespace="t", ids=IDS)
    store.add_documents(DOCS, namespace="t", ids=IDS)
    assert store.stats()["namespaces"]["t"]["vector_count"] == 3


def test_persistence_between_instances(tmp_path):
    path = tmp_path / "s.json"
    VectorStoreManager(mock_path=path).add_documents(DOCS, namespace="t", ids=IDS)
    again = VectorStoreManager(mock_path=path)
    assert again.stats()["total_vector_count"] == 3


def test_filter_by_metadata(store):
    docs = [Document(page_content="Пользователя зовут Иван", metadata={"user_id": "1"}),
            Document(page_content="Пользователя зовут Пётр", metadata={"user_id": "2"})]
    store.add_documents(docs, namespace="m", ids=["1", "2"])
    hits = store.similarity_search("как зовут", namespace="m", filter={"user_id": "2"},
                                   apply_threshold=False)
    assert [d.page_content for d, _ in hits] == ["Пользователя зовут Пётр"]


def test_health_check(store):
    hc = store.health_check()
    assert hc["ok"] and hc["mode"] == "MOCK"
