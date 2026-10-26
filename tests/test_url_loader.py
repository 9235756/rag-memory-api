import pytest

from url_loader import SAMPLE_PAGE, UrlLoadError, fetch_url, ingest_url, parse_html, split_text


def test_parse_html_removes_noise():
    html = """<html><head><title>Тест</title><style>.x{color:red}</style>
    <script>alert('x')</script></head><body><nav>Меню</nav><main>
    <p>Полезный текст.</p></main><footer>Подвал</footer></body></html>"""
    title, text = parse_html(html)
    assert title == "Тест"
    assert "Полезный текст." in text
    for junk in ("alert", "color:red", "Меню", "Подвал"):
        assert junk not in text


def test_split_text_long_gives_many_chunks():
    text = "\n\n".join(f"Абзац номер {i}. " + "Слово " * 60 for i in range(20))
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_fetch_url_validates_scheme():
    with pytest.raises(UrlLoadError):
        fetch_url("ftp://example.com/file")


def test_fetch_url_in_mock_uses_offline_page():
    html, offline = fetch_url("https://ru.wikipedia.org/wiki/Земля")
    assert offline and "Земля" in html


def test_ingest_sample_page(store):
    html = SAMPLE_PAGE.read_text(encoding="utf-8")
    report = ingest_url("https://example.com/earth", store, html=html)
    assert report["chunks"] > 1
    assert report["embedding_dim"] == store.settings.embedding_dim
    assert report["ids"][0].endswith("-0") and len(report["ids"][0].split("-")[0]) == 16
    # повторная загрузка той же страницы — upsert, без дублей
    ingest_url("https://example.com/earth", store, html=html)
    assert store.stats()["namespaces"]["knowledge"]["vector_count"] == report["chunks"]
    hits = store.similarity_search("Сколько лет Земле? возраст", namespace="knowledge")
    assert hits and "4,54" in hits[0][0].page_content
