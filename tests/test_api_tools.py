from unittest.mock import MagicMock

import pytest
import requests

import api_tools
from api_tools import call_cat_api, create_api_request_tool


def _response(status=200, payload=None, json_error=False):
    resp = MagicMock()
    resp.status_code = status
    if json_error:
        resp.json.side_effect = ValueError("Expecting value")
    else:
        resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(api_tools.time, "sleep", lambda *_: None)


def test_success(monkeypatch):
    monkeypatch.setattr(api_tools.requests, "get",
                        lambda *a, **k: _response(payload={"fact": "Cats purr.", "length": 10}))
    res = call_cat_api()
    assert res["ok"] and res["fact"] == "Cats purr." and res["length"] == 10
    assert res["source"] == "https://catfact.ninja/fact" and "latency_ms" in res


def test_timeout_returns_error_after_retries(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.Timeout()

    monkeypatch.setattr(api_tools.requests, "get", boom)
    res = call_cat_api()
    assert res["ok"] is False and "таймаут" in res["error"]
    assert len(calls) == 2  # 2 попытки


def test_invalid_json(monkeypatch):
    monkeypatch.setattr(api_tools.requests, "get", lambda *a, **k: _response(json_error=True))
    res = call_cat_api()
    assert res["ok"] is False and "некорректный" in res["error"]


def test_wrong_schema(monkeypatch):
    monkeypatch.setattr(api_tools.requests, "get",
                        lambda *a, **k: _response(payload={"fact": 42}))
    assert call_cat_api()["ok"] is False


def test_http_404_no_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(api_tools.requests, "get",
                        lambda *a, **k: calls.append(1) or _response(status=404))
    res = call_cat_api()
    assert res["ok"] is False and res["error"] == "HTTP 404" and len(calls) == 1


def test_tool_in_mock_mode():
    tool = create_api_request_tool()
    assert tool.name == "get_random_cat_fact"
    res = tool.invoke({})
    assert res["ok"] and res["source"] == "mock" and res["fact"] in api_tools.MOCK_CAT_FACTS


def test_tool_falls_back_to_mock_on_network_error(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "auto")
    monkeypatch.setattr(api_tools.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    res = create_api_request_tool().invoke({})
    assert res["ok"] and res["source"] == "mock" and "fallback_from_error" in res
