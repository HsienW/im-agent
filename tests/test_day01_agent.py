import pytest

from app.tools.message_search import search_messages


def test_search_existing_message():
    results = search_messages("登入")

    assert len(results) == 1
    assert results[0]["id"] == "m001"


def test_search_no_match():
    assert search_messages("不存在的問題") == []


def test_search_empty_keyword():
    with pytest.raises(ValueError):
        search_messages("")
