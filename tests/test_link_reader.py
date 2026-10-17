"""
Pratyora - tests for services/link_reader.py

These tests never touch the internet: private addresses are rejected before any request is
made, and everything else uses mocks.

Run:  pytest tests/test_link_reader.py -v
"""

import pytest

from services import link_reader as lr
from services.link_reader import LinkReadError, extract_article, read_news_link


# ---------------------------------------------------------------------------
# URL safety (SSRF)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "https://user:pass@example.com/",
        "https://example.com:8080/",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
    ],
)
def test_blocked_targets_are_rejected(url):
    with pytest.raises(LinkReadError) as exc:
        read_news_link(url)
    assert exc.value.status == 400


@pytest.mark.parametrize("value", ["", "   ", "has space.com/a b", "a" * 2500])
def test_invalid_input_is_rejected(value):
    with pytest.raises(LinkReadError) as exc:
        read_news_link(value)
    assert exc.value.code == "INVALID_URL"


def test_schemeless_link_gets_https():
    assert lr.normalize_url("example.com/story") == "https://example.com/story"


def test_redirect_to_private_address_is_blocked(monkeypatch):
    def fake_assert(url):
        if "127.0.0.1" in url:
            raise LinkReadError("blocked", "URL_BLOCKED", 400)

    class Redirect:
        status_code = 302
        is_redirect = True
        headers = {"Location": "http://127.0.0.1/secret"}

        def close(self):
            pass

    monkeypatch.setattr(lr, "assert_public_url", fake_assert)
    monkeypatch.setattr(lr.requests.Session, "get", lambda *a, **k: Redirect())

    with pytest.raises(LinkReadError) as exc:
        read_news_link("https://news.example.com/story")
    assert exc.value.code == "URL_BLOCKED"


# ---------------------------------------------------------------------------
# Social / video links
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/someone/status/123",
        "https://twitter.com/someone/status/123",
        "https://www.facebook.com/post/1",
        "https://m.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://t.me/channel/5",
        "https://www.instagram.com/p/abc/",
    ],
)
def test_social_and_video_links_are_rejected_up_front(url):
    with pytest.raises(LinkReadError) as exc:
        read_news_link(url)
    assert exc.value.code == "UNSUPPORTED_SOURCE"


def test_similar_looking_news_domain_is_not_blocked():
    assert lr._is_unsupported_host("notx.com") is False
    assert lr._is_unsupported_host("www.thehindu.com") is False
    assert lr._is_unsupported_host("m.youtube.com") is True


# ---------------------------------------------------------------------------
# Title cleaning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Rates held steady by central bank | NDTV.com", "Rates held steady by central bank"),
        ("NDTV | Rates held steady by central bank", "Rates held steady by central bank"),
        ("Rates held steady by central bank - The Hindu", "Rates held steady by central bank"),
        # A dash inside a real headline must survive.
        ("Modi - Xi meeting held in Delhi | NDTV", "Modi - Xi meeting held in Delhi"),
        ("Plain headline without any site name", "Plain headline without any site name"),
        ("", ""),
    ],
)
def test_clean_title(raw, expected):
    assert lr._clean_title(raw) == expected


# ---------------------------------------------------------------------------
# Extraction and page handling
# ---------------------------------------------------------------------------
ARTICLE_HTML = (
    "<html><head><title>Rates held | The Hindu</title></head><body><nav>Menu Home Sports</nav>"
    "<h1>Rates held steady</h1>"
    "<p>The central bank announced on Wednesday that benchmark interest rates will remain "
    "unchanged after its policy review meeting in Mumbai this week.</p>"
    "<p>Officials said further details would be published after consultation with local "
    "authorities and affected groups over the coming month.</p></body></html>"
)


def test_extract_article_with_fallback_reader(monkeypatch):
    monkeypatch.setattr(lr, "trafilatura", None)
    title, text = extract_article(ARTICLE_HTML)
    assert title == "Rates held steady"
    assert "benchmark interest rates" in text
    assert "Menu Home Sports" not in text


def test_long_article_is_capped():
    html = "<html><body>" + "".join(
        "<p>%s</p>" % ("word " * 30) for _ in range(1500)
    ) + "</body></html>"
    _, text = extract_article(html)
    assert len(text) <= lr.MAX_ARTICLE_CHARS


def test_page_without_article_text_is_reported(monkeypatch):
    monkeypatch.setattr(lr, "assert_public_url", lambda u: None)
    monkeypatch.setattr(lr, "_download", lambda u: (u, "<html><body><nav>Menu</nav></body></html>"))
    with pytest.raises(LinkReadError) as exc:
        read_news_link("https://news.example.com/")
    assert exc.value.code == "NO_ARTICLE_TEXT"


def test_successful_read(monkeypatch):
    monkeypatch.setattr(lr, "trafilatura", None)
    monkeypatch.setattr(lr, "assert_public_url", lambda u: None)
    monkeypatch.setattr(lr, "_download", lambda u: ("https://news.example.com/final", ARTICLE_HTML))
    page = read_news_link("news.example.com/story")
    assert page["url"] == "https://news.example.com/final"
    assert page["title"] == "Rates held steady"
    assert len(page["text"]) >= lr.MIN_ARTICLE_CHARS


# ---------------------------------------------------------------------------
# Picking a checkable sentence from a feature article
# ---------------------------------------------------------------------------
def test_pick_checkable_sentence_prefers_figures_english():
    text = (
        "Kerala should learn from Odisha, writes our correspondent after a long visit to the state.\n\n"
        "The Odisha government has set aside Rs 1,319 crore for sports in this year's budget.\n\n"
        "Athletes say the food is good and the rooms are comfortable during the whole camp."
    )
    assert "1,319 crore" in lr.pick_checkable_sentence(text)


def test_pick_checkable_sentence_works_for_malayalam():
    text = (
        "\u0d15\u0d3e\u0d2f\u0d3f\u0d15\u0d24\u0d3e\u0d30\u0d19\u0d4d\u0d19\u0d33\u0d4d\u200d\u0d15\u0d4d\u0d15\u0d4d "
        "\u0d05\u0d1f\u0d3f\u0d38\u0d4d\u0d25\u0d3e\u0d28 \u0d38\u0d57\u0d15\u0d30\u0d4d\u0d2f\u0d2e\u0d4a\u0d30\u0d41\u0d15\u0d4d\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d24\u0d3f\u0d32\u0d4d\u200d "
        "\u0d15\u0d47\u0d30\u0d33\u0d02 \u0d2a\u0d3f\u0d28\u0d4d\u0d28\u0d3f\u0d32\u0d3e\u0d23\u0d4d \u0d0e\u0d28\u0d4d\u0d28\u0d4d \u0d2a\u0d31\u0d2f\u0d41\u0d28\u0d4d\u0d28\u0d41 \u0d24\u0d3e\u0d30\u0d19\u0d4d\u0d19\u0d33\u0d4d\u200d.\n"
        "\u0d15\u0d3e\u0d2f\u0d3f\u0d15\u0d30\u0d02\u0d17\u0d24\u0d4d\u0d24\u0d3f\u0d28\u0d4d\u0d31\u0d46 \u0d35\u0d3f\u0d15\u0d38\u0d28\u0d24\u0d4d\u0d24\u0d3f\u0d28\u0d3e\u0d2f\u0d3f "
        "1319 \u0d15\u0d4b\u0d1f\u0d3f \u0d30\u0d42\u0d2a\u0d2f\u0d3e\u0d23\u0d4d \u0d07\u0d24\u0d4d\u0d24\u0d35\u0d23\u0d24\u0d4d\u0d24\u0d46 \u0d2c\u0d1c\u0d31\u0d4d\u0d31\u0d3f\u0d32\u0d4d\u200d "
        "\u0d12\u0d21\u0d40\u0d37 \u0d38\u0d30\u0d4d\u200d\u0d15\u0d4d\u0d15\u0d3e\u0d30\u0d4d\u200d \u0d28\u0d40\u0d15\u0d4d\u0d15\u0d3f\u0d35\u0d1a\u0d4d\u0d1a\u0d3f\u0d30\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d28\u0d4d\u0d28\u0d24\u0d4d."
    )
    assert "1319" in lr.pick_checkable_sentence(text)


def test_pick_checkable_sentence_returns_none_without_figures():
    text = "Everyone agrees that the new stadium looks wonderful and that the town is proud of it."
    assert lr.pick_checkable_sentence(text) is None
    assert lr.pick_checkable_sentence("") is None


def test_add_context_adds_subject_from_headline():
    title = "AC rooms, air travel: Kerala should learn, Odisha cares"
    body = (
        "Odisha runs hostels in Bhubaneswar. Kerala hostels are far behind. Odisha pays for travel. "
        "Kerala athletes wait for allowances. Kerala teams carry their own beds. Odisha gives kits worth Rs 8,000 to children every year."
    )
    out = lr.add_context("Odisha gives kits worth Rs 8,000 to children every year.", title, body)
    assert "Kerala" in out


def test_add_context_keeps_sentence_when_nothing_to_add():
    assert lr.add_context("A plain sentence with 500 people.", "Unrelated headline words", "nothing relevant") == (
        "A plain sentence with 500 people."
    )
    assert lr.add_context(None, "x", "y") is None


def test_add_context_malayalam_inflected_forms():
    title = "\u0d12\u0d21\u0d40\u0d37\u0d2f\u0d41\u0d1f\u0d46 \u0d15\u0d30\u0d41\u0d24\u0d32\u0d4d"
    stem = "\u0d12\u0d21\u0d40\u0d37"
    body = " ".join([stem + "\u0d2f\u0d3f\u0d32\u0d4d", stem + "\u0d2f\u0d3f\u0d32\u0d46", stem + "\u0d2f\u0d41\u0d1f\u0d46"])
    out = lr.add_context("1319 \u0d15\u0d4b\u0d1f\u0d3f \u0d28\u0d40\u0d15\u0d4d\u0d15\u0d3f\u0d35\u0d1a\u0d4d\u0d1a\u0d41.", title, body)
    assert "(" in out and stem in out