"""
Pratyora — read a news article from a pasted link.

read_news_link(url) opens the page and returns its headline and article text so the normal
text pipeline can analyze and verify it.

Because users control the URL, the fetch is hardened against server-side request forgery (SSRF):
  * only http/https on ports 80/443, no credentials in the URL
  * the host must resolve ONLY to public internet addresses (no localhost, private networks,
    link-local / cloud-metadata addresses)
  * redirects are followed by hand and every hop is validated again
  * a connect/read timeout, an overall deadline and a response-size cap
  * only HTML / text responses are accepted

Article extraction uses `trafilatura` when installed (best quality) and falls back to a small
built-in HTML reader otherwise.
"""

import ipaddress
import importlib
import logging
import re
import socket
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

try:  # optional, but strongly recommended: pip install trafilatura
    trafilatura = importlib.import_module("trafilatura")
except ImportError:  # pragma: no cover
    trafilatura = None

logger = logging.getLogger("PratyoraLinkReader")

MAX_URL_LENGTH = 2000
MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024        # 2 MB of HTML is plenty for any article
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
TOTAL_DEADLINE_SECONDS = 20
MAX_ARTICLE_CHARS = 20000                   # same ceiling as the Text tab
MIN_ARTICLE_CHARS = 120                     # less than this is a menu / landing page, not an article
ALLOWED_PORTS = {None, 80, 443}

# Social posts and video sites are not news articles: their text is loaded by JavaScript or sits
# behind a login, so reading them only ever fails. Say so up front instead of after a slow fetch.
UNSUPPORTED_HOSTS = (
    "x.com", "twitter.com", "facebook.com", "fb.com", "fb.watch", "instagram.com", "threads.net",
    "tiktok.com", "youtube.com", "youtu.be", "t.me", "telegram.me", "whatsapp.com", "wa.me",
    "snapchat.com", "reddit.com", "linkedin.com",
)
# Many news sites (Cloudflare, Akamai) refuse anything that announces itself as a bot, so the
# request looks like an ordinary desktop browser opening a page.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


class LinkReadError(Exception):
    """A link that cannot be read. `message` is safe to show to the user."""

    def __init__(self, message, code, status=422):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


# --------------------------------------------------------------------------- #
# URL safety
# --------------------------------------------------------------------------- #
def normalize_url(raw):
    """Accept 'site.com/story' as well as 'https://site.com/story'."""
    value = str(raw or "").strip()
    if not value or len(value) > MAX_URL_LENGTH or re.search(r"\s", value):
        raise LinkReadError("Please paste a full link to a news page.", "INVALID_URL", 400)
    if not re.match(r"^[a-z][a-z0-9+.\-]*://", value, re.I):
        value = "https://" + value
    return value


def _is_unsupported_host(host):
    host = (host or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in UNSUPPORTED_HOSTS)


def _is_public_ip(ip_text):
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def assert_public_url(url):
    """Raise LinkReadError unless `url` is a plain http(s) link to a public host."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise LinkReadError("That does not look like a valid link.", "INVALID_URL", 400)

    if parsed.scheme not in ("http", "https"):
        raise LinkReadError("Only http and https links can be read.", "INVALID_URL", 400)
    if not parsed.hostname:
        raise LinkReadError("That does not look like a valid link.", "INVALID_URL", 400)
    if parsed.username or parsed.password:
        raise LinkReadError("Links with a username or password are not allowed.", "URL_BLOCKED", 400)
    if port not in ALLOWED_PORTS:
        raise LinkReadError("Only standard web addresses (ports 80 and 443) can be read.", "URL_BLOCKED", 400)

    host = parsed.hostname
    if "." not in host and ":" not in host:
        raise LinkReadError("That does not look like a public website address.", "INVALID_URL", 400)

    try:
        infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise LinkReadError("That website address could not be found. Check the link and try again.",
                            "FETCH_FAILED", 422)
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(_is_public_ip(a) for a in addresses):
        logger.warning("Blocked non-public link target: %s -> %s", host, sorted(addresses))
        raise LinkReadError("That link points to a private or internal address, so it can't be opened.",
                            "URL_BLOCKED", 400)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download(url):
    """Fetch the page by hand-following redirects. Returns (final_url, html_text)."""
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
    current = url
    session = requests.Session()
    session.max_redirects = 0
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.6",
        "Cache-Control": "max-age=0",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    for _ in range(MAX_REDIRECTS + 1):
        assert_public_url(current)
        if time.monotonic() > deadline:
            raise LinkReadError("The website took too long to respond.", "FETCH_TIMEOUT", 504)
        try:
            response = session.get(current, headers=headers, stream=True, allow_redirects=False,
                                   timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except requests.exceptions.Timeout:
            raise LinkReadError("The website took too long to respond.", "FETCH_TIMEOUT", 504)
        except requests.exceptions.SSLError:
            raise LinkReadError("The website's security certificate could not be verified.", "FETCH_FAILED", 422)
        except requests.exceptions.RequestException:
            raise LinkReadError("The page could not be opened. Check the link and try again.", "FETCH_FAILED", 422)

        try:
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise LinkReadError("The page redirected nowhere.", "FETCH_FAILED", 422)
                current = urljoin(current, location)
                continue

            if response.status_code in (401, 403, 429, 451):
                logger.warning("Link blocked by %s: HTTP %d", urlparse(current).hostname, response.status_code)
            if response.status_code == 429:
                raise LinkReadError(
                    "That website is limiting requests right now. Wait a minute and try again, "
                    "or copy the article text into the Text tab.", "FETCH_BLOCKED", 422)
            if response.status_code in (401, 403, 451):
                raise LinkReadError(
                    "This website did not allow Pratyora to read the page (it may block automated access, "
                    "need a login, or sit behind a paywall). Copy the article text into the Text tab instead.",
                    "FETCH_BLOCKED", 422)
            if response.status_code == 404 or response.status_code == 410:
                raise LinkReadError("That page was not found (HTTP %d). Check the link." % response.status_code,
                                    "FETCH_FAILED", 422)
            if response.status_code >= 400:
                raise LinkReadError("The website returned an error (HTTP %d)." % response.status_code,
                                    "FETCH_FAILED", 422)

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(t in content_type for t in ("html", "xml", "text/plain")):
                raise LinkReadError(
                    "That link is not a web page (it looks like a file or media). "
                    "Paste a link to the news article itself.", "NOT_A_PAGE", 422)

            chunks, total = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_DOWNLOAD_BYTES or time.monotonic() > deadline:
                    break
            body = b"".join(chunks)[:MAX_DOWNLOAD_BYTES]
            return current, _decode(body, response)
        finally:
            response.close()

    raise LinkReadError("The link redirected too many times.", "FETCH_FAILED", 422)


def _decode(body, response):
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        pass
    encoding = None
    match = re.search(r"charset=([\w\-]+)", response.headers.get("Content-Type", ""), re.I)
    if match:
        encoding = match.group(1)
    else:
        meta = re.search(rb"<meta[^>]+charset=[\"']?([\w\-]+)", body[:4096], re.I)
        if meta:
            encoding = meta.group(1).decode("ascii", "ignore")
    try:
        return body.decode(encoding or "latin-1", errors="replace")
    except LookupError:
        return body.decode("latin-1", errors="replace")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg",
              "iframe", "button", "figure", "figcaption", "template"}


class _FallbackReader(HTMLParser):
    """Tiny article reader: <title>, og:title, <h1> and the longer <p> paragraphs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.h1 = ""
        self.paragraphs = []
        self._skip = 0
        self._in_title = False
        self._in_h1 = False
        self._in_p = False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and attrs.get("property") == "og:title":
            self.og_title = (attrs.get("content") or "").strip()
        elif self._skip == 0 and tag == "h1" and not self.h1:
            self._in_h1 = True
            self._buf = []
        elif self._skip == 0 and tag == "p":
            self._in_p = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "h1" and self._in_h1:
            self._in_h1 = False
            self.h1 = " ".join("".join(self._buf).split())
        elif tag == "p" and self._in_p:
            self._in_p = False
            text = " ".join("".join(self._buf).split())
            if len(text.split()) >= 8:
                self.paragraphs.append(text)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip == 0 and (self._in_p or self._in_h1):
            self._buf.append(data)


def _fallback_extract(html):
    reader = _FallbackReader()
    try:
        reader.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML must not crash the request
        logger.info("Fallback HTML reader hit malformed markup.")
    title = reader.og_title or reader.h1 or " ".join(reader.title.split())
    return title, "\n\n".join(reader.paragraphs)


def _clean_title(title):
    """
    Remove the site name from a page title: 'Headline | NDTV.com' -> 'Headline'.
    Only short trailing/leading parts (up to 3 words) are treated as a site name, so a real
    headline that contains a dash ('Modi - Xi meeting held in Delhi | NDTV') is kept whole.
    """
    title = " ".join(str(title or "").split())
    if not title:
        return title

    def is_site_name(part):
        return len(part.split()) <= 3

    # 1) "|" is almost always a site-name separator, so it may sit at either end.
    parts = [p.strip() for p in title.split("|") if p.strip()]
    if len(parts) > 1:
        while len(parts) > 1 and is_site_name(parts[-1]):
            parts.pop()
        while len(parts) > 1 and is_site_name(parts[0]):
            parts.pop(0)
        title = " | ".join(parts)

    # 2) A dash/colon-style separator is only trusted for a short site name at the END.
    dash_parts = [p.strip() for p in re.split(r"\s+[\-–—]\s+", title) if p.strip()]
    if len(dash_parts) > 1 and is_site_name(dash_parts[-1]) and len(" ".join(dash_parts[:-1]).split()) >= 3:
        title = " - ".join(dash_parts[:-1])

    return title


def extract_article(html):
    """Return (title, text) from page HTML."""
    title, text = "", ""
    if trafilatura is not None:
        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=False,
                                       favor_precision=True) or ""
            meta = trafilatura.extract_metadata(html)
            title = (getattr(meta, "title", "") or "") if meta else ""
        except Exception as exc:  # noqa: BLE001
            logger.info("trafilatura failed, using fallback reader: %s", exc)
            text, title = "", ""
    if len(text) < MIN_ARTICLE_CHARS:
        fb_title, fb_text = _fallback_extract(html)
        if len(fb_text) > len(text):
            text = fb_text
        title = title or fb_title
    elif not title:
        title = _fallback_extract(html)[0]

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > MAX_ARTICLE_CHARS:
        cut = text[:MAX_ARTICLE_CHARS]
        text = cut[:cut.rfind("\n\n")] if "\n\n" in cut[MAX_ARTICLE_CHARS // 2:] else cut
    return _clean_title(title), text


# --------------------------------------------------------------------------- #
# Choosing a checkable sentence
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u0964])\s+|\n+")
_NUMBER = re.compile(r"\d[\d,.]*")


def pick_checkable_sentence(text, min_chars=40, max_chars=320):
    """
    Return the sentence of an article that makes the most specific, checkable claim, or None.

    Feature and opinion headlines ("AC rooms, air travel: Kerala should learn") assert nothing a
    search can confirm, but the article body usually does ("Odisha set aside Rs 1,319 crore").
    Figures are the most language-independent sign of such a claim, so sentences are scored by the
    numbers in them (big numbers count more than a lone "2" or "3"). Works for any script.
    """
    best, best_score = None, 0
    for raw in _SENTENCE_SPLIT.split(text or ""):
        sentence = " ".join(raw.split())
        if not (min_chars <= len(sentence) <= max_chars):
            continue
        if "http" in sentence or sentence.startswith("["):
            continue
        score = 0
        for number in _NUMBER.findall(sentence):
            digits = re.sub(r"\D", "", number)
            if digits:
                score += 2 if len(digits) >= 3 else 1
        if score > best_score:
            best, best_score = sentence, score
    return best


_CONTEXT_STOPWORDS = {
    "this", "that", "with", "from", "have", "should", "would", "could", "about", "their", "there",
    "which", "while", "after", "before", "were", "been", "will", "what", "when", "where", "into",
    "your", "more", "than", "they", "them", "also", "some", "such", "only", "over", "under",
}
# Split on spaces, digits and punctuation rather than matching \w: in Indian scripts the vowel signs
# are combining marks that \w does not cover, which would cut every word into pieces.
_WORD_SPLIT = re.compile(r"[\s\d,.;:!?()\[\]\"'\u201c\u201d\u2018\u2019\-\u2013\u2014|/%&*+=<>@#_]+")


def _words(text):
    return [w for w in _WORD_SPLIT.split(text or "") if w]


def add_context(sentence, title, body, max_terms=2, stem_len=4):
    """
    A sentence picked out of an article often lacks its subject ("Branded kits worth Rs 8,000 are
    given to hostel children every year": where?). Searching that alone returns unrelated pages.
    The subject is usually named in the headline and repeated through the article, so find the
    headline words that recur most often in the body and append them as context:
    "... every year. (Kerala, Odisha)". Words are matched by their first few letters so inflected
    forms in Indian languages (Odisha / Odishayil / Odishayude) still count. Works for any script.
    """
    if not sentence:
        return sentence
    body_words = _words(body)
    sentence_lower = sentence.lower()
    counts = {}
    for word in _words(title):
        if len(word) < stem_len or word.lower() in _CONTEXT_STOPWORDS:
            continue
        stem = word[:stem_len].lower()
        if stem in counts or stem in sentence_lower:
            continue
        matches = [w for w in body_words if len(w) >= stem_len and w[:stem_len].lower() == stem]
        if len(matches) >= 3:
            counts[stem] = (len(matches), word)
    ranked = sorted(counts.values(), key=lambda item: -item[0])[:max_terms]
    if not ranked:
        return sentence
    return "%s (%s)" % (sentence.rstrip(), ", ".join(word for _, word in ranked))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def read_news_link(raw_url):
    """
    Open a pasted news link and return {"url", "title", "text"}.
    Raises LinkReadError (with a user-safe message) when the link can't be used.
    """
    url = normalize_url(raw_url)
    try:
        host = urlparse(url).hostname
    except ValueError:
        host = None
    if _is_unsupported_host(host):
        raise LinkReadError(
            "Social media posts and video pages can't be read as news articles. Copy the text of the "
            "post into the Text tab, or upload a screenshot instead.",
            "UNSUPPORTED_SOURCE", 422)
    assert_public_url(url)
    final_url, html = _download(url)
    title, text = extract_article(html)

    if len(text) < MIN_ARTICLE_CHARS:
        raise LinkReadError(
            "Pratyora could not find article text on that page. It may need a login, load its text with "
            "JavaScript, or not be a news article. Copy the article text into the Text tab instead.",
            "NO_ARTICLE_TEXT", 422)
    return {"url": final_url, "title": title, "text": text}