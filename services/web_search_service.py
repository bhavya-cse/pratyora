"""
Pratyora — Trusted-source web search (Tavily).

Searches, in this order:
  1. official government / institutional sites
  2. trusted news outlets (English query)
  3. the same trusted news outlets using the ORIGINAL-language query
     (so Telugu / Hindi / Tamil... outlets can match native wording)
  4. the open web (flagged "general"; never counted as trusted evidence)

Domain restriction uses Tavily's `include_domains` parameter. (The previous
version put `site:a OR site:b` inside the query text, which Tavily does not
treat as an operator, so the "trusted site" searches were not really restricted.)
"""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

TAVILY_URL = "https://api.tavily.com/search"

from services.trusted_sources import (
    FACT_CHECKERS as FACT_CHECK_SITES,
    FACT_CHECK_PATH_HINTS,
    FACT_CHECK_SECTION_SITES,
    OFFICIAL as OFFICIAL_SITES,
    OFFICIAL_SUFFIXES,
    REGIONAL_NEWS_BY_LANGUAGE,
    all_trusted_news,
)

TRUSTED_NEWS_SITES = all_trusted_news()


def _regional_sites(language_code):
    """Outlets that publish in the claim's language (empty if unknown)."""
    return list(REGIONAL_NEWS_BY_LANGUAGE.get(str(language_code or "").lower()[:2], []))


def _host_matches(host, domain):
    return host == domain or host.endswith("." + domain)


def classify_domain(host, url=None):
    """Return 'official', 'fact_checker', 'trusted_news' or 'general' for a hostname.
    If `url` is given, a fact-check page on a trusted outlet (e.g. indiatoday.in/fact-check/...)
    is promoted to 'fact_checker'."""
    host = (host or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return "general"
    if any(_host_matches(host, d) for d in OFFICIAL_SITES) or host.endswith(OFFICIAL_SUFFIXES):
        return "official"
    if any(_host_matches(host, d) for d in FACT_CHECK_SITES):
        return "fact_checker"
    if any(_host_matches(host, d) for d in TRUSTED_NEWS_SITES + FACT_CHECK_SECTION_SITES):
        path = (urlparse(url).path if url else "").lower()
        if any(hint in path for hint in FACT_CHECK_PATH_HINTS):
            return "fact_checker"
        return "trusted_news" if any(_host_matches(host, d) for d in TRUSTED_NEWS_SITES) else "general"
    return "general"


def _search_tavily(api_key, query, max_results=6, include_domains=None):
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = list(include_domains)

    request = Request(
        TAVILY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = []
    for item in data.get("results", []):
        url = str(item.get("url") or "").strip()
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        results.append({
            "title": str(item.get("title") or "Web result"),
            "url": url,
            "content": _clean_search_snippet(item.get("content")),
            "score": item.get("score"),
            "published_date": item.get("published_date"),
        })
    return results


def _clean_search_snippet(value):
    """Remove obvious ad/boilerplate text from a search-result snippet."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    # Drop common ad labels when they appear at the start of a snippet.
    text = re.sub(
        r"^(?:advertisement|sponsored|subscribe|sign up|read more)\s*[:|.-]?\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()
    # Remove common trailing publisher metadata when clearly appended.
    text = re.split(
        r"\s+(?:Published by|Written by|Last updated|Updated on|Read more|Subscribe)\b",
        text, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()
    # Count words in ANY script. Split on whitespace: Python's \w treats Indic vowel signs
    # as non-word characters and would chop Telugu/Hindi/Tamil words into 1-2 letter pieces.
    if len([w for w in text.split() if len(w.strip(".,;:!?()[]\"'|-")) >= 3]) < 6:
        return ""
    return text[:1200]


def _clean_query(text, limit=300):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


_COMMON_QUERY_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are",
    "has", "have", "had", "said", "says", "about", "into", "after", "before",
    "over", "under", "near", "nearly", "first", "phase", "deal", "news", "report",
    "according", "claim", "india", "indian", "us", "usa", "united", "states",
}


def _relevance_terms(text):
    """Tokenize a query/title/snippet, retaining distinctive terms for a soft relevance gate.
    Splits on whitespace/punctuation (not \\w) so Indian-script words stay whole."""
    tokens = re.split(r"[\s,.;:!?()\[\]\"'“”‘’|/\\\-–—]+", str(text or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _COMMON_QUERY_WORDS}



def _is_relevant_search_result(query, title, content):
    """Require meaningful overlap; a shared name alone is not enough."""
    query_terms = _relevance_terms(query)
    result_terms = _relevance_terms(f"{title or ''} {content or ''}")

    if len(query_terms) < 2:
        return False

    overlap = query_terms & result_terms

    # Require at least two distinctive terms.
    # Do not accept a result based on one long name alone.
    return len(overlap) >= 2



def search_web_for_news(claim_text, original_query=None, focused_query=None, language_code=None):
    """
    Search for reporting on a claim.

    claim_text      the English claim sentence (not a whole article)
    original_query  optional short query in the claim's ORIGINAL language
    focused_query   optional short keyword query (names, places, numbers)
    language_code   optional ISO code of the claim's language (bn, te, ta ...). When it is a
                    supported Indian language, outlets in that language are searched first.

    Sources are tagged with `source_category` (official / fact_checker / trusted_news / general).
    A result's presence does not prove it supports the claim; stance is judged
    separately by services.claim_verifier.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "unavailable",
            "message": "Live web search is not configured (TAVILY_API_KEY missing).",
            "answer": "", "results": [], "search_groups": [],
        }

    query_en = _clean_query(claim_text)
    if len(query_en) < 10:
        return {
            "status": "insufficient_input",
            "message": "Provide a longer news claim to search.",
            "answer": "", "results": [], "search_groups": [],
        }

    query_orig = _clean_query(original_query) if original_query else ""
    focused_query = _clean_query(focused_query or "")
    if len(focused_query) < 10:
        focused_query = query_en

    search_jobs = [
        ("official_sources", focused_query, OFFICIAL_SITES),
        ("fact_checkers", focused_query, FACT_CHECK_SITES),
        ("trusted_news", focused_query, TRUSTED_NEWS_SITES),
    ]

    # Search the full claim separately to preserve the event context.
    if query_en.lower() != focused_query.lower():
        search_jobs.append(("trusted_news_full_claim", query_en, TRUSTED_NEWS_SITES))

    if query_orig and query_orig.lower() not in {
        query_en.lower(), focused_query.lower()
    }:
        regional = _regional_sites(language_code)
        if regional:
            # Native-language outlets first, so native wording matches native reporting.
            search_jobs.append(("regional_language_news", query_orig, regional))
        search_jobs.append(
            ("trusted_news_original_language", query_orig, TRUSTED_NEWS_SITES)
        )

    # Fact-checkers: viral claims are settled by them, so search with "fact check" wording,
    # on dedicated sites AND on outlets' fact-check sections, and in the original language.
    search_jobs.append((
        "fact_check_articles", _clean_query(f"fact check {focused_query}"),
        FACT_CHECK_SITES + FACT_CHECK_SECTION_SITES,
    ))
    if query_orig and query_orig.lower() not in {query_en.lower(), focused_query.lower()}:
        search_jobs.append(("fact_checkers_original_language", query_orig, FACT_CHECK_SITES))

    search_jobs.append(("general_web", query_en, None))

    all_results, seen_urls, completed_groups, errors = [], set(), [], []

    # Each search is isolated: one failure must not discard the others.
    for group_name, query, domains in search_jobs:
        try:
            results = _search_tavily(api_key, query, max_results=6, include_domains=domains)
            completed_groups.append(group_name)
            for item in results:
                # Filter obvious off-topic pages before they reach claim_verifier.
                # Judge title and snippet together; do not trust a domain alone.
                if not _is_relevant_search_result(query, item.get("title"), item.get("content")):
                    continue
                key = _canonical_url(item["url"])
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                host = (urlparse(item["url"]).hostname or "").lower()
                item["source_domain"] = host[4:] if host.startswith("www.") else host
                item["search_group"] = group_name
                # Category comes from the actual domain, not from which query found it.
                item["source_category"] = classify_domain(host, item["url"])
                all_results.append(item)
        except HTTPError as exc:
            if exc.code in (401, 403):
                errors.append(f"{group_name}: authentication failed")
            elif exc.code == 429:
                errors.append(f"{group_name}: rate or credit limit reached")
            else:
                errors.append(f"{group_name}: HTTP {exc.code}")
        except (TimeoutError, URLError, OSError, ValueError):
            errors.append(f"{group_name}: search unavailable")

    if all_results:
        return {
            "status": "success",
            "message": "Search completed. Sources still need claim-level verification.",
            "answer": "",
            "results": all_results,
            "search_groups": completed_groups,
            "search_warnings": errors,
        }
    if errors:
        return {
            "status": "error", "message": "; ".join(errors),
            "answer": "", "results": [], "search_groups": completed_groups,
        }
    return {
        "status": "no_results",
        "message": "No useful web sources were returned. This does not mean the claim is false.",
        "answer": "", "results": [], "search_groups": completed_groups,
    }
def _canonical_url(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""

        host = parsed.hostname.lower()
        if host.startswith("www."):
            host = host[4:]

        path = parsed.path.rstrip("/") or "/"

        # Ignore fragments and common tracking parameters.
        from urllib.parse import parse_qsl, urlencode, urlunparse

        ignored = ("utm_", "fbclid", "gclid")
        query_items = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith(ignored)
        ]

        return urlunparse((
            parsed.scheme.lower(),
            host,
            path,
            "",
            urlencode(sorted(query_items)),
            "",
        ))
    except (TypeError, ValueError):
        return ""