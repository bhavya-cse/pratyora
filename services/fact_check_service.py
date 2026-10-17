import re
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search_fact_checks(claim_text, language_code=None):
    api_key = os.getenv("FACT_CHECK_API_KEY", "").strip()

    if not api_key:
        return {
            "status": "unavailable",
            "message": "Fact-check API key is not configured.",
            "results": [],
        }

    query_params = {
        "query": (claim_text or "").strip()[:1000],
        "key": api_key,
        "pageSize": 5,
    }

    # Use the detected language when a valid ISO two-letter code exists.
    if language_code and re.fullmatch(
        r"[a-zA-Z]{2}", str(language_code)
    ):
        query_params["languageCode"] = str(language_code).lower()

    params = urlencode(query_params)

    request = Request(
        f"{API_URL}?{params}",
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []

        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher") or {}

                results.append({
                    "claim": claim.get("text", ""),
                    "publisher": publisher.get(
                        "name", "Unknown publisher"
                    ),
                    "title": review.get(
                        "title", "Fact-check article"
                    ),
                    "rating": review.get(
                        "textualRating", "Rating not provided"
                    ),
                    "url": review.get("url", ""),
                    "review_date": review.get("reviewDate", ""),
                })

        if not results:
            return {
                "status": "no_match",
                "message": "No matching published fact-check was found.",
                "results": [],
            }

        return {
            "status": "matched",
            "message": "Published fact-check review(s) found.",
            "results": results[:10],
        }

    except HTTPError as exc:
        if exc.code in (400, 401, 403):
            message = (
                "Google rejected the request. Check the API key "
                "and ensure Fact Check Tools API is enabled."
            )
        else:
            message = (
                f"Fact-check service returned HTTP {exc.code}."
            )

        return {
            "status": "error",
            "message": message,
            "results": [],
        }

    except (TimeoutError, URLError, OSError, ValueError):
        return {
            "status": "error",
            "message": (
                "Could not contact Google Fact Check. Try again later."
            ),
            "results": [],
        }