"""
Pratyora — AI-Powered News Credibility Analysis Platform
Main Flask Application and API Endpoints.
"""

import copy
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

try:  # Flask-Limiter protects the link endpoint (it makes the server fetch pages and run the NLI model).
    from flask_limiter import Limiter  # type: ignore[import-not-found]
    try:
        from flask_limiter import get_remote_address  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - older flask-limiter releases expose this via util.
        from flask_limiter.util import get_remote_address  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - the app still runs without it, just unthrottled
    Limiter = None
    get_remote_address = None
from services.claim_verifier import apply_verification, llm_configured, nli_available, verify_claim
from services.image_processor import ImageProcessor
from services.text_analyzer import TextAnalyzer
from services.evidence_assessment import assess_evidence
from services.link_reader import LinkReadError, add_context, pick_checkable_sentence, read_news_link

try:  # Present once services/text_cleaning.py v2 is installed; harmless fallback otherwise.
    from services.text_cleaning import strip_native_news_boilerplate
except ImportError:  # pragma: no cover
    def strip_native_news_boilerplate(text: str) -> str:
        return text or ""

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PratyoraApp")

app = Flask(__name__)
# Allow the frontend from Flask itself or any local dev server (Live Server on
# 5500/5501, localhost or 127.0.0.1) to call the API.
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                r"http://127\.0\.0\.1:\d+",
                r"http://localhost:\d+",
            ]
        }
    },
)

# Security & Upload Configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "pratyora-dev-secret-key-observation-lens")
# Global request body cap: 10 MB (screenshots are limited to 5 MB; text and link requests are tiny)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))

# Rate limiting (per client IP). Override with e.g. LINK_RATE_LIMIT="20 per minute" in .env.
# For several server workers, point RATELIMIT_STORAGE_URI at Redis instead of the in-memory default.
LINK_RATE_LIMIT = os.getenv("LINK_RATE_LIMIT", "10 per minute;100 per day")
if Limiter is not None:
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
    )

    def link_rate_limit(func):
        # Tests (app.config["TESTING"] = True) are exempt so they never hit the limit.
        return limiter.limit(LINK_RATE_LIMIT, exempt_when=lambda: app.config.get("TESTING", False))(func)
else:
    logger.warning("flask-limiter is not installed; /api/analyze-link is NOT rate limited.")

    def link_rate_limit(func):
        return func

# Initialize services
text_analyzer = TextAnalyzer.get_instance()
image_processor = ImageProcessor()


@app.route("/")
def index():
    """Render the Pratyora single-page web application."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Returns application status, machine learning model status,
    and external tool availability (Tesseract OCR).
    """
    model_available = text_analyzer.is_available()
    tesseract_available = image_processor.is_ocr_available()

    status = "healthy" if model_available else "degraded"

    return jsonify({
        "status": status,
        "application": "Pratyora",
        "tagline": "See Beyond the Claim. Seek the Evidence.",
        "version": "1.0.0",
        "model": {
            "available": model_available,
            "type": text_analyzer.metadata.get("model_type", "TF-IDF + LogisticRegression") if model_available else None,
            "trained_at": text_analyzer.metadata.get("trained_at_utc") if model_available else None,
            "evaluation_metrics": text_analyzer.metadata.get("metrics") if model_available else None,
            "instructions": "Run 'python train_model.py' to generate model binary" if not model_available else None,
        },
        "system_dependencies": {
            "tesseract_ocr": {
                "available": tesseract_available,
                "binary_path": image_processor.tesseract_cmd if tesseract_available else None,
                "install_hint": "brew install tesseract" if not tesseract_available else None,
            },
        },
        "verification": {
            "local_nli_model_installed": nli_available(),
            "optional_llm_verifier": llm_configured(),
            "web_search": bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "fact_check_api": bool(os.getenv("FACT_CHECK_API_KEY", "").strip()),
        },
    }), 200


# Longest claim text sent to the search services (OCR/transcripts can be very long).
MAX_VERIFICATION_CHARS = 3000

def _run_safely(step_name, func, fallback, error_sink=None):
    """Run one verification step. A search outage must never break the analysis itself.
    If `error_sink` (a list) is given, the error message is appended so it can be shown on screen."""
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 - external services can fail in many ways
        logger.exception("Verification step failed: %s", step_name)
        if error_sink is not None:
            error_sink.append(f"{type(exc).__name__}: {exc}")
        return fallback


def verify_result(result, claim_text, language_code=None, use_translation=True):
    """
    Check the submitted claim against trusted sources and merge the verdict into
    `result` (in place). Used for text, screenshots/photos and news links, so every input
    type gets the same check, in any language.

    Flow: extract + translate the claim -> search Google Fact Check and trusted
    official/news sites -> judge each source (supports / contradicts / unrelated) ->
    score. When trusted sources back or contradict the claim, that evidence score
    becomes the headline percentage. The TF-IDF model score is always preserved as
    model_*_percentage. A claim no trusted source confirms is never shown as "Likely Real".
    """
    claim_text = (claim_text or "").strip()[:MAX_VERIFICATION_CHARS]
    language_code = language_code or result.get("detected_language")

    # Keep the original model percentages before anything adjusts them.
    result["model_real_percentage"] = result.get("real_percentage")
    result["model_fake_percentage"] = result.get("fake_percentage")

    if not claim_text:
        return result

    errors = []
    outcome = _run_safely(
        "claim verification",
        lambda: verify_claim(
            claim_text,
            english_text=result.get("translated_text") if use_translation else None,
            language_code=language_code,
        ),
        None,
        error_sink=errors,
    )
    if outcome is None:
        # Make the failure visible on the page instead of silently showing model-only numbers.
        result["verification"] = {
            "performed": False,
            "status": "error",
            "label": None,
            "summary": "The trusted-source check could not run, so nothing here has been verified.",
            "action": {
                "type": "caution",
                "headline": "Trusted-source check could not run",
                "advice": [
                    "The percentages below are only the wording model's estimate. They are NOT verified.",
                    "Look at the terminal running app.py for the line 'Verification step failed'.",
                ],
                "correct_info": "",
                "share_text": "",
            },
            "error": errors[0] if errors else "",
            "supporting_sources": [],
            "contradicting_sources": [],
            "related_sources": [],
            "limitations": [],
        }
        result["score_basis"] = "model_only"
        result["model_scores_reliable"] = False
        if result.get("success") and result.get("label") in ("Likely Real", "Likely Fake"):
            result["label"] = "Unverified"
            result["confidence"] = "Low"
        return result

    verification, web_search, fact_check = outcome
    result["web_search"] = web_search
    result["fact_check"] = fact_check

    # Legacy keyword-based assessment: still feeds the "web evidence" panel.
    result["evidence_assessment"] = _run_safely(
        "evidence assessment",
        lambda: assess_evidence(claim_text, web_search, fact_check, result),
        {"status": "search_unavailable", "message": "Evidence assessment is currently unavailable."},
    )

    apply_verification(result, verification)
    return result


def _verify_without_model(raw_text, error_result):
    """
    The model-based analysis could not run (for example the text could not be
    translated), but trusted-source verification does not need it: the claim
    extractor translates by itself. Build a minimal result and verify anyway.
    """
    result = {
        "success": True,
        "label": "Unverified",
        "confidence": "Low",
        "fake_percentage": 50.0,
        "real_percentage": 50.0,
        "insufficient_evidence": True,
        "analyzed_characters": len(raw_text),
        "analyzed_words": len(raw_text.split()),
        "text_snippet": raw_text[:160],
        "detected_language": error_result.get("detected_language"),
        "detected_language_name": error_result.get("detected_language_name"),
        "translated": False,
        "warning": "The wording model could not analyze this text; only the trusted-source check was run.",
    }
    verify_result(result, raw_text)
    return result


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """
    Analyzes submitted news text or article snippet.
    Request body: JSON with 'text' field.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request content type must be application/json.",
            "error_code": "INVALID_CONTENT_TYPE",
        }), 400

    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({
            "success": False,
            "error": "Missing 'text' field in JSON request body.",
            "error_code": "MISSING_FIELD",
        }), 400

    raw_text = payload.get("text", "")
    result = text_analyzer.analyze(raw_text)

    if result.get("success"):
        verify_result(result, raw_text)
    elif result.get("error_code") == "TRANSLATION_FAILED" and isinstance(raw_text, str):
        result = _verify_without_model(raw_text.strip(), result)

    status_code = 200 if result.get("success") else 400
    if result.get("error_code") == "MODEL_UNAVAILABLE":
        status_code = 503

    return jsonify(result), status_code


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    """
    Extracts text from an uploaded image or screenshot via OCR and analyzes it.
    Expects multipart/form-data with an 'image' file field.
    """
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing 'image' file in upload request.",
            "error_code": "MISSING_FILE",
        }), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image file selected for upload.",
            "error_code": "NO_FILE_SELECTED",
        }), 400

    file_bytes = file.read()
    filename = file.filename

    result = image_processor.process_and_analyze(file_bytes, filename)

    if result.get("success"):
        # Screenshots include page furniture (bylines, share buttons); drop it before searching.
        claim_source = result.get("extracted_text", "")
        if result.get("ocr_focus") != "headline":
            # Whole-page OCR still needs page furniture removed; a headline block is already clean.
            claim_source = strip_native_news_boilerplate(claim_source)
        verify_result(result, claim_source)

    status_code = 200 if result.get("success") else 400
    if result.get("error_code") == "OCR_UNAVAILABLE":
        status_code = 503
    elif result.get("error_code") == "MODEL_UNAVAILABLE":
        status_code = 503

    return jsonify(result), status_code


@app.route("/api/analyze-link", methods=["POST"])
@link_rate_limit
def analyze_link():
    """
    Opens a pasted news link, extracts the headline and article text, then analyzes and
    verifies that text exactly like a pasted article.
    Request body: JSON with 'url' field.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request content type must be application/json.",
            "error_code": "INVALID_CONTENT_TYPE",
        }), 400

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload.get("url"), str):
        return jsonify({
            "success": False,
            "error": "Missing 'url' field in JSON request body.",
            "error_code": "MISSING_FIELD",
        }), 400

    try:
        page = read_news_link(payload["url"])
    except LinkReadError as exc:
        return jsonify({
            "success": False,
            "error": exc.message,
            "error_code": exc.code,
        }), exc.status
    except Exception:  # noqa: BLE001 - a bad page must never crash the server
        logger.exception("Unexpected error while reading link")
        return jsonify({
            "success": False,
            "error": "Something went wrong while reading that link. Try again or paste the article text.",
            "error_code": "LINK_READ_FAILED",
        }), 500

    article_text = page["text"]
    title = page["title"]
    result = text_analyzer.analyze(article_text)

    # The headline is the claim readers saw; put it first so verification checks it.
    headline = title.rstrip(" .") + "." if title and not title.rstrip().endswith(("?", "!")) else title
    claim_text = f"{headline}\n\n{article_text}" if headline else article_text

    # Keep the untouched model result: verification overwrites the percentages, and an opinion
    # headline may need a second, focused check below.
    base_result = copy.deepcopy(result)

    if result.get("success"):
        verify_result(result, claim_text)
    elif result.get("error_code") == "TRANSLATION_FAILED":
        result = _verify_without_model(claim_text.strip(), result)

    # Feature / opinion headlines ("AC room, air travel, Kerala should learn...") assert nothing a
    # search can confirm, yet the article itself often states figures. Check the most specific
    # sentence of the article instead of reporting "nothing to verify".
    if (
        result.get("success")
        and base_result.get("success")
        and (result.get("verification") or {}).get("claim_checkable") is False
    ):
        focus = add_context(pick_checkable_sentence(article_text), title, article_text)
        if focus:
            retry = copy.deepcopy(base_result)
            verify_result(retry, focus, use_translation=False)
            if (retry.get("verification") or {}).get("claim_checkable") is not False:
                limitations = (retry.get("verification") or {}).get("limitations")
                if isinstance(limitations, list):
                    limitations.insert(
                        0,
                        "The headline reads like an opinion or feature headline, so Pratyora checked the "
                        "most specific sentence of the article instead (shown as the claim checked).",
                    )
                retry["claim_checked_from"] = "article_sentence"
                result = retry

    if result.get("success"):
        result["source_type"] = "link"
        result["source_url"] = page["url"]
        result["article_title"] = title
        result["extracted_text"] = article_text

    status_code = 200 if result.get("success") else 400
    if result.get("error_code") == "MODEL_UNAVAILABLE":
        status_code = 503

    return jsonify(result), status_code


# Safe Error Handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "success": False,
        "error": "Uploaded payload exceeds the maximum permitted size (10 MB).",
        "error_code": "PAYLOAD_TOO_LARGE",
    }), 413


@app.errorhandler(429)
def handle_rate_limited(e):
    return jsonify({
        "success": False,
        "error": "Too many link checks in a short time. Please wait a minute and try again.",
        "error_code": "RATE_LIMITED",
    }), 429


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        "success": False,
        "error": "The requested resource was not found.",
        "error_code": "NOT_FOUND",
    }), 404


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({
        "success": False,
        "error": "HTTP method not allowed for this endpoint.",
        "error_code": "METHOD_NOT_ALLOWED",
    }), 405


@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.exception("Internal unhandled server error: %s", e)
    return jsonify({
        "success": False,
        "error": "An internal server error occurred while processing the request.",
        "error_code": "INTERNAL_SERVER_ERROR",
    }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    logger.info("Starting Pratyora server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)