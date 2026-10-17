"""
Pratyora — Language Detection and Translation Utilities

Detects submitted text languages and translates non-English text
to English using local Argos/IndicTrans2 models where available,
with an online translation fallback.
"""

import logging
import re
import threading
from collections import OrderedDict
from typing import Iterator, Optional, Tuple

import argostranslate.settings as argos_settings
import argostranslate.translate
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, detect_langs

from services.dialect_markers import bhojpuri_score, looks_bhojpuri

# Avoid Stanza's incomplete Bengali sentence-splitting resources.
argos_settings.chunk_type = argos_settings.ChunkType.MINISBD

# Make language detection deterministic across runs.
DetectorFactory.seed = 0

logger = logging.getLogger("PratyoraLanguage")

MIN_DETECTION_CONFIDENCE = 0.80
CHUNK_SIZE = 4500

# Limit memory use by keeping only a fixed number of successful translations.
MAX_TRANSLATION_CACHE_SIZE = 128
_translation_cache = OrderedDict()
_cache_lock = threading.RLock()

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "ne": "Nepali",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "id": "Indonesian",
    "tr": "Turkish",
    "bho": "Bhojpuri",
}

# langdetect and Argos Translate use different codes for some languages.
ARGOS_CODE_MAP = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def language_display_name(code: str) -> str:
    """Return a readable language name."""
    normalized_code = (code or "unknown").strip().lower()
    return LANGUAGE_NAMES.get(normalized_code, normalized_code)



def detect_language(text):
    """
    Return (language_code, confidence), or ('unknown', 0.0).
    Devanagari text with Bhojpuri words is re-labelled "bho" after statistical detection.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    # Odia-script detection (langdetect has no Odia). Require a meaningful share of
    # Odia characters so one stray character in English text is not treated as Odia.
    odia_chars = len(re.findall(r"[\u0B00-\u0B7F]", text))
    if odia_chars:
        letters = len(re.findall(r"[^\W\d_]|[\u0B00-\u0B7F]", text)) or 1
        if odia_chars / letters >= 0.30:
            return "or", 0.99

    # Statistical detection first (it reports Bhojpuri as Hindi, Nepali or Marathi) ...
    best_lang, best_prob = "unknown", 0.0
    try:
        predictions = detect_langs(text)
        if predictions:
            best_lang, best_prob = predictions[0].lang, float(predictions[0].prob)
    except Exception as exc:
        logger.info("Language detection failed: %s", exc)

    # ... then re-label Devanagari text that carries Bhojpuri words (whole-word matching).
    if looks_bhojpuri(text, best_lang):
        return "bho", min(0.99, 0.80 + 0.05 * bhojpuri_score(text))

    return best_lang, best_prob


def _chunk_text(text: str) -> Iterator[str]:
    """Split text into manageable chunks, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?।॥。！？])\s+", text)
    chunk = ""

    for sentence in sentences:
        while len(sentence) > CHUNK_SIZE:
            if chunk:
                yield chunk
                chunk = ""

            yield sentence[:CHUNK_SIZE]
            sentence = sentence[CHUNK_SIZE:]

        if len(chunk) + len(sentence) + 1 > CHUNK_SIZE and chunk:
            yield chunk
            chunk = ""

        chunk += (" " if chunk else "") + sentence

    if chunk:
        yield chunk


def _get_cached_translation(
    text: str,
    source_lang: str,
) -> Optional[str]:
    """Return a previously successful translation, if available."""
    key = (source_lang, text)

    with _cache_lock:
        translated = _translation_cache.get(key)

        if translated is not None:
            _translation_cache.move_to_end(key)

        return translated


def _cache_translation(
    text: str,
    source_lang: str,
    translated: str,
) -> None:
    """Cache successful translations only."""
    if not translated or not translated.strip():
        return

    key = (source_lang, text)

    with _cache_lock:
        _translation_cache[key] = translated.strip()
        _translation_cache.move_to_end(key)

        while len(_translation_cache) > MAX_TRANSLATION_CACHE_SIZE:
            _translation_cache.popitem(last=False)


def _translate_with_indictrans2(
    text: str,
    source_lang: str,
) -> Tuple[bool, str, Optional[str]]:
    """Use the local IndicTrans2 service for configured languages."""
    import json
    import urllib.request
    import urllib.error

    url = "http://127.0.0.1:8765/translate"

    payload = json.dumps({
        "text": text,
        "source_lang": source_lang,
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        translated = result.get("translated_text", "").strip()

        if result.get("success") and translated:
            return True, translated, None

        return False, "", "IndicTrans2 returned no translation."

    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("IndicTrans2 service unavailable: %s", exc)
        return False, "", "IndicTrans2 service is unavailable."

    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Invalid IndicTrans2 response: %s", exc)
        return False, "", "Invalid response from IndicTrans2."


# --------------------------------------------------------------------------- #
# Guard against machine-translation "hallucinated" acronyms (e.g. Telugu "scripted" -> "QOS")
# --------------------------------------------------------------------------- #
# Acronyms translators legitimately produce from native-script text (उदा. भाजपा -> BJP).
KNOWN_ACRONYMS = {
    "BJP", "INC", "AAP", "RSS", "CBI", "NIA", "FIR", "IPL", "ISRO", "DRDO", "NDA", "UPA", "UPI", "ATM",
    "GST", "RBI", "SEBI", "PTI", "ANI", "IAS", "IPS", "IIT", "IIM", "NASA", "WHO", "NATO", "OPEC", "IMF",
    "UAE", "USA", "COVID", "CAA", "NRC", "EVM", "ECI", "OTP", "KYC", "GPS", "SMS", "SBI", "LPG", "CNG",
    "DNA", "PCR", "ICC", "BCCI", "FIFA", "MLA", "TDP", "DMK", "BRS", "TMC", "NCP", "CPI", "CPM", "RJD",
    "JDU", "BSP", "SP", "NEET", "JEE", "CBSE", "UGC", "AIIMS", "TRAI", "SIM", "PAN", "ID", "FM", "TV",
}
_CAPS_TOKEN = re.compile(r"\b[A-Z]{3,5}\b")


def suspicious_caps(translated: str, source: str = "") -> list:
    """ALL-CAPS 3-5 letter tokens in a translation that are neither in the source text nor a
    well-known acronym. Such tokens are almost always translator noise."""
    source_upper = str(source or "").upper()
    return [t for t in _CAPS_TOKEN.findall(str(translated or ""))
            if t not in KNOWN_ACRONYMS and t not in source_upper]


def strip_suspicious_caps(translated: str, source: str = "") -> str:
    """Remove suspicious acronyms so they cannot poison a search query."""
    bad = set(suspicious_caps(translated, source))
    if not bad:
        return translated
    cleaned = re.sub(r"\b(?:%s)\b" % "|".join(sorted(bad)), "", str(translated))
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _translate_online(text: str, source_lang: str) -> str:
    """Google translation of `text`, or '' on any failure (used as a second opinion)."""
    online_code = "hi" if source_lang == "bho" else source_lang
    try:
        results = GoogleTranslator(source=online_code, target="en").translate_batch(list(_chunk_text(text)))
        if isinstance(results, str):
            results = [results]
        if results and all(isinstance(r, str) and r.strip() for r in results):
            return " ".join(r.strip() for r in results)
    except Exception as exc:  # noqa: BLE001
        logger.info("Second-opinion translation failed: %s", exc)
    return ""


def _prefer_clean_translation(text: str, source_lang: str, translated: str) -> str:
    """If the primary translation contains suspicious acronyms, ask Google and keep whichever
    has fewer of them. The primary result is kept when Google is no better."""
    bad = suspicious_caps(translated, text)
    if not bad:
        return translated
    alt = _translate_online(text, source_lang)
    if alt and len(suspicious_caps(alt, text)) < len(bad):
        logger.info("Replaced translation containing %s with an online second opinion.", bad)
        return alt
    return translated


def translate_to_english(
    text: str,
    source_lang: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Translate text into English.

    Prefer the configured local IndicTrans2 service, then an installed
    Argos translation model, then Google Translate.

    Returns (success, translated_text, error_message).
    """
    if not text or not text.strip():
        return False, "", "No text was available to translate."

    text = text.strip()
    source_lang = (source_lang or "unknown").strip().lower()

    if source_lang == "en":
        return True, text, None

    # Reuse successful translations to reduce repeated online requests.
    cached_translation = _get_cached_translation(text, source_lang)

    if cached_translation is not None:
        logger.info(
            "Using cached translation for %s.",
            language_display_name(source_lang),
        )
        return True, cached_translation, None

    
    # These languages are configured to use your local IndicTrans2 service.
    indictrans_languages = {"ta", "te", "kn", "pa", "ml", "gu", "mr", "or"}

    if source_lang in indictrans_languages:
        success, translated, error = _translate_with_indictrans2(
            text,
            source_lang,
        )

        # Return immediately when the local translation succeeds.
        if success and translated:
            translated = _prefer_clean_translation(text, source_lang, translated)
            _cache_translation(text, source_lang, translated)
            return True, translated, None

        # If local translation fails, continue to the existing fallbacks.

    # Bhojpuri uses the local NLLB route in the translation service.
    # If that service is unavailable, continue to the online fallback below.
    if source_lang == "bho":
        success, translated, error = _translate_with_indictrans2(
            text,
            source_lang,
        )

        if success and translated:
            translated = _prefer_clean_translation(text, source_lang, translated)
            _cache_translation(text, source_lang, translated)
            return True, translated, None

    chunks = list(_chunk_text(text))

    if not chunks:
        return False, "", "No text was available to translate."

    # First, try an installed offline Argos translation model.
    try:
        languages = argostranslate.translate.get_installed_languages()
        argos_source_code = ARGOS_CODE_MAP.get(
            source_lang,
            source_lang,
        )

        source = next(
            (
                lang for lang in languages
                if lang.code == argos_source_code
            ),
            None,
        )

        english = next(
            (lang for lang in languages if lang.code == "en"),
            None,
        )

        if source and english:
            translation = source.get_translation(english)
            results = [
                translation.translate(chunk)
                for chunk in chunks
            ]

            if all(
                isinstance(item, str) and item.strip()
                for item in results
            ):
                translated = " ".join(
                    item.strip() for item in results
                )

                _cache_translation(text, source_lang, translated)
                return True, translated, None

    except Exception as exc:
        logger.info("Local Argos translation unavailable: %s", exc)

    # Fall back to the online translator for supported languages.
    try:
        results = None
        if source_lang == "bho":
            # Try Bhojpuri first; if the online service rejects it, Hindi is a close fallback.
            for online_code in ("bho", "hi"):
                try:
                    results = GoogleTranslator(source=online_code, target="en").translate_batch(chunks)
                    if results:
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.info("Online translation as %s failed: %s", online_code, exc)
        else:
            results = GoogleTranslator(source=source_lang, target="en").translate_batch(chunks)

        if isinstance(results, str):
            results = [results]

        if (
            results
            and len(results) == len(chunks)
            and all(
                isinstance(item, str) and item.strip()
                for item in results
            )
        ):
            translated = " ".join(
                item.strip() for item in results
            )

            _cache_translation(text, source_lang, translated)
            return True, translated, None

        return (
            False,
            "",
            f"Translation for {language_display_name(source_lang)} "
            "returned incomplete results.",
        )

    except Exception as exc:
        logger.warning(
            "Online translation from %s failed: %s",
            source_lang,
            exc,
        )

        return (
            False,
            "",
            f"Translation for {language_display_name(source_lang)} "
            "is unavailable or rate-limited. Please try again later.",
        )