"""
Pratyora — Text Credibility Analyzer Service
Loads the trained scikit-learn TF-IDF + Logistic Regression pipeline,
validates inputs, evaluates vocabulary coverage / evidence sufficiency,
and calculates calibrated model-estimated fake and real percentages.

Multilingual support: non-English text is detected and machine-translated to
English before being scored by the (English-trained) model.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import joblib

from services.language_utils import (
    MIN_DETECTION_CONFIDENCE,
    detect_language,
    language_display_name,
    translate_to_english,
)
from services.text_cleaning import strip_native_news_boilerplate, strip_source_artifacts

logger = logging.getLogger("PratyoraTextAnalyzer")

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.joblib")
)
NON_LATIN_MIN_CONFIDENCE = 0.50
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 20000
MIN_WORD_COUNT = 3
# Below this many words (e.g. a lone headline from a screenshot) there is too little
# wording for the model to judge reliably, so we report insufficient evidence.
MIN_RELIABLE_WORDS = 20
FAKE_THRESHOLD = 65.0   # fake% at/above this => Likely Fake
REAL_THRESHOLD = 40.0   # fake% at/below this => Likely Real
# Translated text is out of distribution for an English (mostly US news) model,
# so 'Likely Fake' needs much stronger evidence when text was translated.
TRANSLATED_FAKE_THRESHOLD = 85.0


# Unicode-aware word patterns for English and the Indian scripts Pratyora supports.
# Python's \w does NOT include combining marks (vowel signs, viramas, anusvara), so
# without them a Telugu/Tamil/Bengali word is split into fragments. INDIC_MARKS lists
# the combining-mark ranges for: Devanagari (Hindi, Marathi, Bhojpuri), Bengali,
# Gurmukhi (Punjabi), Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam.
INDIC_MARKS = (
    "\u0300-\u036f"                                  # Latin combining diacritics
    "\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963"   # Devanagari
    "\u0981-\u0983\u09bc\u09be-\u09cd\u09d7\u09e2-\u09e3"     # Bengali
    "\u0a01-\u0a03\u0a3c\u0a3e-\u0a4d\u0a51\u0a70-\u0a71\u0a75"  # Gurmukhi
    "\u0a81-\u0a83\u0abc\u0abe-\u0acd\u0ae2-\u0ae3"           # Gujarati
    "\u0b01-\u0b03\u0b3c\u0b3e-\u0b4d\u0b56-\u0b57\u0b62-\u0b63"  # Odia
    "\u0b82\u0bbe-\u0bcd\u0bd7"                            # Tamil
    "\u0c00-\u0c04\u0c3e-\u0c56\u0c62-\u0c63"             # Telugu
    "\u0c81-\u0c83\u0cbc\u0cbe-\u0cd6\u0ce2-\u0ce3"       # Kannada
    "\u0d00-\u0d03\u0d3b-\u0d3c\u0d3e-\u0d4d\u0d57\u0d62-\u0d63"  # Malayalam
)

# A word starts with a letter and may continue with letters or combining marks.
LETTER_WORD_RE = re.compile(
    r"[^\W\d_](?:[^\W\d_]|[" + INDIC_MARKS + r"])*",
    re.UNICODE,
)
ANY_WORD_RE = LETTER_WORD_RE


TRANSLATION_NOTE = (
    "This text was machine-translated to English before analysis. The model was trained "
    "on English news, and translation can remove stylistic cues, so reliability is lower "
    "than for native English text."
)


class TextAnalyzer:
    _instance: Optional["TextAnalyzer"] = None

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.metadata = {}
        self.fake_class_index = 1
        self.real_class_index = 0
        self.load_model()

    @classmethod
    def get_instance(cls, model_path: str = DEFAULT_MODEL_PATH) -> "TextAnalyzer":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def load_model(self) -> bool:
        """
        Loads the joblib artifact containing the pipeline and training metadata.
        Inspects classifier classes dynamically to prevent incorrect index assumptions.
        """
        if not os.path.exists(self.model_path):
            logger.warning("Model file not found at %s", self.model_path)
            self.pipeline = None
            self.metadata = {}
            return False

        try:
            artifact = joblib.load(self.model_path)
            if isinstance(artifact, dict) and "pipeline" in artifact:
                self.pipeline = artifact["pipeline"]
                self.metadata = artifact.get("metadata", {})
            else:
                self.pipeline = artifact
                self.metadata = {}

            # Determine class indices dynamically
            classifier = None
            if hasattr(self.pipeline, "named_steps") and "classifier" in self.pipeline.named_steps:
                classifier = self.pipeline.named_steps["classifier"]
            elif hasattr(self.pipeline, "classes_"):
                classifier = self.pipeline

            if classifier is not None and hasattr(classifier, "classes_"):
                classes = list(classifier.classes_)
                # Target convention: 0 = Likely Real, 1 = Likely Fake
                if 1 in classes and 0 in classes:
                    self.fake_class_index = classes.index(1)
                    self.real_class_index = classes.index(0)
                elif "fake" in [str(c).lower() for c in classes]:
                    str_classes = [str(c).lower() for c in classes]
                    self.fake_class_index = str_classes.index("fake")
                    self.real_class_index = str_classes.index("real")
                else:
                    self.fake_class_index = 1 if len(classes) > 1 else 0
                    self.real_class_index = 0

            logger.info(
                "Successfully loaded model from %s. Fake index=%d, Real index=%d (classes=%s)",
                self.model_path,
                self.fake_class_index,
                self.real_class_index,
                classes if classifier is not None and hasattr(classifier, "classes_") else "unknown",
            )
            return True
        except Exception as e:
            logger.error("Failed to load model from %s: %s", self.model_path, e)
            self.pipeline = None
            return False

    def is_available(self) -> bool:
        return self.pipeline is not None

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize whitespace while preserving punctuation
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validates text input for type, length, and presence of intelligible words
        (in any script).
        """
        if text is None or not isinstance(text, str):
            return False, "Input text must be a valid string."

        cleaned = self.clean_text(text)
        if len(cleaned) < MIN_TEXT_LENGTH:
            return (
                False,
                f"Input text is too short ({len(cleaned)} characters). Please provide at least {MIN_TEXT_LENGTH} characters for analysis.",
            )
        if len(cleaned) > MAX_TEXT_LENGTH:
            return (
                False,
                f"Input text exceeds the maximum allowed length of {MAX_TEXT_LENGTH} characters (provided {len(cleaned)}).",
            )

        words = LETTER_WORD_RE.findall(cleaned)
        if len(words) < MIN_WORD_COUNT:
            return (
                False,
                f"Input text must contain at least {MIN_WORD_COUNT} readable words (found {len(words)}).",
            )

        return True, None

    def evaluate_evidence_sufficiency(self, cleaned_text: str) -> Tuple[bool, int, int]:
        """
        Calculates how many vocabulary features from the trained TF-IDF model
        are active in the input text. Returns (is_sufficient, active_features, total_words).
        """
        if not self.is_available() or not hasattr(self.pipeline, "named_steps"):
            return False, 0, 0

        tfidf = self.pipeline.named_steps.get("tfidf")
        if not tfidf or not hasattr(tfidf, "vocabulary_"):
            return True, 1, 1

        tokens = LETTER_WORD_RE.findall(cleaned_text.lower())
        total_words = len(tokens)

        # Transform single input text to sparse TF-IDF vector
        feat_vec = tfidf.transform([cleaned_text])
        active_features = int(feat_vec.nnz)

        # If zero features match or text with >=3 words has <2 matching features,
        # statistical evidence is insufficient to distinguish real from fake.
        if active_features == 0 or (total_words >= 3 and active_features < 2):
            return False, active_features, total_words

        return True, active_features, total_words

    def is_model_statistically_reliable(self) -> Tuple[bool, Optional[str]]:
        """
        Return (is_reliable, warning) based on the loaded model metadata.
        """
        metadata = self.metadata

        if not isinstance(metadata, dict):
            return False, (
                "The model metadata is incomplete. "
                "Its reliability has not been established."
            )

        reliability = metadata.get("reliability_status")
        metrics = metadata.get("metrics", {})
        if not isinstance(metrics, dict):
            return False, (
                "The model's training or evaluation metadata is incomplete. "
                "Its reliability has not been established."
            )

        train_records = metadata.get("train_records", 0)
        accuracy = metrics.get("accuracy")

        if not isinstance(train_records, (int, float)):
            return False, (
                "The model's training record count is invalid. "
                "Its reliability has not been established."
            )

        if not isinstance(accuracy, (int, float)):
            return False, (
                "A valid evaluation accuracy was not found. "
                "The model's reliability has not been established."
            )

        if (
            reliability not in ("validated", "internal_checks_passed")
            or train_records < 100
            or not 0.0 <= accuracy <= 1.0
            or accuracy < 0.65
        ):
            return False, (
                "The model has not met the application's minimum reliability checks. "
                f"Training records: {train_records}; "
                f"reported evaluation accuracy: {accuracy * 100:.1f}%. "
                "Results should be treated as experimental."
            )

        return True, None

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """
        Runs text analysis through the loaded pipeline.
        Non-English input is detected and translated to English first.
        Implements evidence gating: returns 'Insufficient Evidence' when text
        has negligible overlap with model vocabulary, preventing false-positive fake news flags.
        """
        valid, err = self.validate_input(raw_text)
        if not valid:
            return {
                "success": False,
                "error": err,
                "error_code": "INVALID_INPUT",
            }

        if not self.is_available():
            # Try reloading once in case the model was just trained
            if not self.load_model():
                return {
                    "success": False,
                    "error": "The machine learning model is currently unavailable. Please run 'python train_model.py' to train and save the model binary.",
                    "error_code": "MODEL_UNAVAILABLE",
                }

        cleaned_text = self.clean_text(raw_text)
        words = ANY_WORD_RE.findall(cleaned_text)
        word_count = len(words)
        snippet = cleaned_text[:160] + "..." if len(cleaned_text) > 160 else cleaned_text

        # --- Language detection and translation -------------------------------
        # Remove Hindi news-page furniture (bylines, "updated" stamps, share buttons,
        # "---- समाप्त ----") so it is neither translated nor scored as news content.
        detection_text = strip_native_news_boilerplate(cleaned_text) or cleaned_text
        lang, lang_conf = detect_language(detection_text)
        translated = False
        analysis_text = detection_text

        # Indic-script text is easily confused between sibling languages, so the script itself is
        # strong evidence the text is NOT English: accept a lower confidence there instead of
        # silently scoring it as English (which produces meaningless results).
        letters = [c for c in detection_text if c.isalpha()]
        non_latin = bool(letters) and sum(not c.isascii() for c in letters) / len(letters) >= 0.5
        min_conf = NON_LATIN_MIN_CONFIDENCE if non_latin else MIN_DETECTION_CONFIDENCE
        if lang not in ("en", "unknown") and lang_conf >= min_conf:
            ok, translated_text, terr = translate_to_english(detection_text, lang)
            if not ok:
                return {
                    "success": False,
                    "error": (
                        f"Could not translate the {language_display_name(lang)} text to English "
                        f"for analysis. {terr}"
                    ),
                    "error_code": "TRANSLATION_FAILED",
                    "detected_language": lang,
                    "detected_language_name": language_display_name(lang),
                }
            analysis_text = translated_text
            translated = True
            logger.info(
                "Translated %s -> en (%d chars). Preview: %r",
                lang, len(analysis_text), analysis_text[:200],
            )
        else:
            # Unknown or low-confidence detections are treated as English
            lang = "en"

        language_info = {
            "detected_language": lang,
            "detected_language_name": language_display_name(lang),
            "language_confidence": round(lang_conf, 3),
            "translated": translated,
            "translation_note": TRANSLATION_NOTE if translated else None,
            "translated_text": analysis_text if translated else None,
        }

        # Scrub publisher/page-furniture artifacts (same step used in training)
        model_text = strip_source_artifacts(analysis_text)
        model_word_count = len(ANY_WORD_RE.findall(model_text))
        too_short = model_word_count < MIN_RELIABLE_WORDS

        # 1. Evidence sufficiency check
        has_sufficient_evidence, active_features, _ = self.evaluate_evidence_sufficiency(model_text)
        if too_short:
            has_sufficient_evidence = False
        is_reliable, reliability_warning = self.is_model_statistically_reliable()

        if not has_sufficient_evidence:
            logger.info("Insufficient vocabulary overlap (%d active features) for text: %s", active_features, snippet)
            return {
                "success": True,
                "label": "Insufficient Evidence",
                "confidence": "Uncertain",
                "fake_percentage": 50.00,
                "real_percentage": 50.00,
                "insufficient_evidence": True,
                "active_features": active_features,
                "analyzed_characters": len(cleaned_text),
                "analyzed_words": word_count,
                "text_snippet": snippet,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_type": self.metadata.get("model_type", "TF-IDF + LogisticRegression"),
                "reliability_warning": reliability_warning,
                **language_info,
                "warning": (
                    (
                      f"Only {model_word_count} analyzable words were found; at least {MIN_RELIABLE_WORDS} "
"are needed for a reliable estimate. Headlines alone are too short - paste or "
"screenshot the full article text. There is insufficient statistical evidence "
"to reliably estimate credibility."
                        if too_short else
                        "The submitted text contains minimal lexical overlap with the training corpus. "
"There is insufficient statistical evidence to reliably estimate credibility. "
"Please provide more text for a more informative analysis."
                    )
                ),
            }

        try:
            # Predict probabilities using trained pipeline
            probabilities = self.pipeline.predict_proba([model_text])[0]
            prob_fake = float(probabilities[self.fake_class_index])
            prob_real = float(probabilities[self.real_class_index])

            # Convert to percentages rounded to 2 decimal places
            fake_percentage = round(prob_fake * 100.0, 2)
            # Ensure percentages sum precisely to 100.0%
            real_percentage = round(100.0 - fake_percentage, 2)

            # 2. Cautious decision boundaries.
            # Wrongly branding real news as fake is the costlier mistake, and the model
            # was trained on a narrow corpus, so "Likely Fake" needs strong evidence.
            # Anything in the grey zone is reported as inconclusive instead of guessed.
            fake_cutoff = TRANSLATED_FAKE_THRESHOLD if translated else FAKE_THRESHOLD
            if fake_percentage >= fake_cutoff:
                label = "Likely Fake"
                confidence = "High" if (fake_percentage >= 80.0 and is_reliable) else "Moderate"
            elif fake_percentage <= REAL_THRESHOLD:
                label = "Likely Real"
                confidence = "High" if (real_percentage >= 75.0 and is_reliable) else "Moderate"
            else:
                label = "Inconclusive / Mixed Evidence"
                confidence = "Low"

            if not is_reliable:
                confidence = "Low"

            # Translated text loses stylistic cues, so never report High confidence for it
            if translated and confidence == "High":
                confidence = "Moderate"

            warning = (
                "These scores are statistical model estimates based on lexical and stylistic "
                "patterns in the training data. They are not proof of factual truth or falsehood. "
                "Always verify critical claims through primary and independent journalistic sources."
            )
            if translated:
                warning = TRANSLATION_NOTE + " " + warning

            return {
                "success": True,
                "label": label,
                "confidence": confidence,
                "fake_percentage": fake_percentage,
                "real_percentage": real_percentage,
                "insufficient_evidence": False,
                "active_features": active_features,
                "analyzed_characters": len(cleaned_text),
                "analyzed_words": word_count,
                "text_snippet": snippet,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_type": self.metadata.get("model_type", "TF-IDF + LogisticRegression"),
                "reliability_warning": reliability_warning,
                **language_info,
                "warning": warning,
            }
        except Exception as e:
            logger.exception("Error during text classification: %s", e)
            return {
                "success": False,
                "error": "An internal error occurred during text classification.",
                "error_code": "INFERENCE_ERROR",
            }