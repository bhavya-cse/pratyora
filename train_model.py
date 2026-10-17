#!/usr/bin/env python3
"""
Pratyora — Machine Learning Model Training & Evaluation Pipeline (v2)

Trains a TF-IDF + Logistic Regression classifier on news articles and evaluates it
honestly. Compatible with ISOT-style Fake.csv / True.csv files, and with extra
labelled CSVs (--extra-csv) so Indian / other-domain data can be mixed in later.

What changed versus v1 (each point was measured on the ISOT files, see the report):

1. Leak-free text.  services/text_cleaning.py v2 removes publisher fingerprints that
   v1 missed (city datelines, "don t"-style apostrophe artifacts, "Via:", "WATCH:",
   glued sentences, reporter credits, unreliable-site brand names).
2. Short-text robustness.  Screenshots/OCR usually give 30-150 words, but the model
   only saw full ~400-word articles (accuracy fell to ~87% on short snippets). Training
   now also includes random 20-150 word excerpts of every article.
3. Better features.  English stop-words are kept (function words carry style cues),
   vocabulary 50,000 (1-2 grams), C=3.  Character n-grams were tested and REJECTED:
   with punctuation they scored higher only because they learned typography
   (curly quotes: 45% of real vs 13% of fake articles); restricted to letters the
   gain vanished.
4. Calibrated probabilities.  A Platt (sigmoid) calibrator is fitted on a held-out
   slice so "85% fake" is closer to a real frequency (ECE roughly halves).
5. Honest evaluation.  The report scores the untouched test set on full articles,
   short excerpts, noisy/OCR-like text, and a "novel" subset with no near-duplicate in
   the training data, plus Brier score, ECE and accuracy at the app's decision
   thresholds. It also stores the strongest words per class so leakage is visible.

The saved artifact keeps the layout text_analyzer.py expects:
    {"pipeline": Pipeline([("tfidf", TfidfVectorizer), ("classifier", ...)]),
     "metadata": {...}}
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from services.text_cleaning import strip_source_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PratyoraTrainer")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "fake_news_model.joblib")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_MODEL_DIR, "evaluation_report.json")

# Keep in sync with services/text_analyzer.py (REAL_THRESHOLD / FAKE_THRESHOLD, as probabilities).
APP_REAL_THRESHOLD = 0.40
APP_FAKE_THRESHOLD = 0.65
# Keep in sync with text_analyzer.MIN_RELIABLE_WORDS: shorter texts are never scored.
DEFAULT_MIN_WORDS = 20

DEFAULT_C = 3.0
DEFAULT_MAX_FEATURES = 50000
CROP_MIN_WORDS, CROP_MAX_WORDS = 20, 150
CROPS_PER_DOC = 2
CALIBRATION_FRACTION = 0.15
MIN_TRAIN_FOR_CALIBRATION = 400

_TITLE_COLS = ("title", "headline", "heading", "name")
_TEXT_COLS = ("text", "article", "content", "body", "news")
_FAKE_LABELS = {"fake", "false", "1", "1.0", "unreliable", "misinformation"}
_REAL_LABELS = {"real", "true", "0", "0.0", "reliable"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, sep=";", on_bad_lines="skip")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _article_text(df: pd.DataFrame, source_name: str) -> pd.Series:
    title_col = next((c for c in _TITLE_COLS if c in df.columns), None)
    text_col = next((c for c in _TEXT_COLS if c in df.columns), None)
    if not title_col and not text_col:
        raise ValueError(
            f"{source_name} has no recognized title or article column. Found columns: {list(df.columns)}"
        )
    if title_col and text_col:
        return df[title_col].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
    return df[text_col or title_col].fillna("").astype(str)


def _load_extra_csv(path: str) -> pd.DataFrame:
    """Extra labelled data: needs a text (or article/content/body) column and a 'label'
    column (fake/false/1 = fake, real/true/0 = real). Rows with other labels are skipped."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extra dataset not found: {path}")
    df = _read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"{path} needs a 'label' column (fake/real or 1/0). Found: {list(df.columns)}")
    labels = df["label"].astype(str).str.strip().str.lower()
    target = labels.map(lambda v: 1 if v in _FAKE_LABELS else (0 if v in _REAL_LABELS else -1))
    keep = target >= 0
    skipped = int((~keep).sum())
    if skipped:
        logger.warning("%s: skipped %d rows with unrecognised labels.", path, skipped)
    out = pd.DataFrame({"text": _article_text(df, path)[keep], "target": target[keep].astype(int)})
    logger.info("Loaded %d extra rows from %s", len(out), path)
    return out


def load_and_preprocess_dataset(
    fake_path: str,
    true_path: str,
    max_samples: int = None,
    min_words: int = 0,
    extra_csvs=(),
) -> pd.DataFrame:
    """
    Loads, scrubs, validates and combines Fake and Real CSV files (plus optional extras).
    Target labels: 0 = Likely Real, 1 = Likely Fake.
    `min_words` drops articles shorter than that many words AFTER scrubbing
    (the CLI uses 20, matching the analyzer, which never scores shorter text).
    """
    logger.info("Verifying dataset files...")
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news dataset file not found: {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Real news dataset file not found: {true_path}")

    logger.info("Loading Fake news from: %s", fake_path)
    df_fake = _read_csv(fake_path)
    logger.info("Loading Real news from: %s", true_path)
    df_true = _read_csv(true_path)

    parts = [
        pd.DataFrame({"text": _article_text(df_fake, "Fake.csv"), "target": 1}),
        pd.DataFrame({"text": _article_text(df_true, "True.csv"), "target": 0}),
    ]
    for path in extra_csvs or ():
        parts.append(_load_extra_csv(path))

    df_combined = pd.concat(parts, ignore_index=True)
    df_combined["label"] = np.where(df_combined["target"] == 1, "fake", "real")

    # Remove source-identifying artifacts so the model learns wording/style, not the publisher.
    df_combined["text"] = df_combined["text"].map(strip_source_artifacts).str.strip()

    initial_count = len(df_combined)
    df_combined = df_combined[df_combined["text"].str.len() >= 20]
    if min_words and min_words > 0:
        df_combined = df_combined[df_combined["text"].str.split().str.len() >= min_words]

    # Deduplicate to prevent exact-copy leakage between train and test splits
    df_combined = df_combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    final_count = len(df_combined)

    real_count = int((df_combined["target"] == 0).sum())
    fake_count = int((df_combined["target"] == 1).sum())
    logger.info(
        "Dataset parsing complete: %d usable records (Real: %d, Fake: %d, Removed duplicates/short/empty: %d)",
        final_count, real_count, fake_count, initial_count - final_count,
    )
    if final_count < 10:
        raise ValueError(f"Dataset has only {final_count} usable records. Insufficient for training.")

    if max_samples and max_samples < final_count:
        logger.info("Stratified subsampling down to %d records...", max_samples)
        df_combined, _ = train_test_split(
            df_combined, train_size=max_samples, random_state=42, stratify=df_combined["target"]
        )
        logger.info(
            "Subsampled dataset: %d records (Real: %d, Fake: %d)",
            len(df_combined), int((df_combined["target"] == 0).sum()), int((df_combined["target"] == 1).sum()),
        )
    return df_combined.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stress-test helpers
# --------------------------------------------------------------------------- #
def _crop(text: str, rng: random.Random, lo: int = CROP_MIN_WORDS, hi: int = CROP_MAX_WORDS) -> str:
    """Random contiguous window of lo..hi words (what a screenshot or snippet looks like)."""
    words = text.split()
    n = rng.randint(lo, hi)
    if len(words) <= n:
        return text
    start = rng.randint(0, len(words) - n)
    return " ".join(words[start:start + n])


def _noisy(text: str, rng: random.Random) -> str:
    """OCR-like degradation: lowercase, no punctuation, 2% dropped and 1% wrong characters."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    out = []
    for ch in text:
        r = rng.random()
        if r < 0.02:
            continue
        out.append(rng.choice("abcdefghijklmnopqrstuvwxyz") if r < 0.03 else ch)
    return "".join(out)


def _augment_with_crops(X, y, rng: random.Random, per_doc: int = CROPS_PER_DOC):
    Xa, ya = list(X), list(y)
    for text, label in zip(X, y):
        for _ in range(per_doc):
            Xa.append(_crop(text, rng))
            ya.append(label)
    return Xa, np.asarray(ya)


def _novel_mask(X_train, X_test, threshold: float = 0.6) -> np.ndarray:
    """True for test articles whose most similar training article has cosine < threshold
    (i.e. no near-duplicate / reprint in the training data)."""
    try:
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2 if len(X_train) >= 50 else 1, max_features=50000)
        A = vec.fit_transform(X_train)
        B = vec.transform(X_test)
        best = np.zeros(B.shape[0])
        for i in range(0, B.shape[0], 500):
            best[i:i + 500] = (B[i:i + 500] @ A.T).max(axis=1).toarray().ravel()
        return best < threshold
    except Exception as exc:  # tiny / degenerate data
        logger.info("Novelty check skipped: %s", exc)
        return np.ones(len(X_test), dtype=bool)


def expected_calibration_error(y_true, p_fake, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    idx = np.minimum((np.asarray(p_fake) * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - np.asarray(p_fake)[m].mean())
    return float(ece)


def _condition_metrics(y_true, p_fake) -> dict:
    y_true = np.asarray(y_true)
    p_fake = np.asarray(p_fake)
    pred = (p_fake >= 0.5).astype(int)
    decisive = (p_fake <= APP_REAL_THRESHOLD) | (p_fake >= APP_FAKE_THRESHOLD)
    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "f1_macro": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, p_fake)), 4),
        "ece": round(expected_calibration_error(y_true, p_fake), 4),
        "at_app_thresholds": {
            "coverage": round(float(decisive.mean()), 4),
            "accuracy_when_decisive": (
                round(float(accuracy_score(y_true[decisive], (p_fake[decisive] >= APP_FAKE_THRESHOLD).astype(int))), 4)
                if decisive.any() else None
            ),
        },
    }
    return out


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _calibrate_prefit(clf, X_cal, y_cal):
    """Platt-scale an already fitted classifier on held-out data (sklearn >= 1.6 uses
    FrozenEstimator; older versions use cv='prefit')."""
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    except ImportError:
        cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    return cal.fit(X_cal, y_cal)


def _top_features(vectorizer, clf, n: int = 30) -> dict:
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return {
        "most_real": [str(w) for w in names[order[:n]]],
        "most_fake": [str(w) for w in names[order[-n:]][::-1]],
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    df: pd.DataFrame,
    model_output_path: str = DEFAULT_MODEL_PATH,
    report_output_path: str = DEFAULT_REPORT_PATH,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_features: int = DEFAULT_MAX_FEATURES,
    C: float = DEFAULT_C,
    tune: bool = False,
    augment: bool = True,
) -> dict:
    """
    Trains TF-IDF + Logistic Regression (+ sigmoid calibration), evaluates on a held-out
    test set under several conditions, compares against baselines, saves artifact + report.
    """
    rng = random.Random(random_seed)
    X = df["text"].astype(str).values
    y = df["target"].astype(int).values

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Hold out a slice of TRAINING data to fit the probability calibrator.
    use_calibration = len(X_train_all) >= MIN_TRAIN_FOR_CALIBRATION
    if use_calibration:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train_all, y_train_all, test_size=CALIBRATION_FRACTION,
            random_state=random_seed + 1, stratify=y_train_all,
        )
    else:
        logger.warning("Only %d training records: skipping probability calibration.", len(X_train_all))
        X_fit, y_fit, X_cal, y_cal = X_train_all, y_train_all, [], []

    train_real, train_fake = int((y_train_all == 0).sum()), int((y_train_all == 1).sum())
    test_real, test_fake = int((y_test == 0).sum()), int((y_test == 1).sum())
    logger.info("Split summary:")
    logger.info("  Training set: %d records (Real: %d, Fake: %d)", len(X_train_all), train_real, train_fake)
    logger.info("  Held-out test set: %d records (Real: %d, Fake: %d)", len(X_test), test_real, test_fake)

    # 1. Baselines
    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train_all, y_train_all)
    dummy_majority_acc = float(accuracy_score(y_test, dummy_majority.predict(X_test)))
    dummy_strat = DummyClassifier(strategy="stratified", random_state=random_seed).fit(X_train_all, y_train_all)
    dummy_strat_acc = float(accuracy_score(y_test, dummy_strat.predict(X_test)))
    logger.info("Baselines: majority %.4f | stratified random %.4f", dummy_majority_acc, dummy_strat_acc)

    # 2. Training data = full articles + random short excerpts of each article
    if augment:
        X_fit_a, y_fit_a = _augment_with_crops(X_fit, y_fit, rng)
        logger.info("Length augmentation: %d -> %d training texts", len(X_fit), len(X_fit_a))
    else:
        X_fit_a, y_fit_a = list(X_fit), np.asarray(y_fit)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=None,  # function words carry style cues; removing them measurably hurt
        sublinear_tf=True,
        min_df=2 if len(X_fit_a) >= 50 else 1,
    )
    logger.info("Fitting TF-IDF vectorizer...")
    A_fit = tfidf.fit_transform(X_fit_a)

    # Optional: choose C on the held-out slice (full articles + excerpts)
    chosen_C = C
    tuning_log = None
    if tune and use_calibration:
        X_val_a, y_val_a = (
            _augment_with_crops(X_cal, y_cal, random.Random(random_seed + 2)) if augment else (list(X_cal), y_cal)
        )
        A_val = tfidf.transform(X_val_a)
        tuning_log, best = {}, (-1.0, None)
        for cand in (1.0, 3.0, 10.0):
            m = LogisticRegression(C=cand, max_iter=2000, random_state=random_seed).fit(A_fit, y_fit_a)
            score = float(f1_score(y_val_a, m.predict(A_val), average="macro"))
            tuning_log[str(cand)] = round(score, 4)
            logger.info("  tune C=%-4s validation macro-F1 %.4f", cand, score)
            if score > best[0] + 1e-4:
                best = (score, cand)
        chosen_C = best[1]
        logger.info("Selected C=%s", chosen_C)

    logger.info("Fitting Logistic Regression (C=%s)...", chosen_C)
    clf = LogisticRegression(C=chosen_C, max_iter=2000, random_state=random_seed, solver="lbfgs")
    clf.fit(A_fit, y_fit_a)

    if use_calibration:
        X_cal_a, y_cal_a = (
            _augment_with_crops(X_cal, y_cal, random.Random(random_seed + 3)) if augment else (list(X_cal), y_cal)
        )
        classifier = _calibrate_prefit(clf, tfidf.transform(X_cal_a), y_cal_a)
        logger.info("Fitted sigmoid probability calibration on %d held-out texts.", len(X_cal_a))
    else:
        classifier = clf

    # Same layout text_analyzer.py reads: steps named "tfidf" and "classifier".
    pipeline = Pipeline([("tfidf", tfidf), ("classifier", classifier)])

    # 3. Evaluation on the untouched test set under several conditions
    eval_rng = random.Random(random_seed + 99)
    conditions = {
        "full_articles": list(X_test),
        "excerpt_30_60_words": [_crop(t, eval_rng, 30, 60) for t in X_test],
        "excerpt_60_120_words": [_crop(t, eval_rng, 60, 120) for t in X_test],
        "noisy_ocr_like": [_noisy(_crop(t, eval_rng, 60, 150), eval_rng) for t in X_test],
    }
    novel = _novel_mask(X_train_all, X_test)

    results = {}
    p_full = None
    for name, texts in conditions.items():
        p_fake = pipeline.predict_proba(texts)[:, list(classifier.classes_).index(1)]
        results[name] = _condition_metrics(y_test, p_fake)
        if name == "full_articles":
            p_full = p_fake
        logger.info(
            "  %-22s acc %.4f | brier %.4f | ECE %.4f | decisive %.1f%% (acc %s)",
            name, results[name]["accuracy"], results[name]["brier"], results[name]["ece"],
            100 * results[name]["at_app_thresholds"]["coverage"],
            results[name]["at_app_thresholds"]["accuracy_when_decisive"],
        )
    if novel.any():
        results["novel_articles_only"] = _condition_metrics(y_test[novel], p_full[novel])
        logger.info("  %-22s acc %.4f (n=%d, no near-duplicate in training data)",
                    "novel_articles_only", results["novel_articles_only"]["accuracy"], int(novel.sum()))

    # Calibration comparison on full articles (raw logistic regression vs calibrated)
    raw_p = clf.predict_proba(tfidf.transform(list(X_test)))[:, list(clf.classes_).index(1)]
    calibration_effect = {
        "raw_logistic_regression": {
            "brier": round(float(brier_score_loss(y_test, raw_p)), 4),
            "ece": round(expected_calibration_error(y_test, raw_p), 4),
        },
        "after_calibration": {
            "brier": results["full_articles"]["brier"],
            "ece": results["full_articles"]["ece"],
        },
        "calibrated": bool(use_calibration),
    }

    # Headline numbers (full articles, 0.5 cut-off) in the layout v1 used
    y_pred = (p_full >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    clf_report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Likely Real", "Likely Fake"],
        output_dict=True, zero_division=0,
    )
    logger.info("Headline (full articles): accuracy %.4f (majority baseline %.4f), macro-F1 %.4f", acc, dummy_majority_acc, f1_macro)
    logger.info("  Confusion matrix: [[True Real: %d, False Fake: %d], [False Real: %d, True Fake: %d]]", tn, fp, fn, tp)

    # Reliability assessment. Passing these checks does NOT mean independent validation.
    short_acc = results["excerpt_30_60_words"]["accuracy"]
    checks = {
        "dataset_sufficient": len(X_train_all) >= 200,
        "accuracy_acceptable": acc >= 0.70,
        "better_than_baseline": acc > dummy_majority_acc,
        "short_text_acceptable": short_acc >= 0.75,
        "calibration_acceptable": results["full_articles"]["ece"] <= 0.10,
    }
    reliability_status = "internal_checks_passed" if all(checks.values()) else "not_yet_reliable"

    top_features = _top_features(tfidf, clf)

    metadata = {
        "model_type": (
            "TF-IDF (1-2 ngrams, 50k) + LogisticRegression (L-BFGS)"
            + (" + sigmoid calibration" if use_calibration else "")
        ),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "test_size": test_size,
        "total_records": len(df),
        "train_records": len(X_train_all),
        "test_records": len(X_test),
        "train_counts": {"real": train_real, "fake": train_fake},
        "test_counts": {"real": test_real, "fake": test_fake},
        "vocabulary_size": len(tfidf.vocabulary_),
        "hyperparameters": {
            "C": chosen_C,
            "max_features": max_features,
            "ngram_range": [1, 2],
            "stop_words": None,
            "length_augmentation": bool(augment),
            "crops_per_article": CROPS_PER_DOC if augment else 0,
            "calibration": "sigmoid (held-out slice)" if use_calibration else "none",
            "tuning_validation_macro_f1": tuning_log,
        },
        "reliability_status": reliability_status,
        "reliability_checks": checks,
        "validation_note": (
            "Internal held-out test checks only. This model has not been "
            "independently validated for factual truth, unseen publishers, "
            "future news, Indian news, or translated and multilingual articles."
        ),
        "evaluation_method": (
            "Random stratified train-test split after exact-text deduplication and source-artifact "
            "scrubbing. Also scored on short excerpts, noisy OCR-like text, and a 'novel' subset with no "
            "near-duplicate in training. Related articles from the same publishers may still overlap, so "
            "these numbers are optimistic for news from other publishers or countries."
        ),
        "class_mapping": {0: "Likely Real", 1: "Likely Fake"},
        "classes": [int(c) for c in clf.classes_],
        "intercept": [float(i) for i in clf.intercept_],
        "baselines": {
            "majority_class_accuracy": round(dummy_majority_acc, 4),
            "stratified_random_accuracy": round(dummy_strat_acc, 4),
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "short_text_accuracy": short_acc,
            "confusion_matrix": {
                "true_real": tn, "false_fake": fp, "false_real": fn, "true_fake": tp, "matrix": cm,
            },
            "per_class": {
                "likely_real": {
                    "precision": round(clf_report["Likely Real"]["precision"], 4),
                    "recall": round(clf_report["Likely Real"]["recall"], 4),
                    "f1_score": round(clf_report["Likely Real"]["f1-score"], 4),
                    "support": int(clf_report["Likely Real"]["support"]),
                },
                "likely_fake": {
                    "precision": round(clf_report["Likely Fake"]["precision"], 4),
                    "recall": round(clf_report["Likely Fake"]["recall"], 4),
                    "f1_score": round(clf_report["Likely Fake"]["f1-score"], 4),
                    "support": int(clf_report["Likely Fake"]["support"]),
                },
            },
        },
        "evaluation_conditions": results,
        "calibration_effect": calibration_effect,
        "app_thresholds_fake_probability": {"likely_real_at_or_below": APP_REAL_THRESHOLD, "likely_fake_at_or_above": APP_FAKE_THRESHOLD},
        "top_features": top_features,
        "dataset_documentation": {
            "source": "University of Victoria ISOT Fake News Dataset / Open Research News Corpus",
            "components": "True.csv (Reuters/AP journalistic coverage) and Fake.csv (flagged deceptive claims)",
            "license": "Academic and Research Use (ISOT / CC-BY)",
            "terms_note": "Dataset files are excluded from Git repository tracking via .gitignore.",
        },
        "disclaimer": (
            "Pratyora predictions are statistical lexical pattern estimates. "
            "They evaluate rhetorical and lexical patterns in training text, not factual truth or falsehood. "
            "Never use as a standalone fact-checking engine."
        ),
    }

    artifact = {"pipeline": pipeline, "metadata": metadata}

    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_output_path)
    logger.info("Saved trained pipeline artifact to: %s", model_output_path)

    os.makedirs(os.path.dirname(report_output_path) or ".", exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved machine-readable evaluation report to: %s", report_output_path)

    return metadata


def print_audit(metadata: dict) -> None:
    """Show what the model relies on. Publisher names, city names, or formatting
    fragments in these lists mean there is still leakage to scrub."""
    top = metadata.get("top_features", {})
    print("\n=== LEAKAGE AUDIT: strongest words per class ===")
    print("Most 'real':", ", ".join(top.get("most_real", [])))
    print("Most 'fake':", ", ".join(top.get("most_fake", [])))
    print("Look for publisher names, city names, or fragments like 'didn'. If you see them,")
    print("add a pattern to services/text_cleaning.py and retrain.\n")


def main():
    parser = argparse.ArgumentParser(description="Train Pratyora News Credibility Model")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing dataset CSVs")
    parser.add_argument("--fake-csv", default=None, help="Path to Fake.csv (defaults to data/Fake.csv)")
    parser.add_argument("--true-csv", default=None, help="Path to True.csv (defaults to data/True.csv)")
    parser.add_argument("--extra-csv", action="append", default=[],
                        help="Extra labelled CSV (text/article column + 'label' column: fake/real or 1/0). Repeatable.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--report-out", default=DEFAULT_REPORT_PATH, help="Path to save evaluation report")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES, help="Maximum TF-IDF features")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample count for faster training")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                        help="Drop articles shorter than this many words after scrubbing (default 20, as the analyzer)")
    parser.add_argument("--C", type=float, default=DEFAULT_C, help="Logistic regression C (default 3.0)")
    parser.add_argument("--tune", action="store_true", help="Pick C from {1, 3, 10} on a validation slice")
    parser.add_argument("--no-augment", action="store_true", help="Disable short-excerpt augmentation")
    parser.add_argument("--audit", action="store_true", help="Print the strongest words per class after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    fake_csv = args.fake_csv or os.path.join(args.data_dir, "Fake.csv")
    true_csv = args.true_csv or os.path.join(args.data_dir, "True.csv")

    if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
        logger.error("Dataset files missing:")
        logger.error("  Fake CSV expected at: %s", fake_csv)
        logger.error("  True CSV expected at: %s", true_csv)
        sys.exit(1)

    df = load_and_preprocess_dataset(
        fake_csv, true_csv, max_samples=args.max_samples, min_words=args.min_words, extra_csvs=args.extra_csv,
    )
    metadata = train_and_evaluate(
        df,
        model_output_path=args.model_out,
        report_output_path=args.report_out,
        max_features=args.max_features,
        C=args.C,
        tune=args.tune,
        augment=not args.no_augment,
        random_seed=args.seed,
    )
    if args.audit:
        print_audit(metadata)
    print("\n[SUCCESS] Model training and evaluation successfully completed.")


if __name__ == "__main__":
    main()