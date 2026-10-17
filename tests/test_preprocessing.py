"""
Pratyora — Unit and Regression Tests for Text Preprocessing and Analysis Service
Includes tests for label mapping, probability calculations, evidence sufficiency,
margin calibration, and small dataset reliability flags.
"""

import pytest
from services.text_analyzer import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, TextAnalyzer


def test_clean_text():
    analyzer = TextAnalyzer.get_instance()
    raw = "   Headline:\t\tBreaking   News\r\n\r\nNew policy announced today.   "
    cleaned = analyzer.clean_text(raw)
    assert cleaned == "Headline: Breaking News New policy announced today."


def test_validate_input_valid():
    analyzer = TextAnalyzer.get_instance()
    valid_text = "This is a legitimate news article about the economy with sufficient characters."
    is_valid, err = analyzer.validate_input(valid_text)
    assert is_valid is True
    assert err is None


def test_validate_input_empty_and_short():
    analyzer = TextAnalyzer.get_instance()
    is_valid, err = analyzer.validate_input("")
    assert is_valid is False
    assert "too short" in err.lower()

    is_valid, err = analyzer.validate_input("Too short.")
    assert is_valid is False
    assert "too short" in err.lower()


def test_validate_input_none_and_non_string():
    analyzer = TextAnalyzer.get_instance()
    is_valid, err = analyzer.validate_input(None)
    assert is_valid is False

    is_valid, err = analyzer.validate_input(12345)
    assert is_valid is False


def test_validate_input_excessive_length():
    analyzer = TextAnalyzer.get_instance()
    long_text = "A" * (MAX_TEXT_LENGTH + 10)
    is_valid, err = analyzer.validate_input(long_text)
    assert is_valid is False
    assert "exceeds the maximum" in err.lower()


def test_validate_input_malformed_numbers_and_symbols():
    analyzer = TextAnalyzer.get_instance()
    # Has characters but no intelligible words
    symbols_only = "$$$$$ 1234567890 ++++ ---- !!!!!"
    is_valid, err = analyzer.validate_input(symbols_only)
    assert is_valid is False
    assert "readable words" in err.lower()


def test_percentage_calculation_and_sum():
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not yet trained")

    test_text = (
        "WASHINGTON (Reuters) - The Federal Reserve decided to hold interest rates steady "
        "following its monetary policy review on Wednesday, citing balanced employment metrics."
    )
    result = analyzer.analyze(test_text)
    assert result["success"] is True
    assert "fake_percentage" in result
    assert "real_percentage" in result
    assert isinstance(result["fake_percentage"], float)
    assert isinstance(result["real_percentage"], float)

    # Must approximately sum to 100.0%
    total = result["fake_percentage"] + result["real_percentage"]
    assert round(total, 1) == 100.0
    assert result["label"] in ["Likely Real", "Likely Fake", "Inconclusive / Mixed Evidence", "Insufficient Evidence"]
    assert "warning" in result


def test_label_mapping_and_classes():
    """
    Regression Test: Verifies that class 0 maps to Likely Real and class 1 maps to Likely Fake.
    Verifies that class probabilities are assigned to the correct labels.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    # In our trained artifact, classes are [0, 1]
    assert analyzer.real_class_index != analyzer.fake_class_index
    assert analyzer.metadata.get("class_mapping", {}).get(0) == "Likely Real"
    assert analyzer.metadata.get("class_mapping", {}).get(1) == "Likely Fake"


def test_insufficient_evidence_gating():
    """
    Regression Test: When input text has zero or negligible vocabulary overlap
    with the model's training distribution, the system MUST return Insufficient Evidence
    rather than asserting false-positive Likely Fake.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    nonsense_text = "Klaatu barada nikto zanzibar qwertyuiop asdfghjkl zxcvbnm"
    result = analyzer.analyze(nonsense_text)

    assert result["success"] is True
    assert result["label"] == "Insufficient Evidence"
    assert result["confidence"] == "Uncertain"
    assert result["fake_percentage"] == 50.0
    assert result["real_percentage"] == 50.0
    assert result["insufficient_evidence"] is True
    assert "insufficient statistical evidence" in result["warning"].lower()


def test_true_news_headline_no_false_positive():
    """
    Regression Test: Legitimate journalistic headlines must not be
    misclassified as Likely Fake due to sparse positive intercepts.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    real_headline = (
        "WASHINGTON (Reuters) - Federal Reserve holds benchmark interest rate steady "
        "following monetary policy meeting in Washington."
    )
    result = analyzer.analyze(real_headline)

    assert result["success"] is True
    assert result["label"] == "Insufficient Evidence"
    assert result["confidence"] == "Uncertain"
    assert result["fake_percentage"] == 50.0
    assert result["real_percentage"] == 50.0
    assert result["insufficient_evidence"] is True


def test_sensational_claim_correctly_flagged():
    """
    Regression Test: Sensational clickbait conspiracy claims must be classified as Likely Fake.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    fake_headline = (
        "SHOCKING BOMBSHELL: Leaked documents prove globalist elites put mind control nanobots "
        "in public drinking water! Share before censored by corrupt media!"
    )
    result = analyzer.analyze(fake_headline)

    assert result["success"] is True
    assert result["label"] == "Likely Fake"
    assert result["fake_percentage"] > result["real_percentage"]
    assert result["fake_percentage"] >= 55.0


def test_margin_calibration_inconclusive(monkeypatch):
    """
    Regression Test: Predictions within +/- 5% of 50% must be labeled Inconclusive / Mixed Evidence.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    # Mock predict_proba to return near 50-50 (e.g. 52% fake, 48% real)
    fake_idx = analyzer.fake_class_index
    real_idx = analyzer.real_class_index
    mock_probs = [0.0, 0.0]
    mock_probs[fake_idx] = 0.52
    mock_probs[real_idx] = 0.48

    monkeypatch.setattr(analyzer.pipeline, "predict_proba", lambda X: [mock_probs])
    # Ensure evidence sufficiency passes
    monkeypatch.setattr(analyzer, "evaluate_evidence_sufficiency", lambda text: (True, 10, 10))

    result = analyzer.analyze(
    "The government department announced a public policy review on Monday. "
    "Officials discussed the proposed changes during a scheduled meeting. "
    "The department will publish further details after consulting local authorities "
    "and reviewing the available evidence. The final report will explain the "
    "implementation timeline and the expected impact on public services."
)
    assert result["success"] is True
    assert result["label"] == "Inconclusive / Mixed Evidence"
    assert result["confidence"] == "Low"
    assert result["fake_percentage"] == 52.0
    assert result["real_percentage"] == 48.0


def test_small_dataset_reliability_flag(monkeypatch):
    """
    Regression Test: If metadata indicates an unreliable small sample or poor accuracy,
    analyzer must flag reliability warning and downgrade confidence to Low.
    """
    analyzer = TextAnalyzer.get_instance()
    if not analyzer.is_available():
        pytest.skip("Model not available")

    orig_meta = analyzer.metadata
    try:
        analyzer.metadata = {
            "reliability_status": "unreliable_small_sample",
            "train_records": 19,
            "metrics": {"accuracy": 0.40},
        }

        # Mock valid prediction with high percentage
        fake_idx = analyzer.fake_class_index
        real_idx = analyzer.real_class_index
        mock_probs = [0.0, 0.0]
        mock_probs[fake_idx] = 0.85
        mock_probs[real_idx] = 0.15

        monkeypatch.setattr(analyzer.pipeline, "predict_proba", lambda X: [mock_probs])
        monkeypatch.setattr(analyzer, "evaluate_evidence_sufficiency", lambda text: (True, 10, 10))

        result = analyzer.analyze(
            "This report describes a public policy announcement made by government officials "
            "during a scheduled meeting on Monday. The department explained the proposed "
            "changes, outlined the implementation timeline, and said that further details "
            "would be published after consultation with local authorities and affected groups."
        )
        assert result["success"] is True
        assert result["label"] == "Likely Fake"
        assert result["fake_percentage"] == 85.0
        assert result["real_percentage"] == 15.0
        assert result["confidence"] == "Low"
        assert result["reliability_warning"] is not None
    finally:
        analyzer.metadata = orig_meta


def test_model_unavailable_handling(monkeypatch):
    analyzer = TextAnalyzer.get_instance()
    orig_pipeline = analyzer.pipeline
    try:
        analyzer.pipeline = None
        monkeypatch.setattr(analyzer, "load_model", lambda: False)
        result = analyzer.analyze(
            "This report describes a public policy announcement made by government officials "
            "during a scheduled meeting on Monday. The department explained the proposed "
            "changes, outlined the implementation timeline, and said that further details "
            "would be published after consultation with local authorities and affected groups."
        )
        assert result["success"] is False
        assert result["error_code"] == "MODEL_UNAVAILABLE"
    finally:
        analyzer.pipeline = orig_pipeline
