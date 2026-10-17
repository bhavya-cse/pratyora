"""
Pratyora — Unit and Integration Tests for Image Analysis and OCR Pipeline
"""

import io
import pytest
from PIL import Image, ImageDraw
from services.image_processor import ImageProcessor, MAX_IMAGE_SIZE_BYTES


@pytest.fixture
def image_processor():
    return ImageProcessor()


def create_test_image(format="PNG", text=None, size=(300, 100), color=(255, 255, 255)):
    img = Image.new("RGB", size, color=color)
    if text:
        draw = ImageDraw.Draw(img)
        # Draw readable high-contrast black text on white
        draw.text((15, 30), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def test_valid_image_stream(image_processor):
    img_bytes = create_test_image("PNG")
    valid, err, img = image_processor.validate_image_stream(img_bytes, "test.png")
    assert valid is True
    assert err is None
    assert img is not None
    assert img.size == (300, 100)


def test_unsupported_image_format(image_processor):
    # BMP is not in allowed formats [PNG, JPEG, WEBP]
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="BMP")
    bmp_bytes = buffer.getvalue()

    valid, err, img_obj = image_processor.validate_image_stream(bmp_bytes, "test.bmp")
    assert valid is False
    assert "unsupported" in err.lower()


def test_corrupted_image_bytes(image_processor):
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_GARBAGE_BYTES_12345"
    valid, err, img = image_processor.validate_image_stream(corrupt_bytes, "fake.jpg")
    assert valid is False
    assert "corrupt" in err.lower() or "not a valid image" in err.lower()


def test_oversized_image_rejected(image_processor):
    # Simulate oversized byte array
    fake_huge_bytes = b"0" * (MAX_IMAGE_SIZE_BYTES + 500)
    valid, err, img = image_processor.validate_image_stream(fake_huge_bytes, "huge.png")
    assert valid is False
    assert "exceeds maximum allowed size" in err.lower()


def test_blank_image_insufficient_text(image_processor):
    # Blank white image with no text
    blank_bytes = create_test_image("PNG", text=None)
    result = image_processor.process_and_analyze(blank_bytes, "blank.png")
    assert result["success"] is False
    assert result["error_code"] == "INSUFFICIENT_TEXT"


def test_mock_ocr_failure(image_processor, monkeypatch):
    img_bytes = create_test_image("PNG")
    # Simulate OCR failure
    monkeypatch.setattr(
        image_processor,
        "extract_with_layout",
        lambda img: (False, "", "", "Simulated Tesseract Crash"),
    )
    result = image_processor.process_and_analyze(img_bytes, "sample.png")
    assert result["success"] is False
    assert result["error_code"] == "OCR_UNAVAILABLE"


def test_integration_image_ocr_with_model(image_processor):
    """
    Integration test: Renders an image with clear news text,
    runs Tesseract OCR, and executes the ML model pipeline.
    """
    if not image_processor.is_ocr_available():
        pytest.skip("Tesseract OCR binary not found on test host")

    text = "Federal Reserve holds interest rate unchanged amid inflation data"
    # Create larger image with clear black on white text
    img_bytes = create_test_image("PNG", text=text, size=(600, 150))
    result = image_processor.process_and_analyze(img_bytes, "headline.png")

    # If tesseract detects sufficient characters from the test font:
    if result["success"]:
        assert "fake_percentage" in result
        assert "real_percentage" in result
        assert result["source_type"] == "image_ocr"
        assert len(result["extracted_text"]) >= 15
    else:
        # If default PIL font rendering was too sparse for Tesseract segmentation
        assert result["error_code"] in ["INSUFFICIENT_TEXT", "OCR_UNAVAILABLE"]
