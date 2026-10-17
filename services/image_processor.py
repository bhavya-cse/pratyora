"""
Pratyora — Image and Screenshot Analysis Service
Validates image uploads, protects against decompression bombs and malformed files,
performs multilingual OCR text extraction using Tesseract, and feeds extracted text
into the ML model (non-English text is translated by the TextAnalyzer).
"""

import io
import logging
import os
import re
import shutil
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageOps
import pytesseract

from services.text_analyzer import MIN_TEXT_LENGTH, TextAnalyzer

logger = logging.getLogger("PratyoraImageProcessor")

# Security safeguards against decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 25_000_000  # Max 25 megapixels
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_DIMENSION = 5000  # Max width/height in pixels
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}

# Tesseract language packs to use, joined with '+'. Only packs that are actually
# installed are used (see get_ocr_languages). Override with OCR_LANGS in .env.
DEFAULT_OCR_LANGS = "eng+hin+kan+tam+tel+mal+mar+ben+guj+pan+urd"


def find_tesseract_binary() -> Optional[str]:
    """
    Locates the tesseract executable across environment variables and standard paths.
    Particularly handles Apple Silicon Homebrew (/opt/homebrew/bin/tesseract) and Linux/macOS standard paths.
    """
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("tesseract")
    if which_path and os.path.isfile(which_path) and os.access(which_path, os.X_OK):
        return which_path

    candidates = [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
        "/bin/tesseract",
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    return None


# --------------------------------------------------------------------------- #
# Layout-aware OCR: find the article headline on a full-window screenshot
# --------------------------------------------------------------------------- #
def _is_native_script(text: str) -> bool:
    """True when at least half of the letters are non-ASCII (Devanagari, Telugu, Tamil ...)."""
    letters = [c for c in str(text or "") if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if not c.isascii()) / len(letters) >= 0.5


def build_ocr_lines(data: Dict[str, list], min_conf: float = 20.0):
    """Group pytesseract.image_to_data() words into lines with a bounding box and a text height.
    Returns (lines_in_reading_order, lines_sorted_top_to_bottom)."""
    groups: Dict[tuple, list] = {}
    for i, raw in enumerate(data.get("text", [])):
        word = str(raw or "").strip()
        if not word:
            continue
        try:
            if float(data["conf"][i]) < min_conf:
                continue
        except (TypeError, ValueError, KeyError):
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(
            (data["left"][i], data["top"][i], data["width"][i], data["height"][i], word)
        )

    ordered = []
    for words in groups.values():          # dict keeps Tesseract's reading order
        words.sort(key=lambda w: w[0])
        heights = sorted(w[3] for w in words)
        ordered.append({
            "text": " ".join(w[4] for w in words),
            "left": min(w[0] for w in words),
            "top": min(w[1] for w in words),
            "right": max(w[0] + w[2] for w in words),
            "bottom": max(w[1] + w[3] for w in words),
            "h": heights[len(heights) // 2],      # median word height = font size
        })
    return ordered, sorted(ordered, key=lambda l: (l["top"], l["left"]))


def select_headline_block(lines, page_width: int, max_lead_lines: int = 6) -> str:
    """
    Pick the article headline (and the lead paragraph under it) from OCR lines of a whole-window
    screenshot. Browser bars, tab titles, menus and bylines are small text, ads are mostly
    Latin, and sidebars sit outside the headline's column, so:

      * only native-script lines with 3+ words are candidates
      * the headline is the TOPMOST run of lines whose text height is close to the largest
        (a big image caption further down is therefore not mistaken for it)
      * the lead is the native text under it, in the same column, at least 30% of its height
    Returns headline and lead separated by a blank line, or '' when nothing convincing is found.
    """
    cands = [l for l in lines if _is_native_script(l["text"]) and len(l["text"].split()) >= 3]
    if not cands:
        return ""
    tallest = max(l["h"] for l in cands)
    # Seeds need 3+ words, but a headline's last line is often one or two words long, so any native
    # line of headline size may extend the run.
    big = sorted((l for l in lines if _is_native_script(l["text"]) and l["h"] >= 0.7 * tallest),
                 key=lambda l: l["top"])

    runs, run = [], [big[0]]
    for line in big[1:]:
        if line["top"] - run[-1]["bottom"] <= 1.3 * run[-1]["h"]:
            run.append(line)
        else:
            runs.append(run)
            run = [line]
    runs.append(run)

    head = next((r for r in runs if sum(len(l["text"].split()) for l in r) >= 6), None)
    if head is None:
        return ""

    head_h = sorted(l["h"] for l in head)[len(head) // 2]
    left = min(l["left"] for l in head)
    right = max(l["right"] for l in head)
    slack = 0.03 * page_width
    bottom = max(l["bottom"] for l in head)

    lead_cands = [l for l in lines if _is_native_script(l["text"]) and len(l["text"].split()) >= 2]
    lead, last_bottom = [], bottom
    for line in lead_cands:
        if line["top"] < bottom or line in head:
            continue
        if line["h"] >= 0.9 * head_h:      # another headline-sized block (image caption): stop
            break
        if (line["h"] >= 0.3 * head_h and line["left"] >= left - slack and line["right"] <= right + slack
                and line["top"] - last_bottom <= 8 * head_h):
            lead.append(line)
            last_bottom = line["bottom"]
            if len(lead) >= max_lead_lines:
                break

    headline_text = " ".join(l["text"] for l in head)
    lead_text = " ".join(l["text"] for l in lead)
    # blank line: claim extraction treats the headline and the lead as separate paragraphs
    return f"{headline_text}\n\n{lead_text}".strip()



class ImageProcessor:
    def __init__(self):
        self.tesseract_cmd = find_tesseract_binary()
        self._ocr_languages: Optional[str] = None
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            logger.info("Configured Tesseract OCR binary: %s", self.tesseract_cmd)
        else:
            logger.warning(
                "Tesseract binary not found in standard system paths. OCR features will report unavailable until installed."
            )

    def is_ocr_available(self) -> bool:
        cmd = self.tesseract_cmd or find_tesseract_binary()
        if cmd:
            self.tesseract_cmd = cmd
            pytesseract.pytesseract.tesseract_cmd = cmd
            return True
        return False

    def get_ocr_languages(self) -> str:
        """
        Returns the '+'-joined Tesseract language string, restricted to language packs
        that are installed on this machine (so a missing pack never crashes OCR).
        Falls back to English if none of the requested packs are installed.
        """
        if self._ocr_languages is not None:
            return self._ocr_languages

        requested = [
            lang.strip()
            for lang in os.getenv("OCR_LANGS", DEFAULT_OCR_LANGS).split("+")
            if lang.strip()
        ]
        try:
            installed = set(pytesseract.get_languages(config=""))
        except Exception as e:
            logger.warning("Could not list installed Tesseract languages: %s", e)
            installed = {"eng"}

        usable = [lang for lang in requested if lang in installed]
        missing = [lang for lang in requested if lang not in installed]
        if missing:
            logger.info("Tesseract language packs not installed (skipped): %s", ", ".join(missing))
        if not usable:
            usable = ["eng"]

        self._ocr_languages = "+".join(usable)
        logger.info("OCR languages in use: %s", self._ocr_languages)
        return self._ocr_languages

    def validate_image_stream(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """
        Validates the file bytes, verifying magic signatures and size limits.
        Rejects files exceeding size limits, non-image content, and decompression hazards.
        """
        if not file_bytes:
            return False, "Uploaded image file is empty.", None

        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            size_mb = round(len(file_bytes) / (1024 * 1024), 2)
            return (
                False,
                f"Image size ({size_mb} MB) exceeds maximum allowed size of 5 MB.",
                None,
            )

        try:
            image_stream = io.BytesIO(file_bytes)
            img = Image.open(image_stream)

            # Force Pillow to read and verify header integrity
            img.verify()

            # Re-open after verify() as Pillow requires
            image_stream.seek(0)
            img = Image.open(image_stream)

            fmt = (img.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                return (
                    False,
                    f"Unsupported image format: '{fmt}'. Supported formats are PNG, JPG, JPEG, and WEBP.",
                    None,
                )

            width, height = img.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                return (
                    False,
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION}px.",
                    None,
                )

            # Load into memory safely to break dependency on input stream
            img.load()
            return True, None, img

        except Image.DecompressionBombError:
            return False, "Image rejected: Exceeds safe decompression limits.", None
        except Exception as e:
            logger.warning("Invalid image file provided (%s): %s", filename, e)
            return (
                False,
                "The uploaded file is corrupt or not a valid image. Please provide a valid PNG, JPG, or WEBP image.",
                None,
            )

    def preprocess_for_ocr(self, img: Image.Image) -> Image.Image:
        """
        Preprocesses image for optimal Tesseract character recognition:
        converts to grayscale and normalizes contrast.
        """
        if img.mode != "L":
            img = img.convert("L")
        # Auto-contrast to enhance text edge distinction
        img = ImageOps.autocontrast(img, cutoff=2)
        return img

    def extract_text_from_image(self, img: Image.Image) -> Tuple[bool, str, Optional[str]]:
        """
        Executes multilingual Tesseract OCR on the given PIL Image.
        Returns (success, extracted_text, error_message).
        """
        if not self.is_ocr_available():
            return (
                False,
                "",
                "Tesseract OCR is not installed or could not be located on the system. On macOS, install with: 'brew install tesseract tesseract-lang'.",
            )

        try:
            processed = self.preprocess_for_ocr(img)
            # PSM 3: Fully automatic page segmentation (optimal for headlines, social posts, banners)
            raw_text = pytesseract.image_to_string(
                processed, lang=self.get_ocr_languages(), config="--psm 3"
            )
            cleaned_text = re.sub(r"[\r\n\t]+", " ", raw_text)
            cleaned_text = re.sub(r" {2,}", " ", cleaned_text).strip()

            return True, cleaned_text, None
        except pytesseract.TesseractNotFoundError:
            return (
                False,
                "",
                "Tesseract OCR executable was not found. Please install Tesseract on your host system.",
            )
        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return (
                False,
                "",
                f"An error occurred while extracting text from the image: {str(e)}",
            )

    def extract_with_layout(self, img: Image.Image) -> Tuple[bool, str, str, Optional[str]]:
        """
        One OCR pass that keeps the page layout.
        Returns (success, full_text_with_line_breaks, headline_focus_text, error_message).
        Line breaks matter: claim extraction tells headlines from page furniture line by line,
        and extract_text_from_image() flattens them.
        """
        if not self.is_ocr_available():
            return False, "", "", (
                "Tesseract OCR is not installed or could not be located on the system. "
                "On macOS, install with: 'brew install tesseract tesseract-lang'."
            )
        try:
            processed = self.preprocess_for_ocr(img)
            data = pytesseract.image_to_data(
                processed, lang=self.get_ocr_languages(), config="--psm 3",
                output_type=pytesseract.Output.DICT,
            )
            ordered, by_top = build_ocr_lines(data)
            full_text = "\n".join(l["text"] for l in ordered).strip()
            focus = select_headline_block(by_top, processed.size[0])
            return True, full_text, focus, None
        except pytesseract.TesseractNotFoundError:
            return False, "", "", "Tesseract OCR executable was not found. Please install Tesseract on your host system."
        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return False, "", "", f"An error occurred while extracting text from the image: {str(e)}"

    def process_and_analyze(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Complete pipeline for image analysis:
        1. Validate image
        2. Run OCR (multilingual)
        3. Pass extracted text to TextAnalyzer ML pipeline
        """
        valid, err_msg, img = self.validate_image_stream(file_bytes, filename)
        if not valid or img is None:
            return {
                "success": False,
                "error": err_msg or "Image validation failed.",
                "error_code": "INVALID_IMAGE",
                "filename": filename,
            }

        ocr_ok, full_text, focus_text, ocr_err = self.extract_with_layout(img)
        flat_full = re.sub(r"\s+", " ", full_text).strip()
        flat_focus = re.sub(r"\s+", " ", focus_text).strip()
        # Prefer the headline block of a screenshot; fall back to the whole page if it is too short.
        use_focus = len(flat_focus) >= MIN_TEXT_LENGTH
        # Keep line breaks in extracted_text: claim extraction needs them.
        extracted_text = focus_text if use_focus else full_text
        analysis_text = flat_focus if use_focus else flat_full

        logger.debug("OCR for %s: ok=%s text=%r", filename, ocr_ok, extracted_text)

        if not ocr_ok:
            return {
                "success": False,
                "error": ocr_err,
                "error_code": "OCR_UNAVAILABLE",
                "filename": filename,
                "extracted_text": "",
            }

        if len(analysis_text) < MIN_TEXT_LENGTH:
            return {
                "success": False,
                "error": (
                    f"Insufficient readable text extracted ({len(analysis_text)} characters found). "
                    f"Minimum required is {MIN_TEXT_LENGTH} characters. Please ensure the screenshot or "
                    "image contains clear news headlines or article text, or paste the text manually."
                ),
                "error_code": "INSUFFICIENT_TEXT",
                "filename": filename,
                "extracted_text": extracted_text,
            }

        # Analyze extracted text with ML model (translates non-English text automatically)
        analyzer = TextAnalyzer.get_instance()
        analysis_res = analyzer.analyze(analysis_text)

        logger.debug("Model analysis result: %s", analysis_res)

        if not analysis_res.get("success"):
            return {
                "success": False,
                "error": analysis_res.get("error"),
                "error_code": analysis_res.get("error_code", "ANALYSIS_FAILED"),
                "filename": filename,
                "extracted_text": extracted_text,
            }

        # Merge results with image context
        analysis_res["filename"] = filename
        analysis_res["extracted_text"] = extracted_text
        analysis_res["full_ocr_text"] = full_text
        analysis_res["ocr_focus"] = "headline" if use_focus else "full_page"
        analysis_res["source_type"] = "image_ocr"
        return analysis_res