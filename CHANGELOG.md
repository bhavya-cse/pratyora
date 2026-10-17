# Pratyora — AI-Powered News Credibility Analysis Platform

> **“See Beyond the Claim. Seek the Evidence.”**  
> Inspired by the Sanskrit epistemological concept *pratyakṣa* (प्रत्यक्ष) — direct observation, perceptual awareness, and evidence verification before forming belief.

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen.svg)]()

---

## 1. Project Overview

**Pratyora** is an educational and portfolio-grade multi-modal news credibility analysis platform. It inspects submitted text, article screenshots, and links to news articles:
- **Text Analysis**: Tokenizes, cleans, and vectorizes claims to compute model-estimated credibility scores using a scikit-learn TF-IDF and Logistic Regression pipeline.
- **Image & Screenshot OCR**: Extracts text from headlines, social posts, tickers, and banners with Tesseract OCR, running extracted text through the ML model.
- **News Link Analysis**: Opens a pasted news URL, extracts the headline and article text, and runs that text through the same analysis and trusted-source verification as pasted text. Private and internal addresses are blocked, and pages that need a login or sit behind a paywall are reported clearly instead of guessed at.

---

## 2. Platform Architecture

```
                                  ┌──────────────────────────┐
                                  │      Pratyora Web UI     │
                                  │  (Vanilla JS + CSS SPA)  │
                                  └─────────────┬────────────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          │                     │                     │
                    POST /api/analyze    POST /api/analyze-image   POST /api/analyze-link
                          │                     │                     │
                          ▼                     ▼                     ▼
                  ┌───────────────┐     ┌───────────────┐     ┌───────────────────────┐
                  │ Text Analyzer │     │Image Processor│     │      Link Reader      │
                  │  Service      │     │  (Pillow+OCR) │     │ (safe fetch + article │
                  └───────┬───────┘     └───────┬───────┘     │      extraction)      │
                          │                     │             └───────────┬───────────┘
                          │                     ▼                         │
                          │             Tesseract OCR Engine              │
                          │                     │                         │
                          ▼                     ▼                         ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │          Scikit-Learn Machine Learning Pipeline           │
                  │    TF-IDF Vectorizer (1-2 N-grams) + LogisticRegression   │
                  └─────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │     Claim verification against trusted sources            │
                  └─────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │   Unified Credibility Dashboard (Percentages)             │
                  │   Likely Real / Likely Fake + Limitations Notice          │
                  └───────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Frontend**: HTML5 Semantic Markup, CSS3 Design System with CSS variables (Light & Dark mode support), Vanilla JavaScript (no heavy frontend frameworks), Jinja2 templates, Fetch API.
- **Backend**: Python 3.14, Flask 3.1, Werkzeug (strict upload limits & MIME validation), Python-dotenv.
- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`.
  - Feature extraction: TF-IDF vectorizer (unigrams + bigrams, English stop-word filtering, sublinear term frequency).
  - Classifier: Regularized Logistic Regression with L-BFGS solver.
- **Image Processing & Computer Vision**: `Pillow` (with decompression-bomb security safeguards), `pytesseract`, Tesseract OCR engine (v5.5+).
- **Link Reading**: `requests` (guarded page download) and `trafilatura` (article and headline extraction, with a built-in fallback reader).
- **Automated Testing**: `pytest` unit and integration tests.

---

## 4. Key Features

1. **Paste News Analysis**:
   - Analyzes claims up to 20,000 characters.
   - Character counter and instant sample loaders (Journalistic Real vs. Sensational Claim).
   - Real-time probability outputs (`predict_proba()`) rounded to two decimal places.
2. **Image & Screenshot Analysis**:
   - Drag-and-drop and file-picker upload for PNG, JPG, JPEG, and WEBP (max 5 MB).
   - High-contrast grayscale preprocessing for robust OCR extraction.
   - Guarded against malformed files and decompression bombs.
3. **News Link Analysis**:
   - Paste the link of a news article; Pratyora opens the page and reads its headline and article text.
   - Accepts `https://site.com/story` as well as `site.com/story`.
   - The extracted text is shown in the results, then analyzed and verified exactly like pasted text.
   - Reports clearly when a site blocks automated readers, needs a login, or has no readable article text, and suggests pasting the text instead.
4. **Unified Results Dashboard**:
   - Animated visual score bars (teal for Real, red for Fake).
   - Confidence levels, word counts, character counts, and analysis timestamps.
   - One-click text copying for extracted article text and OCR snippets.
   - Prominent limitation disclaimer.
5. **Privacy & Ephemeral Storage**:
   - Uploaded images are processed in temporary isolated directories (`tempfile.TemporaryDirectory`) and purged immediately upon completion or failure.
   - No uploaded media is permanently stored.
   - For links, the server fetches the page you give it; the article text is analyzed in memory and not stored.
   - Extracted text is never logged into routine server logs.

---

## 5. macOS Installation & Setup Guide

### Step 1: Verify Python Version
Ensure Python 3.11 or higher is installed (tested on Python 3.14):
```bash
python3 --version
```

### Step 2: Install System Dependencies (Tesseract)
Install the binary dependencies using Homebrew:
```bash
brew install tesseract
```

Verify the installation:
```bash
tesseract --version
```

### Step 3: Clone and Setup Virtual Environment
```bash
cd /path/to/pratyora
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default settings run locally without external API keys.

---

## 6. Model Training & Evaluation

Pratyora is trained on a comprehensive benchmark corpus based on the **ISOT Fake and Real News Dataset** (University of Victoria / Open Research News Corpus), consisting of 24,351 cleaned and validated articles.

### Dataset Structure & Acquisition
- **Real News (`data/True.csv`)**: 13,245 articles from verified journalistic coverage (Reuters and AP).
- **Fake News (`data/Fake.csv`)**: 11,106 articles flagged as deceptive, fabricated, or clickbait by independent fact-checking institutions (PolitiFact, Wikipedia unverified repositories).
- **Licensing & Terms**: Academic and Research Use (ISOT / CC-BY).
- **Git Exclusion**: In strict compliance with research licensing and repository hygiene, all dataset CSV files in `data/*.csv` are ignored by Git via `.gitignore`.

### Training Execution
To train the model:
```bash
python train_model.py
```

Optional arguments:
- `--max-features 25000`: Set maximum TF-IDF unigram and bigram vocabulary size (default: 25,000).
- `--max-samples <N>`: Train on an optional stratified subset for rapid testing.

### Verified Held-Out Test Evaluation Results
The dataset was split using an 80/20 stratified holdout:
- **Training Set**: 19,480 articles (10,596 Real, 8,884 Fake)
- **Held-Out Test Set**: 4,871 articles (2,649 Real, 2,222 Fake)

| Evaluation Metric | Baseline / Pratyora Score |
|---|---|
| **Majority Class Baseline** | **54.38%** |
| **Stratified Random Baseline** | **49.97%** |
| **Pratyora Held-Out Accuracy** | **97.23%** (vs 54.38% baseline) |
| **Macro F1-Score** | **0.9721** |
| **Weighted F1-Score** | **0.9723** |
| **Real Class Precision / Recall** | **97.47% / 97.43%** (F1: 97.45%) |
| **Fake Class Precision / Recall** | **96.94% / 96.98%** (F1: 96.96%) |

#### Held-Out Confusion Matrix
```
                    Predicted Real     Predicted Fake
Actual Real (2,649)     2,581 (97.4%)        68 (2.6%)
Actual Fake (2,222)        67 (3.0%)      2,155 (97.0%)
```

---

## 7. Running the Application

Activate the virtual environment and start the Flask server:
```bash
source .venv/bin/activate
python app.py
```
Open your browser and navigate to:
```
http://localhost:5001
```

---

## 8. API Endpoint Documentation

| Method | Endpoint | Description | Request Format |
|---|---|---|---|
| `GET` | `/` | Web application home interface | HTML |
| `GET` | `/api/health` | System and dependency availability | `application/json` |
| `POST` | `/api/analyze` | Text credibility analysis | JSON: `{"text": "..."}` |
| `POST` | `/api/analyze-image` | OCR extraction and image analysis | Multipart Form (`image` file) |
| `POST` | `/api/analyze-link` | Read a news article from a link, then analyze it | JSON: `{"url": "https://..."}` |

### Sample Response Structure (`POST /api/analyze`)
```json
{
  "success": true,
  "label": "Likely Real",
  "confidence": "Moderate",
  "fake_percentage": 37.72,
  "real_percentage": 62.28,
  "active_features": 14,
  "insufficient_evidence": false,
  "analyzed_characters": 95,
  "analyzed_words": 14,
  "timestamp": "2026-10-12T00:33:00Z",
  "model_type": "TF-IDF (1-2 ngrams) + LogisticRegression (L-BFGS)",
  "warning": "These scores are statistical model estimates based on lexical and stylistic patterns in the training data. They are not proof of factual truth or falsehood. Always verify critical claims through primary and independent journalistic sources."
}
```

---

## 9. Automated Testing Suite

Execute the automated test suite with pytest:
```bash
source .venv/bin/activate
pytest -v
```

### Test Coverage
The suite covers the Flask endpoints (including `/api/analyze-link`), the link reader (blocked private addresses, redirects to private addresses, unsupported social links, title cleaning, article extraction), text preprocessing and validation, image/OCR analysis, multilingual handling and model behavior. The link tests use mocks and need no internet connection.
Run `pytest -v` to see the current results.

---

## 10. Critical Distinctions and False-Positive Mitigation

### Why the False-Positive Problem Occurred and How It Was Resolved
1. **Root Cause**: When a model is trained on a tiny sample (e.g. 19 articles), its vocabulary lacks general journalistic terms. Unseen headlines produce an all-zero TF-IDF vector, reducing prediction to the model's intercept. If the intercept happens to be positive ($b_0 > 0$), the model unconditionally outputs $P(\text{Fake}) > 50\%$ for any unknown headline, producing false-positive "Likely Fake" verdicts on legitimate news.
2. **Mitigations Implemented**:
   - **Trained on 24,351 Records**: Rich 25,000-feature vocabulary covering broad political, economic, scientific, and international reporting.
   - **Evidence Sufficiency Gating**: Input text is evaluated for active vocabulary features ($X_{vec}.nnz$). If active feature count is zero or negligible, the system outputs `Insufficient Evidence` with neutral 50/50 scores rather than guessing.
   - **Decision Margin Calibration**: When probabilities fall within $\pm 5\%$ of neutral (45%–55%), the prediction is labeled `Inconclusive / Mixed Evidence` with `Low` confidence.
   - **Model Reliability Enforcement**: Models trained on $<100$ records or with $<65\%$ accuracy are automatically flagged as unreliable, downgrading confidence to `Low` with prominent warnings.

---

## 10. Critical Distinctions and Model Limitations

1. **Text-Based Pattern Detection Only**:
   - Pratyora’s classifier evaluates lexical, rhetorical, and stylistic markers learned from training data (e.g. clickbait syntax, emotional urgency, sensational punctuation vs. formal journalistic attributions).
   - **It does NOT perform live fact-checking:** A logically consistent claim written in formal journalistic tone may be classified as "Likely Real" even if the event never occurred. Conversely, an informal genuine eyewitness report may receive an elevated "Fake" score.
2. **No Deepfake or Visual Tampering Detection**:
   - OCR of a screenshot does NOT verify visual authenticity.
   - The platform does not check edited images, face swaps, or generative AI artifacts.
3. **Links Are Read As Plain Text**:
   - Pratyora reads only the article text from a page. Sites that block automated readers, need a login, sit behind a paywall, or load their text with JavaScript cannot be read; paste the text instead.
   - Social media posts (X, Facebook, Instagram, Telegram and similar) and video pages (YouTube and similar) are rejected up front. Paste the text or upload a screenshot instead.
4. **Out-of-Distribution Data**:
   - Breaking news events, satire (e.g., The Onion), political humor, and non-Western journalistic conventions may differ significantly from the training distribution.
5. **Verification Recommendation**:
   - Always verify critical claims through primary institutional records, reputable independent fact-checking services (Snopes, AP Fact Check, PolitiFact), and official statements.

---

## 11. Security & Privacy Safeguards

- **Safe Subprocess Execution**: The external binary (`tesseract`) is strictly invoked using parameter lists, preventing shell command injection attacks.
- **Safe Link Fetching (SSRF protection)**: Only `http`/`https` links on ports 80 and 443 are opened. Hosts must resolve to public internet addresses only (no localhost, private networks, or cloud-metadata addresses), every redirect is re-checked, and links containing a username or password are refused. Downloads have timeouts and a 2 MB size cap, and only HTML or text pages are accepted.
- **Rate Limiting**: `/api/analyze-link` allows 10 requests per minute and 100 per day per IP by default (`LINK_RATE_LIMIT` in `.env`). Set `RATELIMIT_STORAGE_URI` to a Redis URL when running more than one server worker.
- **Known limitation (DNS rebinding)**: the host is checked before the download and resolved again by the HTTP library, so a hostile DNS server could in theory return a different address the second time. Run the app without access to internal networks (firewall or container egress rules) if you deploy it publicly.
- **Decompression Bomb Protection**: Pillow's `MAX_IMAGE_PIXELS` limit is configured to reject decompression bombs, along with strict maximum image dimension constraints (5000px).
- **Resource Constraints**:
  - Max text length: 20,000 characters.
  - Max image size: 5 MB.
  - Max request size: 10 MB.
  - Max linked page size: 2 MB, with a 20-second overall fetch deadline.
  - Max article text analyzed from a link: 20,000 characters.
- **Ephemeral Storage**: All uploaded images and extracted intermediate files are stored in volatile temporary directories and deleted upon request termination.

---

## 12. Future Enhancements

- Live web retrieval and evidence cross-referencing with verified news archives.
- Multilingual OCR and classification (Hindi, Tamil, Bengali, Spanish, French).
- Fine-tuned transformer models (RoBERTa / DeBERTa) for nuanced context understanding.
- Reading JavaScript-rendered and paywalled pages, and links to social posts.
- Visual artifact and image provenance verification (C2PA metadata inspection).

---

## 13. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.