"""
Pratyora — Source-artifact scrubbing shared by training and inference.

The ISOT dataset leaks its *source*: every real article is a Reuters wire story
("WASHINGTON (Reuters) - ...") and many fake ones carry web/social page furniture
("Featured image via Getty Images", "Via:", "WATCH:", t.co links). A model trained
on raw text learns 'Reuters = real, image/video = fake' instead of credibility
signals. Removing these artifacts BEFORE vectorizing (at train time and at
prediction time) forces the model to rely on wording and style instead.

Version 2 additions (found by auditing the corpus, see train_model.py --audit):
  * Datelines are removed ANYWHERE in the text, not just at the start. Training
    text is "title + body", so the dateline is never at position 0 and the old
    anchored pattern left city names (WASHINGTON, LONDON, ...) behind as a
    strong 'real' signal.
  * Apostrophes are normalised. The corpus renders "don't" as "don t" in 57% of
    fake but only 7% of real articles, which made fragments like "didn" and "isn"
    look like 'fake' words. Contractions are now glued to a single form.
  * Glued sentences ("laughable.The") are re-spaced (81% of fake vs 0.5% of real).
  * Reporter credits, "Via:", "WATCH:/VIDEO:" labels, "video below" boilerplate,
    "(1000 GMT)" timestamps and a few unreliable-site brand names are removed.
"""

import re
import unicodedata

_APOS = "’'`´"

_PATTERNS = [
    # Dateline ANYWHERE: "WASHINGTON (Reuters) -", "RAMALLAH, West Bank (Reuters) -",
    # "NEW YORK/WASHINGTON (Reuters) -", "(AP) —". Needs 2+ capitals to start so
    # ordinary words and "U.S." are not swallowed.
    (re.compile(r"\b[A-Z]{2,}(?:[ /.,&'-]+[A-Za-z][A-Za-z.'-]*){0,4}?\s*\((?:Reuters|AP|AFP)\)\s*[-–—]*"), " "),
    (re.compile(r"\(\s*(?:Reuters|AP|AFP)\s*\)", re.I), " "),
    (re.compile(r"\b(?:reuters|associated press|afp)\b", re.I), " "),
    # Reporter / editor credits at the end of wire stories
    (re.compile(r"\(?\s*(?:additional\s+)?(?:reporting|writing|editing|compiled)\s+by\b[^()]{0,160}\)?", re.I), " "),
    # URLs and social handles / hashtags
    (re.compile(r"(?:https?://|www\.)\S+", re.I), " "),
    (re.compile(r"\b\S+\.(?:com|org|net|co|in|uk)(?:/\S*)?\b", re.I), " "),
    (re.compile(r"pic\.twitter\.com/\S+", re.I), " "),
    (re.compile(r"[@#]\w+"), " "),
    # Web page furniture / image credits
    (re.compile(r"\b(?:featured\s+image|image\s+via|photo\s+by|photo\s+via|getty\s+images?|screen\s*shot|screengrab|"
                r"video\s+screenshot|read\s+more|share\s+this|watch\s+(?:the\s+)?video|"
                r"click\s+here|subscribe|advertisement|related\s+articles?|"
                r"the\s+post\s+.{0,120}?\s+appeared\s+first\s+on)\b", re.I), " "),
    # Photo credits in brackets: "(Photo- ITG)", "(Photo: Screengrab)", "(Image - PTI)"
    (re.compile(r"\(\s*(?:photo|image|pic|credit)s?\s*[-:\u2013]\s*[^)]{0,40}\)", re.I), " "),
    (re.compile(r"\bfollow\s+us(?:\s+on\s+[A-Za-z ]{1,30})?", re.I), " "),
    # Labels such as "WATCH:", "VIDEO:", "BREAKING:", "(VIDEO)", "Via:", "Source:"
    (re.compile(r"\b(?:watch|video|breaking|just\s+in|look|photos?|images?|via|source|credit)\s*[:|]", re.I), " "),
    (re.compile(r"\(\s*(?:video|photos?|images?|pics?)\s*\)", re.I), " "),
    (re.compile(r"\bvia\b", re.I), " "),
    # "watch the video below", "tweets below" and Reuters-style "(1000 GMT)" timestamps
    (re.compile(r"\b(?:video|clip|tweets?|post|photos?|images?)s?\s+below\b:?", re.I), " "),
    (re.compile(r"\(\s*\d{3,4}\s*(?:GMT|UTC|EST|EDT)\s*\)", re.I), " "),
    # Time-zone abbreviations (Reuters timestamps; no news content)
    (re.compile(r"\b(?:EDT|EST|CDT|CST|PDT|PST|GMT|UTC|IST)\b"), " "),
    # Unreliable-site brand names (publisher identity, not wording)
    (re.compile(r"\b(?:21st\s+century\s+wire|infowars|gateway\s+pundit|breitbart|daily\s+mail)\b", re.I), " "),
    # Weekday / month names (Reuters style says "on Tuesday", dataset-era artifact)
    (re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
                r"january|february|march|april|june|july|august|september|october|november|december)\b\.?",
                re.I), " "),
]

# Re-space glued sentences: "laughable.The" -> "laughable. The"
_GLUED = re.compile(r"(?<=[a-z0-9\)])([.!?])(?=[A-Z][a-z])")

# Apostrophe normalisation: "don't" / "don’t" / "don t" / "dont" -> "dont"
_APOS_INSIDE = re.compile(r"(?<=[A-Za-z])[%s](?=[A-Za-z])" % re.escape(_APOS))
_SPLIT_NT = re.compile(r"\b([A-Za-z]*n) t\b")
_SPLIT_CLITIC = re.compile(r"\b([A-Za-z]{2,}) (s|ll|ve|re)\b")
_SPLIT_I = re.compile(r"\bI (m|d|ve|ll)\b")

_WS = re.compile(r"\s+")


def strip_source_artifacts(text: str) -> str:
    if not text:
        return ""
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    text = _GLUED.sub(r"\1 ", text)
    text = _APOS_INSIDE.sub("", text)
    text = _SPLIT_NT.sub(r"\1t", text)
    text = _SPLIT_CLITIC.sub(r"\1\2", text)
    text = _SPLIT_I.sub(r"I\1", text)
    return _WS.sub(" ", text).strip()



# --------------------------------------------------------------------------- #
# Native-script (Hindi) news-page furniture, removed BEFORE language detection and
# translation. Otherwise bylines, "updated" stamps, share buttons and end markers are
# translated and scored as if they were news content (and counted towards the 20-word
# minimum).
#
# Source of these patterns: one outlet only (Aaj Tak article + homepage, Oct 2026).
# Other Hindi sites use different furniture; extend _NATIVE_PATTERNS as you meet them.
# --------------------------------------------------------------------------- #
_HI_MONTHS = ("जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|"
              "अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर")
_HI_DATE = r"\d{1,2}\s+(?:%s)\s+\d{4}" % _HI_MONTHS
_HI_CLOCK = r"(?:,?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*(?:IST)?)?"
_DEV = r"[\u0900-\u097F]+"

_NATIVE_PATTERNS = [
    # "(अपडेटेड 15 अक्टूबर 2026, 6:03 PM IST)"
    (re.compile(r"\(?\s*(?:अपडेटेड|प्रकाशित)\s*:?\s*" + _HI_DATE + _HI_CLOCK + r"\s*\)?"), " "),
    # Dateline: "प्रयागराज, 15 अक्टूबर 2026," (city + date), or a bare date
    (re.compile(r"(?:" + _DEV + r"(?:\s" + _DEV + r")?,\s*)?" + _HI_DATE + _HI_CLOCK + r",?"), " "),
    # End-of-article marker: "---- समाप्त ----"
    (re.compile(r"-{2,}\s*समाप्त\s*-{2,}"), " "),
    # Share / navigation / related-links labels
    (re.compile(r"(?:फेसबुक|टि्वटर|ट्विटर|कैंसिल|बुकमार्क|और\s+पढ़ें|सम्बंधित\s+ख\u093cबरें|"
                r"संबंधित\s+ख\u093cबरें|संबंधित\s+खबरें|ये\s+भी\s+देखें|यह\s+भी\s+पढ़ें|ये\s+भी\s+पढ़ें|"
                r"विज्ञापन)\s*[:：]?"), " "),
    (re.compile(r"सबसे\s+तेज\u093c?\s+ख\u093cबरों\s+के\s+लिए\s+आजतक\s+ऐप\s*डाउनलोड\s+करें"), " "),
    # Photo credits: "(Photo- ITGD)", "(फोटो: PTI)"
    (re.compile(r"\(\s*(?:Photo|फोटो|Image|तस्वीर)\s*[-:\u2013]\s*[^)]{0,40}\)", re.I), " "),
    # Publisher identity (not wording)
    (re.compile(r"आज\s?तक|इंडिया\s+टुडे|India\s+Today(?:\s+Group)?|Aaj\s?Tak|AajTak|\bITGD?\b", re.I), " "),
    (re.compile(r"\bFollow\s+us(?:\s+on\s+[A-Za-z ]{1,30})?", re.I), " "),
    (re.compile(r"\(\s*\)"), " "),
]


def strip_native_news_boilerplate(text: str) -> str:
    """Remove Hindi news-page furniture before detection/translation. Text without any
    of these patterns is returned unchanged apart from Unicode normalisation."""
    if not text:
        return ""
    # NFC splits composed nukta letters (U+0959 -> U+0916 U+093C) so one spelling matches.
    text = unicodedata.normalize("NFC", text)
    for pat, repl in _NATIVE_PATTERNS:
        text = pat.sub(repl, text)
    return _WS.sub(" ", text).strip()