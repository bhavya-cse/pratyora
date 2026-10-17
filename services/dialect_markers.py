"""
Pratyora — Bhojpuri marker detection (no heavy dependencies).

Bhojpuri has no ISO 639-1 code and is written in Devanagari like Hindi, so statistical language
detectors (langdetect, translators) report it as Hindi. This module scores Devanagari text on
common Bhojpuri words and verb endings that standard Hindi does not use. Matching is whole-word
(so Hindi "जुबानी" does not match "बानी").
"""
import re

# Weight 2: distinctive Bhojpuri words (one is nearly enough on its own).
_STRONG = {
    "रउआ", "रउरा", "रउरे", "हमनी", "हमनीं", "तोहनी", "तोहरा", "तोहार", "तोहके",
    "बाड़े", "बाड़ें", "बाड़ी", "बाड़न", "बाड़ऽ", "बाटे", "बाटें", "बाटी", "बिया", "हउवे",
    "भइल", "भईल", "भइली", "कइल", "कईल", "कइले", "कईले", "गइल", "गईल", "गइली", "गइलऽ",
    "दिहल", "देहल", "लिहल", "रहल", "कहलें", "कहलस", "बतवले", "बतावल", "होखे", "होखी",
    "होखता", "चाहतानी", "नइखे", "नईखे", "एगो", "एहिजा", "ओहिजा", "कइसन", "कइसे",
}
# Weight 1: common in Bhojpuri but also seen in other Devanagari languages / dialects.
_WEAK = {"बानी", "एकर", "ओकर", "एह", "काहे", "होई", "जाई", "ओकरा", "एकरा", "सभे", "हमके", "हमार"}

_SPLIT = re.compile(r"[\s,.;:!?()\[\]\"'“”‘’|/\\\-–—।॥]+")
_CLAUSE_END_BA = re.compile(r"(?:^|\s)बा\s*(?:[।.!?,]|$)")   # "...भइल बा।" — the Bhojpuri copula
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text):
    letters = re.findall(r"[^\W\d_]", str(text or ""))
    if not letters:
        return False
    return sum(1 for c in letters if _DEVANAGARI.match(c)) / len(letters) >= 0.5


def bhojpuri_score(text):
    """Higher = more Bhojpuri. Hindi text normally scores 0."""
    text = str(text or "")
    tokens = [t for t in _SPLIT.split(text) if t]
    score = sum(2 for t in tokens if t in _STRONG) + sum(1 for t in tokens if t in _WEAK)
    if _CLAUSE_END_BA.search(text) or (tokens and tokens[-1] == "बा"):
        score += 2
    return score


def looks_bhojpuri(text, detected_lang=None):
    """True when Devanagari text is probably Bhojpuri. Marathi (also Devanagari) needs stronger proof."""
    if not has_devanagari(text):
        return False
    need = 4 if str(detected_lang or "").lower() == "mr" else 2
    return bhojpuri_score(text) >= need