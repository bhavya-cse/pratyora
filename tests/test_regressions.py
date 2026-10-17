"""
Pratyora regression tests - one test per real failure seen in screenshots.
Run:  pytest tests/test_regressions.py -v
Offline: the NLI model, Tesseract and translators are stubbed.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- stubs so the modules import without heavy dependencies -----------------------------------
if "pytesseract" not in sys.modules:
    m = types.ModuleType("pytesseract")
    m.pytesseract = types.SimpleNamespace(tesseract_cmd=None)
    m.Output = types.SimpleNamespace(DICT="dict")
    m.TesseractNotFoundError = Exception
    sys.modules["pytesseract"] = m
if "services.text_analyzer" not in sys.modules:
    ta = types.ModuleType("services.text_analyzer")
    ta.MIN_TEXT_LENGTH = 20
    ta.TextAnalyzer = object
    sys.modules["services.text_analyzer"] = ta

import services.claim_verifier as cv          # noqa: E402
import services.image_processor as ip         # noqa: E402


def L(text, left, top, h, right=None):
    return {"text": text, "left": left, "top": top, "right": right or left + len(text) * h // 2,
            "bottom": top + h, "h": h}


def _entails(monkeypatch):
    monkeypatch.setattr(cv, "_nli_is_multilingual", lambda: True)
    monkeypatch.setattr(cv, "_nli_probs",
                        lambda p, h: {"entailment": 0.97, "contradiction": 0.01, "neutral": 0.02})


def _src(title, text, domain, category, kind="web"):
    return {"kind": kind, "category": category, "title": title, "url": f"https://{domain}/x",
            "domain": domain, "published_date": "", "rating": "", "text": text}


# ---------------------------------------------------------------------------------------------
# 1. Headline selection on full-window screenshots (Loksatta, Factly, Manorama)
# ---------------------------------------------------------------------------------------------
def test_loksatta_headline_ignores_ads_tab_and_image_caption():
    lines = [
        L("'सरकार तरुणांचा आवाज दाबू शकत नाही...' दिल्लीत सीजेपीच्या समर्थनार्थ हजारो तरुण रस्त्यावर?", 1270, 150, 22, 2500),
        L("IQ Test For Adults", 120, 510, 50, 350),
        L("ज्ञानेश कुमार यांच्या विरोधात आंदोलनासाठी", 380, 490, 62, 1230),
        L("जंतरमंतरवर जमले हजारो लोक? रस्त्यावर चालायलाही", 380, 570, 62, 1450),
        L("नाही जागा; नक्की वाचा, व्हायरल VIDEO मागचं सत्य", 380, 650, 62, 1440),
        L("ताब्यात घेतले.", 380, 770, 28, 500),
        L("CJP आंदोलनाच्या नावाखाली", 1000, 1060, 75, 1460),
        L("दिवसातून एकदाच जेवल्याने वजन कमी होते की उलट वाढते?", 1513, 1020, 30, 1800),
    ]
    focus = ip.select_headline_block(lines, 2380)
    headline = focus.split("\n\n")[0]
    assert headline.startswith("ज्ञानेश कुमार")
    assert "दिवसातून" not in focus and "IQ" not in focus and "CJP" not in focus
    assert "सरकार तरुणांचा" not in focus            # browser tab title


def test_factly_telugu_headline():
    lines = [
        L("बांग्लादेशद हिंदू व्यक्तियोबरु तम्मन्नु रक्षिसुव", 650, 150, 22, 1180),
        L("‘ఫుడ్ జిహాద్’ పేరుతో వైరల్ అవుతున్న ఈ వీడియో నిజమైన ఘటన", 270, 260, 85, 2200),
        L("కాదు, ఇదొక స్క్రిప్టెడ్ వీడియో", 270, 350, 85, 1100),
    ]
    assert "ఫుడ్ జిహాద్" in ip.select_headline_block(lines, 2380)


def test_manorama_malayalam_headline_not_ad_or_tab():
    lines = [
        L("നിത അംബാനിക്ക് അമേരിക്കൻ അസോസിയേഷൻ ഓഫ് ഫിസിഷ്യൻസ് ഓഫ്", 1660, 150, 22, 2350),
        L("നിത അംബാനിക്ക് അമേരിക്കൻ അസോസിയേഷൻ", 385, 490, 85, 1880),
        L("ഓഫ് ഫിസിഷ്യൻസ് ഓഫ് ഇന്ത്യൻ ഒറിജിന്റെ", 385, 590, 85, 1650),
        L("ഹ്യൂമാനിറ്റേറിയൻ അവാർഡ്", 385, 690, 85, 1220),
        L("ഈ കേരളപ്പിറവിയിൽ കേരളം ഓടുന്നു", 2040, 580, 45, 2350),
    ]
    focus = ip.select_headline_block(lines, 2380)
    assert "അവാർഡ്" in focus and "കേരളപ്പിറവിയിൽ" not in focus


def test_english_only_page_returns_no_native_focus():
    assert ip.select_headline_block([L("Breaking news about the election today", 100, 100, 40)], 1000) == ""


# ---------------------------------------------------------------------------------------------
# 2. Manorama: real Times of India article was rejected by the entity check
# ---------------------------------------------------------------------------------------------
TOI = ("Title: Nita Ambani receives honour in the US for contributions to healthcare, education and "
       "social impact | India News. Ambani was honoured with the AAPI Humanitarian Award at a ceremony "
       "in Tampa, Florida.")
CLEAN = "Nita Ambani received the American Association of Physicians of Indian Origin's Humanitarian Award."
JUNK = ("Nita Ambani (1000611 Association of Physicians of... U 4 GLOBAL 1 GLOBAL US News Nita Ambani "
        "received the American Association of Physicians of Indian Origin's Humanitarian Award No.")


def test_acronym_snippet_counts_as_covering_full_name():
    assert cv._entity_overlap(cv._key_entities(CLEAN), TOI, CLEAN) == 1.0


def test_ocr_junk_entities_are_ignored():
    entities = cv._key_entities(JUNK)
    assert not ({"1000611", "1", "4", "global", "news"} & entities)


def test_toi_supports_even_with_polluted_claim(monkeypatch):
    _entails(monkeypatch)
    toi = _src("Nita Ambani receives honour in the US | The Times of India", TOI,
               "timesofindia.indiatimes.com", "trusted_news")
    pib = _src("Press Release: PIB", "VP lauds Indian origin physicians; India-US collaboration in health "
               "can reap benefits for the entire world", "pib.gov.in", "official")
    for claim in (CLEAN, JUNK):
        judged, _, _ = cv.judge_sources_nli(claim, [toi, pib], "")
        assert [j["stance"] for j in judged] == ["supports", "unrelated"]


# ---------------------------------------------------------------------------------------------
# 3. Factly: a debunk of a DIFFERENT scripted video must not count as support
# ---------------------------------------------------------------------------------------------
TE_CLAIM = ("‘ఫుడ్ జిహాద్’ పేరుతో వైరల్ అవుతున్న ఈ వీడియో నిజమైన ఘటన కాదు, ఇదొక స్క్రిప్టెడ్ వీడియో. "
            "చెరకు రసంలో హానికరమైన పదార్థం కలిపి అమ్ముతున్నాడంటూ వీధి వ్యాపారిపై ఒక వ్యక్తి దాడి చేస్తున్న వీడియో")
EN_CLAIM = "The video is not a real incident, it is a scripted video where a man attacks a street vendor."


def test_other_scripted_video_is_not_support(monkeypatch):
    _entails(monkeypatch)
    dowry = _src("కట్నం ఇవ్వకుంటే పెళ్లి మధ్యలోనే ఆపేస్తానని బెదిరిస్తున్న ఈ దృశ్యాలు నిజమైన ఘటనకు సంబంధించినవి కావు, ఇది ఒక స్క్రిప్టెడ్ వీడియో",
                 "బీహార్‌లో ఒక వ్యక్తి పెళ్లి పీటలపైనే కట్నం ఇవ్వాలంటూ బెదిరిస్తున్నాడంటూ ఒక వీడియోని షేర్ చేసిన పోస్ట్ వైరల్ అవుతుంది. ఇది స్క్రిప్టెడ్ వీడియో.",
                 "factly.in", "fact_checker")
    judged, _, _ = cv.judge_sources_nli(EN_CLAIM, [dowry], TE_CLAIM)
    assert judged[0]["stance"] == "unrelated"


def test_matching_factly_article_still_supports(monkeypatch):
    _entails(monkeypatch)
    right = _src("‘ఫుడ్ జిహాద్’ పేరుతో వైరల్ అవుతున్న ఈ వీడియో నిజమైన ఘటన కాదు, ఇదొక స్క్రిప్టెడ్ వీడియో",
                 "చెరకు రసంలో హానికరమైన పదార్థం కలిపి అమ్ముతున్నాడంటూ వీధి వ్యాపారిపై ఒక వ్యక్తి దాడి చేస్తున్న వీడియో. ఫుడ్ జిహాద్ అని ప్రచారం.",
                 "factly.in", "fact_checker")
    judged, _, _ = cv.judge_sources_nli(EN_CLAIM, [right], TE_CLAIM)
    assert judged[0]["stance"] == "supports"


# ---------------------------------------------------------------------------------------------
# 4. Garbage in -> "unclear", never a verdict; debunk headlines are recognised
# ---------------------------------------------------------------------------------------------
def test_garbled_text_is_flagged():
    assert cv._looks_garbled("5) 6 < 2 loksatta.com a) x 0) + Validate claim verifier")


def test_leading_ocr_markers_stripped():
    assert cv._strip_leading_junk("b) The video is going viral") == "The video is going viral"


def test_debunk_headlines_detected():
    assert cv.input_is_debunk("Thousands gathered at Jantar Mantar? Read this, the truth behind the viral video")
    assert cv.input_is_debunk("This video is not a real incident, it is a scripted video")
    assert not cv.input_is_debunk("Nita Ambani received the Humanitarian Award")


def test_no_support_means_unverified_not_real():
    scored = cv.score_sources([])
    assert scored["status"] == "unverified" and scored["real_percentage"] is None


def test_model_scores_hidden_for_ocr_results():
    result = {"success": True, "label": "Likely Fake", "source_type": "image_ocr",
              "real_percentage": 20.0, "fake_percentage": 80.0}
    verification = {"status": "unverified", "summary": "", "dialect": "", "claim_unclear": False}
    cv.apply_verification(result, verification)
    assert result["score_basis"] == "model_only"
    assert result["model_scores_reliable"] is False


# ---------------------------------------------------------------------------------------------
# 5. Manorama sports feature: an opinion headline is not a factual claim
# ---------------------------------------------------------------------------------------------
def test_opinion_headline_not_checkable_but_real_claims_are():
    assert not cv.claim_is_checkable("ACs, air travel, Kerala should learn, Odisha cares")
    assert cv.claim_is_checkable(CLEAN)
    assert cv.claim_is_checkable("Thousands gathered at Jantar Mantar? No room on the road; read this, "
                                 "the truth behind the viral video")
    assert cv.claim_is_checkable("Modi, Shah and Nadda meet in Delhi")
    assert cv.claim_is_checkable("Opposition says PM must resign")


def test_opinion_headline_triggers_no_search(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("search must not run for a non-claim")
    monkeypatch.setattr(cv, "search_fact_checks", boom)
    monkeypatch.setattr(cv, "search_web_for_news", boom)
    verification, _, _ = cv.verify_claim(
        "എ.സി റൂം, വിമാന യാത്ര, കേരളം കണ്ടുപഠിക്കണം, ഒഡീഷയുടെ കരുതൽ",
        english_text="ACs, air travel, Kerala should learn, Odisha cares", language_code="ml")
    assert verification["status"] == "unverified"
    assert verification["claim_checkable"] is False
    assert "opinion or feature" in verification["action"]["headline"]
    assert verification["supporting_sources"] == [] and verification["related_sources"] == []


def test_manorama_sports_feature_headline_and_lead():
    lines = [
        L("കേരളം കണ്ടുപഠിക്കണം, ഒഡീഷയുടെ കരുതൽ ! | മനോരമ ഓൺലൈൻ", 1660, 150, 22, 2350),
        L("എ.സി റൂം, വിമാന യാത്ര, കേരളം കണ്ടുപഠിക്കണം, ഒഡീഷയുടെ കരുതൽ", 335, 615, 62, 2045),
        L("ഭുവനേശ്വറിൽനിന്ന് ഷീന കെ.തോമസ്", 335, 800, 25, 810),
        L("കായികരംഗത്തെ അടിസ്ഥാനസൗകര്യ വികസനത്തിൽ കേരളത്തിന് പാഠമായി ഒഡീഷ മോഡൽ", 360, 1035, 28, 1570),
    ]
    focus = ip.select_headline_block(lines, 2380)
    head, _, lead = focus.partition("\n\n")
    assert head.startswith("എ.സി റൂം") and "മനോരമ" not in focus
    assert "ഒഡീഷ മോഡൽ" in lead

# ---------------------------------------------------------------------------------------------
# 6. The Hindu headline + standfirst: claim must be the headline; same-event coverage must count
# ---------------------------------------------------------------------------------------------
HINDU_PASTE = ("Supreme Court-appointed panel launches portal for complaints over July violence at Jantar Mantar\n"
               "Priority will be given to matters pertaining to the use of pellet guns, targeted violence, "
               "harassment of female protestors, and excessive violence and destruction of property occasioned on either side")
HINDU_CLAIM = "Supreme Court-appointed panel launches portal for complaints over July violence at Jantar Mantar"


def test_headline_is_not_glued_to_standfirst_or_cut():
    ex = cv.extract_claim(HINDU_PASTE)
    assert ex["claim_en"] == HINDU_CLAIM


def test_same_event_coverage_is_partial_not_unverified(monkeypatch):
    monkeypatch.setattr(cv, "_nli_is_multilingual", lambda: True)
    monkeypatch.setattr(cv, "_nli_probs", lambda p, h: {"entailment": 0.1, "contradiction": 0.02, "neutral": 0.88})
    srcs = [
        _src("Supreme Court HPEC probe: SC directs committee to probe pellet gun use, violence against women in CJP's Jantar Mantar case",
             "Pellet guns, cop injuries, excessive violence: SC sets CJP probe panel priorities for the Jantar Mantar protest violence in July.",
             "indiatoday.in", "trusted_news"),
        _src("Jantar Mantar protest: Supreme Court prioritises pellet-use probe, protects witnesses",
             "The Supreme Court has directed the high-powered enquiry committee constituted by it to prioritise the Jantar Mantar protest violence.",
             "telegraphindia.com", "trusted_news"),
    ]
    judged, _, _ = cv.judge_sources_nli(HINDU_CLAIM, srcs, "")
    assert {j["stance"] for j in judged} == {"related"}
    scored = cv.score_sources(judged)
    assert scored["status"] == "partially_verified"
    assert 50 < scored["real_percentage"] <= 65      # capped: details unconfirmed


def test_unrelated_topic_page_is_still_unrelated(monkeypatch):
    monkeypatch.setattr(cv, "_nli_is_multilingual", lambda: True)
    monkeypatch.setattr(cv, "_nli_probs", lambda p, h: {"entailment": 0.1, "contradiction": 0.02, "neutral": 0.88})
    who = _src("WHO addresses violence against women as a gender equality and health priority",
               "Preventing violence against women and girls continues to be a worldwide public health priority.",
               "who.int", "official")
    judged, _, _ = cv.judge_sources_nli(HINDU_CLAIM, [who], "")
    assert judged[0]["stance"] == "unrelated"


# ---------------------------------------------------------------------------------------------
# 7. NDTV paste: text pasted twice, Title Case headline, date missing from snippets
# ---------------------------------------------------------------------------------------------
_BLOCK = ("'Go, Demonstrate In US': Top Lawyer On 'Leave India' Notice To Imran Khan Actor Imran Khan, a US citizen "
          "and an Overseas Citizen of India (OCI) cardholder, will approach the Bombay High Court challenging a notice "
          "issued by the Mumbai Police directing him to leave India by October 28 for allegedly violating rules by "
          "participating in a political demonstration.")


def test_pasted_twice_is_deduplicated_and_cut_cleanly():
    claim = cv.extract_claim(_BLOCK + _BLOCK + " ")["claim_en"]
    assert claim.count("Go, Demonstrate") == 1
    assert claim.endswith("October 28")                  # no dangling "for allegedly" / "by the"
    assert len(claim) <= 300


def test_title_case_detected():
    assert cv._is_title_case("Imran Khan To Move Bombay HC Over Leave India Notice From Mumbai Police")
    assert not cv._is_title_case("Imran Khan will move the Bombay High Court over the notice.")


def test_snippet_missing_a_date_is_same_event_coverage(monkeypatch):
    monkeypatch.setattr(cv, "_nli_is_multilingual", lambda: True)
    monkeypatch.setattr(cv, "_nli_probs", lambda p, h: {"entailment": 0.1, "contradiction": 0.02, "neutral": 0.88})
    claim = cv.extract_claim(_BLOCK)["claim_en"]
    src = _src("Imran Khan To Move Bombay HC Over Leave India Notice",
               "The FRRO notice alleges that Khan violated provisions of the Immigration and Foreigners Act. "
               "Khan will move the Bombay High Court.", "ndtv.com", "trusted_news")
    other = _src("Imran Khan, Lekha Washington Among 400 Detained In Mumbai",
                 "Actor Imran Khan and his girlfriend were among more than 400 people detained.",
                 "hindustantimes.com", "trusted_news")
    judged, _, _ = cv.judge_sources_nli(claim, [src, other], "")
    assert judged[0]["stance"] == "related"
    assert judged[1]["stance"] == "unrelated"


# ---------------------------------------------------------------------------------------------
# 8. Five-article acceptance test (Chandrayaan-3 true / altered, 5G hoax, Hindi, opinion)
# ---------------------------------------------------------------------------------------------
C3 = "The Chandrayaan-3 Mission of India successfully landed near the Southern Pole of the Moon on 23 August 2023."
C3_2024 = "India's Chandrayaan-3 mission successfully soft-landed near the Moon's south pole on 23 August 2024, according to ISRO."
PIB = ("India's Mission Chandrayaan-3", "Chandrayaan-3 Mission of India successfully landed near the Southern Pole of the Moon on 23 August 2023.")


def _judge_all(monkeypatch, claim, sources):
    _entails(monkeypatch)
    return cv.judge_sources_nli(claim, sources, "")[0]


def test_true_claim_not_called_fake_by_off_topic_factchecks(monkeypatch):
    srcs = [
        _src(PIB[0], PIB[1], "static.pib.gov.in", "official"),
        _src("No, BBC did not criticize Chandrayaan 3. Old clip viral; false report about India's Moon landing",
             "Fact check: an old BBC clip is viral with a false claim that the BBC criticised the Chandrayaan 3 Moon landing.",
             "altnews.in", "fact_checker"),
        _src("Footage From Mars Falsely Shared As First Visuals From Chandrayaan-3 Moon landing",
             "Fact check: footage from Mars is falsely shared as the first visuals from Chandrayaan-3.",
             "boomlive.in", "fact_checker"),
    ]
    judged = _judge_all(monkeypatch, C3, srcs)
    assert [j["stance"] for j in judged][1:] == ["unrelated", "unrelated"]
    scored = cv.score_sources(judged)
    assert scored["status"] != "likely_false"


def test_official_support_blocks_likely_fake():
    j = lambda st, cat, d, kind="web": {"stance": st, "category": cat, "domain": d, "url": d, "kind": kind}
    scored = cv.score_sources([j("partial", "official", "pib.gov.in"), j("contradicts", "fact_checker", "altnews.in")])
    assert scored["status"] == "disputed"


def test_wrong_date_contradicts(monkeypatch):
    src = _src("Chandrayaan-3 lands on Moon", "India's Chandrayaan-3 mission soft-landed near the Moon's south "
               "pole on 23 August 2023, ISRO said. The Vikram lander touched down.", "indiatoday.in", "trusted_news")
    pib = _src(PIB[0], PIB[1], "pib.gov.in", "official")
    judged = _judge_all(monkeypatch, C3_2024, [src, pib])
    assert [j["stance"] for j in judged] == ["contradicts", "contradicts"]
    assert "different date" in judged[0]["reason"]
    assert cv.score_sources(judged)["status"] in ("likely_false", "questionable")


def test_same_date_is_not_a_conflict():
    assert cv._date_conflict(C3, "Landed on August 23, 2023 near the south pole") == ""
    assert cv._date_conflict(C3_2024, "Landed on August 23, 2023") != ""
    assert cv._date_conflict("Meeting held in 2024", "founded in 2019") == ""     # years alone never conflict


def test_clickbait_words_are_not_required_names():
    ents = cv._key_entities("SHOCKING: Health officials have confirmed that the new 5G mobile towers are spreading "
                            "the coronavirus, and thousands of people living near the towers have fallen ill.")
    assert "5g" in ents and "shocking" not in ents and "health" not in ents


def test_opinion_explainer_headline_not_checkable():
    assert not cv.claim_is_checkable("Why our cities deserve better monsoon planning")
    assert cv.claim_is_checkable("Why Chandrayaan-3 landed on 23 August 2023 near the Moon's south pole")