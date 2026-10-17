import re
from urllib.parse import urlparse


from services.trusted_sources import (
    FACT_CHECKERS, OFFICIAL, OFFICIAL_SUFFIXES, all_trusted_news,
)

# Built from services/trusted_sources.py so the lists can never drift apart.
TRUSTED_DOMAIN_HINTS = tuple(OFFICIAL_SUFFIXES) + tuple(OFFICIAL) + tuple(FACT_CHECKERS) + tuple(all_trusted_news())

STOPWORDS = set("""
a an the and or but if then than to of in on at by for from with about
is are was were be been being it its this that these those as into after
before over under has have had will would can could should may might
according reportedly says said news claim announced confirmed stated
""".split())


# Cross-language concept aliases.
# These cover common terms, not every possible sentence or language.
CONCEPT_ALIASES = {

    "chandrayaan3": [
        "chandrayaan-3", "chandrayaan 3", "chandrayaan3",
        "vikram lander", "pragyan rover",
        "चंद्रयान-3", "चंद्रयान 3",
        "చంద్రయాన్-3", "చంద్రయాన్ 3",
        "சந்திரயான்-3", "ಚಂದ್ರಯಾನ-3",
    ],
    "isro": [
        "isro", "indian space research organisation",
        "indian space research organization",
        "भारतीय अंतरिक्ष अनुसंधान संगठन",
        "భారత అంతరిక్ష పరిశోధనా సంస్థ",
    ],
    "moon_landing": [
        "soft landing", "lunar landing", "landed on the moon",
        "moon landing", "lunar surface",
        "चंद्रमा पर सफल लैंडिंग",
        "చంద్రునిపై ల్యాండింగ్",
    ],

    "el_nino": [
        "el nino", "elnino", "ఎల్ నినో", "ఎల్‌నినో",
        "ಎಲ್ ನಿನೊ", "எல் நினோ", "എൽ നിനോ",
    ],
    "telangana": [
        "telangana", "తెలంగాణ", "ತೆಲಂಗಾಣ",
        "தெலங்கானா", "తెలంగాణలో",
    ],
    "water_shortage": [
        "water shortage", "water scarcity", "water crisis",
        "drinking water shortage", "నీటి కొరత", "నీటి ఎద్దడి",
        "నీటి సమస్య", "पानी की कमी", "जल संकट",
        "ನೀರಿನ ಕೊರತೆ", "தண்ணீர் தட்டுப்பாடு", "ജലക്ഷാമം",
    ],
    "electricity_supply": [
        "electricity supply", "power supply", "power shortage",
        "power cuts", "electricity shortage", "విద్యుత్ సరఫరా",
        "విద్యుత్ కొరత", "కరెంటు సమస్యలు", "बिजली की कमी",
        "बिजली आपूर्ति", "ವಿದ್ಯುತ್ ಪೂರೈಕೆ",
        "மின்சார தட்டுப்பாடு", "വൈദ്യുതി വിതരണം",
    ],
    "demand": [
        "rising demand", "increasing demand", "growing demand",
        "పెరుగుతున్న డిమాండ్", "డిమాండ్ పెరుగుదల",
        "बढ़ती मांग", "ಹೆಚ್ಚುತ್ತಿರುವ ಬೇಡಿಕೆ",
        "அதிகரிக்கும் தேவை", "വർധിച്ച ആവശ്യം",
    ],
    "rainfall_deficit": [
        "rainfall deficit", "deficient rainfall", "below normal rainfall",
        "low rainfall", "drought", "వర్షాభావం", "తక్కువ వర్షపాతం",
        "कम बारिश", "कम वर्षा", "ಮಳೆಯ ಕೊರತೆ",
        "மழை பற்றாக்குறை", "മഴക്കുറവ്",
    ],
    "government_preparation": [
        "action plan", "preparedness", "officials directed",
        "government preparation", "contingency plan",
        "చర్యల ప్రణాళిక", "అధికారులను ఆదేశించారు",
        "सरकार ने तैयारी", "कार्ययोजना",
    ],
}


def _normalize(text):
    text = str(text or "").lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text):
    # Unicode-aware: preserves words written in Indian scripts.
    words = re.findall(r"[^\W_]+", _normalize(text), flags=re.UNICODE)
    return {
        word for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def _concepts(text):
    normalized = _normalize(text)
    found = set()

    for concept, aliases in CONCEPT_ALIASES.items():
        for alias in aliases:
            if _normalize(alias) in normalized:
                found.add(concept)
                break

    return found


def _is_trusted_domain(domain):
    domain = domain.lower().split(":")[0].removeprefix("www.")

    for hint in TRUSTED_DOMAIN_HINTS:
        if hint.startswith("."):
            if domain.endswith(hint):
                return True
        elif domain == hint or domain.endswith("." + hint):
            return True

    return False


def _valid_percentage(value):
    try:
        value = float(value)
        if 0 <= value <= 100:
            return value
    except (TypeError, ValueError):
        pass
    return None


def _fact_check_matches_claim(claim, result):
    """Check whether a published review is textually relevant to a claim."""
    if not isinstance(result, dict):
        return False

    review_text = " ".join([
        str(result.get("claim") or ""),
        str(result.get("title") or ""),
    ])

    claim_tokens = _tokens(claim)
    review_tokens = _tokens(review_text)
    shared_tokens = claim_tokens & review_tokens

    claim_concepts = _concepts(claim)
    review_concepts = _concepts(review_text)
    shared_concepts = claim_concepts & review_concepts

    # A shared topic alone is not sufficient: require some
    # additional textual overlap.
    if shared_concepts and len(shared_tokens) >= 2:
        return True

    # For claims without recognized concepts, require several
    # shared words and meaningful overlap with the shorter text.
    denominator = min(len(claim_tokens), len(review_tokens))

    if (
        len(shared_tokens) >= 3
        and denominator > 0
        and len(shared_tokens) / denominator >= 0.25
    ):
        return True

    return False

def assess_evidence(claim, web_search, fact_check=None, model_result=None):
    """
    Produce a cautious evidence-adjusted estimate.

    Concept matching can find relevant multilingual reporting, but cannot
    prove that an article supports every detail of a claim. The adjustment
    is therefore capped and explicitly marked as heuristic.
    """

    claim = str(claim or "").strip()
    web_search = web_search or {}
    fact_check = fact_check or {}
    model_result = model_result or {}

    results = web_search.get("results", [])
    if not isinstance(results, list):
        results = []

    original_real = _valid_percentage(
        model_result.get("real_percentage")
    )
    original_fake = _valid_percentage(
        model_result.get("fake_percentage")
    )

    if original_real is not None and original_fake is not None:
        # Normalize to ensure the two percentages total 100.
        total = original_real + original_fake
        if total > 0:
            original_real = original_real / total * 100
            original_fake = 100 - original_real

    claim_tokens = _tokens(claim)
    claim_concepts = _concepts(claim)

    relevant_sources = []
    supporting_candidates = []
    trusted_domains = set()
    independent_domains = set()
    seen_urls = set()

    for item in results:
        if not isinstance(item, dict):
            continue

        url = str(item.get("url") or "").strip()

        try:
            parsed = urlparse(url)
        except ValueError:
            continue

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue

        domain = parsed.hostname.lower().removeprefix("www.")
        normalized_url = f"{parsed.scheme}://{domain}{parsed.path}".rstrip("/").lower()

        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)

        title = str(item.get("title") or "Web source")
        content = str(item.get("content") or "")
        source_text = f"{title} {content}"

        source_tokens = _tokens(source_text)
        shared_tokens = claim_tokens & source_tokens
        overlap = (
            len(shared_tokens) / len(claim_tokens)
            if claim_tokens else 0.0
        )

        source_concepts = _concepts(source_text)
        shared_concepts = claim_concepts & source_concepts

        trusted = _is_trusted_domain(domain)

        # For multilingual claims, use shared concepts where possible.
        # Otherwise fall back to ordinary token overlap.
        if claim_concepts:
            relevant = len(shared_concepts) >= min(
                2, len(claim_concepts)
            )
            concept_match = bool(shared_concepts)
        else:
            relevant = (
                len(shared_tokens) >= 3 and overlap >= 0.15
            )
            concept_match = False

        if not relevant:
            continue

        source_record = {
            "title": title,
            "url": url,
            "domain": domain,
            "trusted_domain": trusted,
            "keyword_overlap": round(overlap * 100, 1),
            "matched_concepts": sorted(shared_concepts),
            "verification_status": "candidate_support_not_semantically_verified",
            "supports_claim": None,
            "contradicts_claim": None,
        }

        relevant_sources.append(source_record)

        # Only a provisional evidence candidate. It is not proof of truth.
        # An article discussing the same topic can still contradict details.
        if concept_match:
            supporting_candidates.append(source_record)
            independent_domains.add(domain)
            if trusted:
                trusted_domains.add(domain)
# Do not change the model score just because search results
# mention similar words or come from well-known domains.
# Keyword overlap is not semantic claim verification.

    adjustment = 0.0
    evidence_strength = "insufficient"

    if supporting_candidates:
        evidence_strength = "potentially_relevant_sources"

    adjusted_real = original_real
    adjusted_fake = original_fake

    fact_status = fact_check.get("status", "unavailable")
    fact_results = fact_check.get("results", [])

    if not isinstance(fact_results, list):
        fact_results = []

    relevant_fact_checks = [
        item for item in fact_results
        if _fact_check_matches_claim(claim, item)
    ]

    if not claim:
        status = "insufficient_input"
        message = "Enter a news claim or article to assess its evidence."

    elif (
        fact_status in ("success", "matched")
        and relevant_fact_checks
    ):
        status = "fact_check_found"
        message = (
            "A potentially relevant published fact-check was found. "
            "Its findings should be reviewed to determine whether they "
            "address every detail of the claim."
        )

    elif supporting_candidates:
        status = "cross_source_evidence_found"
        message = (
            "Multiple sources matched recognized claim concepts. "
            "The score adjustment is provisional: article meaning and "
            "all claim details have not been independently verified."
        )

    elif relevant_sources:
        status = "sources_found_unverified"
        message = (
            "Potentially relevant sources were found, but there is not "
            "enough cross-language evidence to adjust the score."
        )

    elif web_search.get("status") == "success":
        status = "no_relevant_sources"
        message = (
            "The search completed, but no sufficiently relevant sources "
            "were identified. This does not mean the claim is false."
        )

    else:
        status = "search_unavailable"
        message = (
            "A usable web search was not available. The original model "
            "estimate has been retained."
        )

    return {
        "supporting_sources": [],
        "contradicting_sources": [],
        "unverified_sources": relevant_sources,
        "semantic_verification_performed": False,
        "evidence_strength": evidence_strength,
        "model_real_percentage": original_real,
        "model_fake_percentage": original_fake,
        "evidence_adjustment": 0.0,
        "adjusted_real_percentage": adjusted_real,
        "adjusted_fake_percentage": adjusted_fake,
        "verification_limit": (
            "Sources are matched using keywords and a limited multilingual "
            "concept dictionary. Their meaning has not been verified against "
            "every detail of the claim. The model percentage is an experimental "
            "estimate, not a calibrated probability that the claim is true."
        ),
    }