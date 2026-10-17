"""
Pratyora — trusted source lists. EDIT THIS FILE to add or remove sources.

Rules for adding a source
  * Use the bare domain only ("example.com"), no https:// and no path. Subdomains match automatically
    (hindi.news18.com is covered by news18.com).
  * Add only editorially independent outlets. Do NOT add party newspapers, partisan portals or
    anonymous aggregators: every source in these lists counts as evidence for or against a claim.
  * Put an outlet under the language(s) it publishes in. REGIONAL_NEWS_BY_LANGUAGE drives the
    native-language search for a claim written in that language.

Language keys are ISO 639-1 codes: hi mr gu bn or as te ta kn ml pa ur.
"""

# --- National / English-language outlets ---------------------------------------------------------
NATIONAL_NEWS = [
    "ndtv.com", "indiatoday.in", "news18.com", "abplive.com", "indiatvnews.com",
    "thehindu.com", "indianexpress.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "zeenews.india.com", "deccanherald.com", "livemint.com", "business-standard.com",
    "ptinews.com", "thewire.in", "theprint.in", "scroll.in", "newindianexpress.com",
    "telegraphindia.com", "tribuneindia.com", "outlookindia.com", "deccanchronicle.com",
    "thehansindia.com", "mid-day.com", "freepressjournal.in", "etvbharat.com",
]

# --- Regional-language outlets, by language ---------------------------------------------------------
REGIONAL_NEWS_BY_LANGUAGE = {
    "hi": ["aajtak.in", "ndtv.in", "bhaskar.com", "jagran.com", "amarujala.com", "livehindustan.com",
           "navbharattimes.indiatimes.com", "jansatta.com", "patrika.com", "prabhatkhabar.com"],
    "mr": ["lokmat.com", "loksatta.com", "maharashtratimes.com", "esakal.com", "tv9marathi.com",
           "pudhari.news"],
    "gu": ["divyabhaskar.co.in", "gujaratsamachar.com", "sandesh.com", "tv9gujarati.com"],
    "bn": ["anandabazar.com", "eisamay.com", "bartamanpatrika.com", "sangbadpratidin.in",
           "tv9bangla.com"],
    "or": ["sambadenglish.com", "dharitri.com", "odishatv.in", "orissapost.com"],
    "as": ["pratidintime.com", "assamtribune.com", "sentinelassam.com"],
    "te": ["eenadu.net", "sakshi.com", "andhrajyothy.com", "ntvtelugu.com", "tv9telugu.com",
           "telanganatoday.com"],
    "ta": ["dinamalar.com", "dailythanthi.com", "dinamani.com", "hindutamil.in",
           "puthiyathalaimurai.com", "vikatan.com", "dinakaran.com", "maalaimalar.com"],
    "kn": ["prajavani.net", "vijaykarnataka.com", "kannadaprabha.com", "vijayavani.net",
           "tv9kannada.com", "udayavani.com", "publictv.in"],
    "ml": ["mathrubhumi.com", "manoramaonline.com", "asianetnews.com", "madhyamam.com",
           "mediaoneonline.com", "twentyfournews.com", "keralakaumudi.com"],
    "pa": ["punjabijagran.com", "jagbani.com", "ajitjalandhar.com", "babushahi.com"],
    "ur": ["urdupoint.com", "inquilab.com", "siasat.com"],
}

# --- Regional / state outlets that publish mostly in English ---------------------------------------
REGIONAL_ENGLISH_NEWS = [
    # North-east
    "eastmojo.com", "theshillongtimes.com", "nagalandpost.com", "arunachaltimes.in",
    "thesangaiexpress.com",
    # Jammu & Kashmir, Goa
    "greaterkashmir.com", "dailyexcelsior.com", "heraldgoa.in", "thegoan.net",
]

# --- Dedicated fact-checking organisations ----------------------------------------------------------
# Most publish in several Indian languages on the same domain (BOOM, Newschecker, Vishvas News,
# Fact Crescendo, Factly, NewsMeter ...).
FACT_CHECKERS = [
    "boomlive.in", "altnews.in", "factly.in", "vishvasnews.com", "newschecker.in",
    "newsmeter.in", "factcrescendo.com", "logicallyfacts.com", "dfrac.org",
    "newsmobile.in", "factcheck.afp.com", "youturn.in", "digiteye.in", "thip.media",
    "fullfact.org", "snopes.com", "politifact.com", "factcheck.org", "leadstories.com",
]

# General outlets that run dedicated fact-check sections. An article on one of these is promoted
# to "fact_checker" when its URL contains a fact-check path (see FACT_CHECK_PATH_HINTS).
FACT_CHECK_SECTION_SITES = [
    "indiatoday.in", "thequint.com", "thehindu.com", "indianexpress.com", "hindustantimes.com",
    "ndtv.com", "reuters.com", "apnews.com", "dw.com", "telugupost.com", "deccanherald.com",
]
FACT_CHECK_PATH_HINTS = ("/fact-check", "/factcheck", "/fact_check", "/webqoof", "/ap-fact-check")

# --- Official sources ---------------------------------------------------------------------------------
# Any *.gov.in / *.nic.in host (every state government, police, ministry) is also treated as official
# automatically; list here only official bodies that are NOT on those suffixes.
OFFICIAL = [
    "pib.gov.in", "factcheck.pib.gov.in", "mausam.imd.gov.in", "imd.gov.in", "rbi.org.in",
    "eci.gov.in", "isro.gov.in", "who.int", "mygov.in",
]
OFFICIAL_SUFFIXES = (".gov.in", ".nic.in", ".gov")

INTERNATIONAL_NEWS = ["reuters.com", "apnews.com", "bbc.com", "bbc.co.uk"]


def all_trusted_news():
    regional = [d for sites in REGIONAL_NEWS_BY_LANGUAGE.values() for d in sites]
    seen, out = set(), []
    for d in NATIONAL_NEWS + regional + REGIONAL_ENGLISH_NEWS + INTERNATIONAL_NEWS:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out