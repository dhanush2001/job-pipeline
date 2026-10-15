"""
locations.py -- map a job's location text to a canonical country name.

Two uses:
  1. Drop roles in countries we aren't targeting BEFORE any LLM call (no tokens
     spent tailoring a role in a country the candidate isn't pursuing).
  2. Bucket output by country: job-applications/<Country>/<Company>__<Role>/.

Adzuna jobs already know their country (the API is queried per-country), so the
scanner tags those directly; company-board jobs (Greenhouse/Lever/Ashby) carry
only a free-text location, which `country_of` resolves heuristically.
"""
from __future__ import annotations

import re

# Adzuna country code -> canonical display name (also the folder name).
# NB: Adzuna has NO endpoint for Hong Kong or Taiwan -- those two are detected
# from company-board locations only, never fetched from Adzuna.
ADZUNA_COUNTRY_NAME = {
    "us": "United States", "ca": "Canada", "gb": "United Kingdom",
    "de": "Germany", "nl": "Netherlands", "au": "Australia", "sg": "Singapore",
    "fr": "France", "it": "Italy", "es": "Spain", "pl": "Poland",
    "at": "Austria", "be": "Belgium", "ch": "Switzerland", "nz": "New Zealand",
    "ie": "Ireland", "in": "India",
}

REMOTE = "Remote"
OTHER = "Other"

# Country-name + major-city signals, checked in this order. The non-US countries
# are matched by their own names/cities FIRST so an ambiguous state-like token
# (e.g. "CA") is never mistaken -- "Toronto, ON, Canada" resolves via "canada",
# not via a state code.
_SIGNALS: list[tuple[str, list[str]]] = [
    ("Canada", ["canada", "toronto", "vancouver", "montreal", "montréal", "ottawa",
                "waterloo", "calgary", "edmonton", "ontario", "québec", "quebec"]),
    # NB: "cambridge"/"oxford" omitted on purpose -- they're also US cities
    # (Cambridge MA, Oxford MS), so they'd misroute US roles to the UK.
    ("United Kingdom", ["united kingdom", "england", "scotland", "wales", "london",
                        "manchester", "edinburgh", "bristol", "leeds", "glasgow"]),
    ("Germany", ["germany", "deutschland", "berlin", "munich", "münchen", "muenchen",
                 "hamburg", "frankfurt", "cologne", "köln", "stuttgart"]),
    ("Netherlands", ["netherlands", "amsterdam", "rotterdam", "the hague", "den haag",
                     "utrecht", "eindhoven", "holland"]),
    ("Australia", ["australia", "sydney", "melbourne", "brisbane", "perth",
                   "canberra", "adelaide"]),
    # --- Asia (visa-friendly tech hubs) ---
    ("Singapore", ["singapore"]),
    ("Hong Kong", ["hong kong", "hongkong"]),
    ("Taiwan", ["taiwan", "taipei", "hsinchu"]),
    # --- Europe. Favor country NAMES + unambiguous capitals; risky US-colliding
    # city names (Vienna VA, Rome GA, Valencia CA, Warsaw IN) are left out. ---
    ("France", ["france", "paris", "lyon", "toulouse"]),
    ("Ireland", ["ireland", "dublin", "cork", "galway"]),
    ("Switzerland", ["switzerland", "zurich", "zürich", "geneva", "lausanne"]),
    ("Sweden", ["sweden", "stockholm", "gothenburg"]),
    ("Denmark", ["denmark", "copenhagen"]),
    ("Norway", ["norway", "oslo"]),
    ("Finland", ["finland", "helsinki"]),
    ("Portugal", ["portugal", "lisbon", "porto"]),
    ("Spain", ["spain", "madrid", "barcelona"]),
    ("Italy", ["italy", "italia", "milan", "milano", "turin"]),
    ("Poland", ["poland", "warszawa", "kraków", "krakow", "wrocław", "wroclaw", "gdańsk"]),
    ("Belgium", ["belgium", "brussels"]),
    ("Austria", ["austria", "wien"]),
    ("United States", ["united states", "u.s.a", "u.s.a.", "u.s.", "new york",
                       "san francisco", "seattle", "austin", "boston", "chicago",
                       "los angeles", "mountain view", "palo alto", "sunnyvale",
                       "denver", "atlanta", "washington", "san jose", "bellevue",
                       "redmond", "cupertino", "menlo park", "san diego",
                       "los gatos", "nyc", "bay area"]),
]

# Recognizable but NOT targeted -> resolve to OTHER so the caller drops them
# (rather than mislabeling a foreign role as a target country).
_OTHER_SIGNALS = [
    "india", "bangalore", "bengaluru", "mumbai", "hyderabad", "pune", "gurgaon",
    "gurugram", "noida", "new delhi", "chennai", "japan", "tokyo", "osaka",
    "china", "beijing", "shanghai", "shenzhen", "brazil", "são paulo",
    "sao paulo", "mexico", "guadalajara", "israel", "tel aviv", "south africa",
    "philippines", "manila", "korea", "seoul", "dubai", "uae", "abu dhabi",
    "argentina", "colombia", "bogota", "romania", "bucharest", "vietnam",
    "indonesia", "jakarta", "malaysia", "thailand", "bangkok",
]

# US state abbreviations (for "City, XX" strings). "ca"/"wa"/"in"/"co" etc. are
# ambiguous, so this is a LAST resort, only after every name/city check above.
_US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
}
_STATE_RE = re.compile(r",\s*([a-z]{2})\b")


def _has(text: str, needle: str) -> bool:
    # Word-ish boundary so "us" doesn't match "campus" and "or" doesn't match "work".
    return re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", text) is not None


def country_of(location: str) -> str:
    """
    Resolve a free-text location to a canonical country name, or REMOTE when it
    only says remote with no country, or OTHER when it's a recognizable foreign
    place or can't be resolved. OTHER is the signal to drop a role.
    """
    t = (location or "").strip().lower()
    if not t:
        return REMOTE
    for country, signals in _SIGNALS:
        if any(_has(t, s) for s in signals):
            return country
    # Foreign place named -> Other (so it's dropped, not mislabeled).
    if any(_has(t, s) for s in _OTHER_SIGNALS):
        return OTHER
    # "City, XX" US state abbreviation, last resort.
    m = _STATE_RE.search(t)
    if m and m.group(1) in _US_STATES:
        return "United States"
    # Bare US / USA tokens.
    if _has(t, "usa") or _has(t, "us"):
        return "United States"
    if "remote" in t:
        return REMOTE
    return OTHER


def is_targeted(country: str, targets: list[str]) -> bool:
    """Keep a job if its country is a target, or if it's Remote (country-agnostic,
    so we don't lose a US-remote role whose text just says 'Remote')."""
    if not targets:
        return True
    return country in targets or country == REMOTE
