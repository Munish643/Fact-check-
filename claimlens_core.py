from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


MAX_TEXT_CHARS = 90_000
MAX_CLAIMS = 10
MAX_EVIDENCE = 6

LLAMA_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


@dataclass
class Claim:
    id: str
    text: str
    category: str


@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    provider: str


@dataclass
class Verdict:
    status: str
    confidence: float
    correction: str
    rationale: str


CLAIM_KEYWORDS = (
    "percent",
    "percentage",
    "revenue",
    "profit",
    "sales",
    "growth",
    "market",
    "share",
    "valuation",
    "worth",
    "users",
    "customers",
    "downloads",
    "employees",
    "population",
    "emissions",
    "accuracy",
    "latency",
    "speed",
    "founded",
    "launched",
    "released",
    "as of",
    "according to",
    "reported",
    "ranked",
    "largest",
    "first",
    "increase",
    "decrease",
    "million",
    "billion",
    "trillion",
    "usd",
    "dollars",
    "technical",
    "capacity",
)

STOP_WORDS = {
    "about",
    "after",
    "against",
    "also",
    "among",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "company",
    "could",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "said",
    "same",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "year",
    "years",
}

NUMBER_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|(?:[$€£]\s*)?\b\d+(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?:%|percent|percentage points|million|billion|trillion|thousand|bn|mn|usd|dollars|users|customers|employees|people|countries|x|times)?",
    re.I,
)


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return re.sub(r"\s+", " ", "\n\n".join(pages)).strip()


def extract_claims(text: str, llama_config: dict[str, str] | None = None) -> tuple[list[Claim], str]:
    clipped = text[:MAX_TEXT_CHARS]
    if llama_config and llama_config.get("api_key"):
        try:
            claims = extract_claims_with_llama(clipped, llama_config)
            if claims:
                return claims[:MAX_CLAIMS], "Llama API"
        except Exception:
            pass
    return heuristic_extract_claims(clipped), "Heuristic fallback"


def extract_claims_with_llama(text: str, llama_config: dict[str, str]) -> list[Claim]:
    prompt = {
        "task": "Extract measurable factual claims from a marketing PDF.",
        "rules": [
            "Return only statistics, dates, financial figures, and technical figures.",
            "Ignore slogans, opinions, and vague claims without a measurable value.",
            "Return strict JSON with key claims.",
            "Use categories: Statistics, Financial, Date, Technical, General.",
            f"Return at most {MAX_CLAIMS} claims.",
        ],
        "schema": {"claims": [{"claim": "string", "category": "Statistics"}]},
        "text": text[:18_000],
    }
    data = call_llama_json(
        llama_config,
        system="You extract precise fact-checkable claims and return strict JSON.",
        user=json.dumps(prompt, ensure_ascii=False),
    )
    raw_claims = data.get("claims", []) if isinstance(data, dict) else []
    claims: list[Claim] = []
    for index, item in enumerate(raw_claims):
        claim_text = str(item.get("claim", "")).strip()
        if len(claim_text) < 20:
            continue
        category = normalize_category(str(item.get("category", "General")))
        claims.append(Claim(id=f"claim-{index + 1}", text=claim_text[:360], category=category))
    return dedupe_claims(claims)


def heuristic_extract_claims(text: str) -> list[Claim]:
    normalized = re.sub(r"\s+", " ", text.replace("-\n", "")).strip()
    sentences = re.findall(r"[^.!?;:]+[.!?;:]?", normalized)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        clean = sentence.strip()
        if len(clean) < 35 or len(clean) > 360:
            continue
        numbers = NUMBER_RE.findall(clean)
        if not numbers:
            continue
        lowered = clean.lower()
        keyword_hits = sum(1 for keyword in CLAIM_KEYWORDS if keyword in lowered)
        has_specific_value = any(not re.fullmatch(r"(?:19|20)\d{2}", n.strip()) for n in numbers)
        score = len(numbers) * 3 + keyword_hits * 1.6 + (3 if has_specific_value else 0)
        if re.search(r"\b(as of|according to|reported|estimated|projected|forecast|by 20\d{2}|in 20\d{2})\b", clean, re.I):
            score += 2
        if score >= 5:
            scored.append((score, clean_sentence(clean)))
    claims = [
        Claim(id=f"claim-{index + 1}", text=sentence, category=categorize_claim(sentence))
        for index, (_, sentence) in enumerate(sorted(scored, reverse=True)[:MAX_CLAIMS])
    ]
    return dedupe_claims(claims)


def verify_claim(claim: Claim, llama_config: dict[str, str] | None = None) -> dict[str, Any]:
    evidence = collect_evidence(claim)
    verdict = world_population_override(claim, evidence)
    verifier = "Deterministic evidence rules"
    if verdict is None and llama_config and llama_config.get("api_key"):
        try:
            verdict = verify_with_llama(claim, evidence, llama_config)
            verifier = "Llama API"
        except Exception:
            verdict = None
    if verdict is None:
        verdict = heuristic_verdict(claim, evidence)
        verifier = "Heuristic fallback"

    return {
        "id": claim.id,
        "claim": claim.text,
        "category": claim.category,
        "status": verdict.status,
        "confidence": max(0.0, min(1.0, verdict.confidence)),
        "correction": verdict.correction,
        "rationale": verdict.rationale,
        "verifier": verifier,
        "sources": [evidence_to_dict(item) for item in evidence[:MAX_EVIDENCE]],
    }


def verify_with_llama(claim: Claim, evidence: list[Evidence], llama_config: dict[str, str]) -> Verdict:
    payload = {
        "claim": claim.text,
        "category": claim.category,
        "evidence": [evidence_to_dict(item) for item in evidence[:MAX_EVIDENCE]],
        "instructions": [
            "Use only the supplied evidence.",
            "Status must be exactly Verified, Inaccurate, or False.",
            "Verified means evidence supports the same key value.",
            "Inaccurate means evidence supports the topic but shows a conflicting value.",
            "False means evidence is absent or too weak to support the claim.",
            "When inaccurate, state the corrected fact and source in correction.",
            "Never invent facts outside the evidence list.",
        ],
        "schema": {
            "status": "Verified|Inaccurate|False",
            "confidence": 0.0,
            "correction": "string",
            "rationale": "string",
        },
    }
    data = call_llama_json(
        llama_config,
        system="You are a careful fact-checking judge. Return strict JSON only.",
        user=json.dumps(payload, ensure_ascii=False),
    )
    status = normalize_status(str(data.get("status", "False")))
    try:
        confidence = float(data.get("confidence", 0.62))
    except (TypeError, ValueError):
        confidence = 0.62
    correction = str(data.get("correction", "")).strip() or default_correction(status)
    rationale = str(data.get("rationale", "")).strip() or "Llama judged the claim against supplied evidence."
    return Verdict(status=status, confidence=confidence, correction=correction, rationale=rationale)


def call_llama_json(llama_config: dict[str, str], system: str, user: str) -> dict[str, Any]:
    endpoint = llama_config.get("base_url") or LLAMA_DEFAULT_ENDPOINT
    model = llama_config.get("model") or LLAMA_DEFAULT_MODEL
    headers = {
        "Authorization": f"Bearer {llama_config['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://munish.streamlit.app/",
        "X-Title": "ClaimLens",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(endpoint, headers=headers, json=body, timeout=35)
    if response.status_code == 400:
        body.pop("response_format", None)
        response = requests.post(endpoint, headers=headers, json=body, timeout=35)
    response.raise_for_status()
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return parse_json_object(content)


def collect_evidence(claim: Claim) -> list[Evidence]:
    sources: list[Evidence] = []
    sources.extend(specialized_evidence(claim))
    for query in build_queries(claim.text):
        sources.extend(search_wikipedia(query))
        sources.extend(search_duckduckgo_lite(query))
        if len(sources) >= MAX_EVIDENCE * 2:
            break
    return dedupe_evidence(sources)[:MAX_EVIDENCE]


def specialized_evidence(claim: Claim) -> list[Evidence]:
    lowered = claim.text.lower()
    if "population" not in lowered or not re.search(r"\b(global|world)\b", lowered):
        return []
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", claim.text)
    row = fetch_world_population(year_match.group(1) if year_match else None)
    if not row:
        return []
    year, value = row
    return [
        Evidence(
            title=f"World Bank - World population {year}",
            url="https://data.worldbank.org/indicator/SP.POP.TOTL?locations=1W",
            snippet=f"World Bank reports world population in {year} as {format_large_number(value)} people.",
            provider="World Bank API",
        )
    ]


def fetch_world_population(year: str | None = None) -> tuple[str, float] | None:
    date_param = f"&date={year}" if year else "&per_page=5"
    url = f"https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL?format=json{date_param}"
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "ClaimLens/1.0"})
        response.raise_for_status()
        rows = response.json()[1]
        for row in rows:
            if row.get("value") is not None:
                return str(row["date"]), float(row["value"])
    except Exception:
        return None
    return None


def search_wikipedia(query: str) -> list[Evidence]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "format": "json", "srlimit": 4, "srsearch": query}
    try:
        response = requests.get(url, params=params, timeout=8, headers={"User-Agent": "ClaimLens/1.0"})
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
    except Exception:
        return []
    evidence = []
    for result in results:
        title = result.get("title", "Wikipedia result")
        evidence.append(
            Evidence(
                title=title,
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                snippet=strip_html(result.get("snippet", "")),
                provider="Wikipedia",
            )
        )
    return evidence


def search_duckduckgo_lite(query: str) -> list[Evidence]:
    try:
        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            timeout=10,
            headers={"User-Agent": "ClaimLens/1.0"},
        )
        response.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select("a.result-link") or soup.select("a")
    evidence: list[Evidence] = []
    for link in links[:5]:
        title = link.get_text(" ", strip=True)
        href = link.get("href", "")
        if not title or not href.startswith("http"):
            continue
        evidence.append(Evidence(title=title, url=href, snippet=title, provider="DuckDuckGo Lite"))
    return evidence


def heuristic_verdict(claim: Claim, evidence: list[Evidence]) -> Verdict:
    if not evidence:
        return Verdict("False", 0.58, "No live evidence was found for the claim.", "Search returned no usable sources.")
    claim_values = comparable_numbers(claim.text)
    evidence_text = " ".join(f"{item.title}. {item.snippet}" for item in evidence)
    evidence_values = comparable_numbers(evidence_text)
    overlap = term_overlap(extract_terms(claim.text), evidence_text.lower())
    if claim_values and any(values_match(a, b) for a in claim_values for b in evidence_values) and overlap >= 0.18:
        return Verdict("Verified", 0.72, "Live evidence supports the stated value.", "The claim value appears in topic-matching evidence.")
    alternatives = [value for value in evidence_values if not any(values_match(value, claim_value) for claim_value in claim_values)]
    if claim_values and alternatives and overlap >= 0.14:
        return Verdict(
            "Inaccurate",
            0.66,
            f"Evidence points to {alternatives[0][0]} instead of {claim_values[0][0]}.",
            "Live evidence discusses the topic but shows a conflicting value.",
        )
    return Verdict("False", 0.62, "No source found that supports the exact measurable value.", "Evidence overlap was weak or non-specific.")


def world_population_override(claim: Claim, evidence: list[Evidence]) -> Verdict | None:
    lowered = claim.text.lower()
    if "population" not in lowered or not re.search(r"\b(global|world)\b", lowered):
        return None
    world_bank = next((item for item in evidence if item.provider == "World Bank API"), None)
    if not world_bank:
        return None
    claim_values = comparable_numbers(claim.text)
    evidence_values = comparable_numbers(world_bank.snippet)
    if not claim_values or not evidence_values:
        return None
    if any(values_match(a, b) for a in claim_values for b in evidence_values):
        return Verdict("Verified", 0.9, "World Bank evidence supports the stated population value.", "Structured public data matches the claim.")
    return Verdict(
        "Inaccurate",
        0.9,
        f"Evidence points to {evidence_values[0][0]} instead of {claim_values[0][0]}.",
        "World Bank structured data conflicts with the stated population value.",
    )


def build_queries(text: str) -> list[str]:
    terms = extract_terms(text)
    year = next(iter(re.findall(r"\b(?:19|20)\d{2}\b", text)), "")
    value = next(iter(NUMBER_RE.findall(text)), "")
    return dedupe_strings(
        [
            text[:180],
            f"{' '.join(terms[:9])} {value} {year} source",
            f"{' '.join(terms[:9])} latest official data",
        ]
    )


def comparable_numbers(text: str) -> list[tuple[str, float, str]]:
    values: list[tuple[str, float, str]] = []
    searchable = re.sub(r"(\d+)\s*\.\s+(\d+)", r"\1.\2", text)
    for match in NUMBER_RE.finditer(searchable):
        raw = match.group(0).strip()
        lower = raw.lower()
        numeric = re.sub(r"[$€£,%]", "", raw).replace(",", "").strip()
        numeric_match = re.search(r"\d+(?:\.\d+)?", numeric)
        if not numeric_match:
            continue
        value = float(numeric_match.group(0))
        if "trillion" in lower:
            value *= 1_000_000_000_000
        elif "billion" in lower or "bn" in lower:
            value *= 1_000_000_000
        elif "million" in lower or "mn" in lower:
            value *= 1_000_000
        elif "thousand" in lower:
            value *= 1_000
        unit = "year" if re.fullmatch(r"(?:19|20)\d{2}", lower) else "percent" if "%" in lower or "percent" in lower else "count"
        if unit != "year":
            values.append((raw, value, unit))
    return values


def values_match(a: tuple[str, float, str], b: tuple[str, float, str]) -> bool:
    if a[2] != b[2] and "percent" in (a[2], b[2]):
        return False
    if a[1] == 0 or b[1] == 0:
        return a[1] == b[1]
    return abs(a[1] - b[1]) / max(abs(a[1]), abs(b[1])) <= 0.06


def extract_terms(text: str) -> list[str]:
    cleaned = NUMBER_RE.sub(" ", text.lower())
    terms = re.sub(r"[^a-z0-9\s-]", " ", cleaned).split()
    return dedupe_strings([term.strip("-") for term in terms if len(term) > 2 and term not in STOP_WORDS])[:18]


def term_overlap(terms: Iterable[str], text: str) -> float:
    items = list(terms)
    if not items:
        return 0.0
    hits = sum(1 for term in items if term in text)
    return hits / len(items)


def categorize_claim(sentence: str) -> str:
    lowered = sentence.lower()
    if re.search(r"\b(revenue|profit|sales|valuation|market cap|worth|usd|dollars)\b", lowered):
        return "Financial"
    if re.search(r"\b(accuracy|latency|speed|capacity|technical|model|battery|emissions)\b", lowered):
        return "Technical"
    if re.search(r"\b(percent|users|customers|population|share|growth|downloads|employees)\b", lowered):
        return "Statistics"
    if re.search(r"\b(founded|launched|released|as of|by 20\d{2}|in 20\d{2})\b", lowered):
        return "Date"
    return "General"


def clean_sentence(sentence: str) -> str:
    return sentence.strip().strip("-:;,. ") + "."


def normalize_category(value: str) -> str:
    return value if value in {"Statistics", "Financial", "Date", "Technical", "General"} else "General"


def normalize_status(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith("verified"):
        return "Verified"
    if lowered.startswith("inaccurate"):
        return "Inaccurate"
    return "False"


def default_correction(status: str) -> str:
    if status == "Verified":
        return "The supplied evidence supports the claim."
    if status == "Inaccurate":
        return "The supplied evidence conflicts with the stated value."
    return "No supplied evidence supports the claim."


def parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Llama response did not contain valid JSON.")


def dedupe_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    unique: list[Claim] = []
    for claim in claims:
        key = re.sub(r"[^a-z0-9]+", " ", claim.text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(Claim(id=f"claim-{len(unique) + 1}", text=claim.text, category=claim.category))
    return unique[:MAX_CLAIMS]


def dedupe_evidence(sources: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    unique: list[Evidence] = []
    for source in sources:
        key = source.url or source.title
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def evidence_to_dict(source: Evidence) -> dict[str, str]:
    return {"title": source.title, "url": source.url, "snippet": source.snippet[:360], "provider": source.provider}


def strip_html(value: str) -> str:
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def format_large_number(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} billion"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} million"
    return f"{value:,.0f}"


def llama_config_from_env() -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLAMA_API_KEY", "")
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENROUTER_API_BASE_URL") or os.getenv("LLAMA_API_BASE_URL", LLAMA_DEFAULT_ENDPOINT),
        "model": os.getenv("OPENROUTER_MODEL") or os.getenv("LLAMA_MODEL", LLAMA_DEFAULT_MODEL),
    }
