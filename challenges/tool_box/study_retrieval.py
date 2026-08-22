"""Bounded passage retrieval for Tool Box Phase 2 study materials."""

from __future__ import annotations

import math
import re
from functools import lru_cache

import httpx


STUDY_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com/study-materials"
DOCUMENT_SOURCES = (
    ("1", "Meridian Trench Research Station"),
    ("2", "Ashgrove Metropolitan Transit Authority"),
    ("3", "Velmara Compound Phase II Trial Record"),
    ("4", "Hollowlight Engine Technical Handbook"),
    ("5", "Thornmere Growers Cooperative Yearbook"),
)
# The source material is ordinary English prose. This conservative byte cap
# produces roughly 400-550 o200k tokens in live evaluations, leaving ample room
# below both the 900-token retrieval limit and the 1,200-token tool limit.
TARGET_RESPONSE_BYTES = 3_000
MAX_RESPONSE_BYTES = 3_300
MAX_PASSAGE_BYTES = 560

_WORD_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "into", "is",
    "it", "last", "of", "on", "or", "our", "the", "their", "there", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "you", "your",
}
_ALIASES = (
    {
        "personnel", "staff", "staffing", "crew", "crewed", "worker", "workers",
        "workforce", "population", "resident", "residents", "occupancy", "people",
        "individual", "individuals", "aboard", "live", "living", "headcount",
    },
    {
        "facility", "station", "habitat", "outpost", "site", "base", "complex",
        "centre", "center",
    },
    {
        "simultaneous", "simultaneously", "concurrent", "concurrently", "together",
        "overlap", "overlapping", "occupancy",
    },
    {"amount", "count", "number", "many", "total", "capacity", "level", "levels"},
    {"align", "aligned", "alignment", "calibrate", "calibrated", "calibration", "recalibrated"},
    {"sensor", "sensors", "grid", "array", "acoustic", "hydrophone", "detector"},
    {"cost", "costs", "price", "priced", "fee", "fees", "charge", "budget"},
    {"date", "day", "when", "year", "month"},
    {"duration", "long", "length", "hours", "days", "weeks", "months", "period"},
    {"often", "frequency", "interval", "schedule", "cycle", "regularly", "every"},
    {"maximum", "max", "limit", "limited", "ceiling", "highest", "most"},
    {"minimum", "min", "least", "lowest", "floor"},
    {"leader", "lead", "led", "director", "chair", "chief", "head", "manager", "supervisor"},
    {"begin", "began", "start", "started", "commence", "commenced", "launch", "launched"},
    {"end", "ended", "finish", "finished", "complete", "completed", "conclude", "concluded"},
    {"main", "primary", "principal", "first"},
    {"backup", "secondary", "reserve", "alternate", "alternative"},
    {"restore", "restored", "restart", "restarted", "reactivate", "reactivated", "online"},
    {"store", "stored", "storage", "kept", "held", "repository", "archive", "vault"},
    {"cause", "caused", "reason", "root", "traced", "failure", "fault", "incident"},
    {"vehicle", "craft", "submersible", "vessel", "fleet"},
    {"participant", "participants", "patient", "patients", "subject", "subjects", "cohort"},
    {"medicine", "medication", "compound", "drug", "dose", "dosing", "dosage"},
)


def retrieve_study_passages(question: str, semantic_context: str) -> list[str]:
    """Return passages using both the original question and an LLM query rewrite."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(semantic_context, str) or not semantic_context.strip():
        raise ValueError("semantic_context must be a non-empty string")
    return select_passages(
        question,
        _load_documents(),
        semantic_context=semantic_context,
    )


@lru_cache(maxsize=1)
def _load_documents() -> tuple[str, ...]:
    """Fetch all documents over one keep-alive connection and cache the corpus."""

    def fetch(client: httpx.Client, source: tuple[str, str]) -> str:
        document_id, title = source
        response = client.get(
            f"{STUDY_BASE_URL}/{document_id}",
        )
        response.raise_for_status()
        if not response.text.strip():
            raise RuntimeError(f"study material {document_id} was empty")
        # The endpoint documents begin at H2, so add the index title as H1.
        # This preserves which organisation or facility each passage belongs to.
        return f"# {title}\n\n{response.text}"

    transport = httpx.HTTPTransport(retries=1)
    with httpx.Client(
        headers={"Accept": "text/markdown, text/plain;q=0.9"},
        timeout=httpx.Timeout(7.0, connect=3.5),
        follow_redirects=True,
        transport=transport,
    ) as client:
        return tuple(fetch(client, source) for source in DOCUMENT_SOURCES)


def select_passages(
    question: str,
    documents: tuple[str, ...] | list[str],
    *,
    semantic_context: str = "",
) -> list[str]:
    """Rank passages with semantic rewrite, lexical, and fuzzy word evidence."""

    passages = [passage for document in documents for passage in _split_markdown(document)]
    if not passages:
        raise ValueError("study materials did not contain any passages")

    original_terms = _terms(question)
    semantic_terms = _terms(semantic_context)
    query_terms = _expand_query(original_terms + semantic_terms)
    passage_terms = [_terms(passage) for passage in passages]
    document_frequency = {
        term: sum(term in terms for terms in passage_terms)
        for term in query_terms
    }
    average_length = sum(len(terms) for terms in passage_terms) / len(passage_terms)

    distinctive = [
        term
        for term in original_terms + semantic_terms
        if term not in _STOP_WORDS
    ]
    ranked: list[tuple[float, int, str]] = []
    for index, (passage, terms) in enumerate(zip(passages, passage_terms, strict=True)):
        score = _bm25_score(
            terms,
            query_terms,
            document_frequency,
            len(passages),
            average_length,
        )
        term_set = set(terms)
        score += 2.5 * sum(term in term_set for term in distinctive)
        score += _fuzzy_morphology_score(distinctive, term_set)
        ranked.append((score, -index, passage))

    ranked.sort(reverse=True)
    selected: list[str] = []
    used_bytes = 0
    section_counts: dict[str, int] = {}
    for score, _, passage in ranked:
        if score <= 0 and selected:
            break
        passage = _truncate_to_bytes(passage, MAX_PASSAGE_BYTES)
        byte_count = len(passage.encode("utf-8"))
        if used_bytes + byte_count > TARGET_RESPONSE_BYTES:
            continue
        section = passage.partition("\n")[0]
        if section_counts.get(section, 0) >= 2:
            continue
        selected.append(passage)
        used_bytes += byte_count
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) >= 6 or used_bytes >= 2_500:
            break

    if not selected:
        selected = [_truncate_to_bytes(ranked[0][2], TARGET_RESPONSE_BYTES)]
    if sum(len(item.encode("utf-8")) for item in selected) > MAX_RESPONSE_BYTES:
        raise RuntimeError("internal retrieval response exceeded the token budget")
    return selected


def _split_markdown(document: str) -> list[str]:
    title = "Study material"
    heading = ""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        body = "\n".join(current).strip()
        current.clear()
        if not body:
            return
        prefix = " — ".join(part for part in (title, heading) if part)
        blocks.extend(_bounded_blocks(prefix, body))

    for line in document.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(line) - len(line.lstrip("#"))
            if level == 1:
                title = match.group(1).strip()
                heading = ""
            else:
                heading = match.group(1).strip()
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def _bounded_blocks(prefix: str, body: str) -> list[str]:
    full = f"{prefix}\n{body}" if prefix else body
    if len(full.encode("utf-8")) <= MAX_PASSAGE_BYTES:
        return [full]

    chunks: list[str] = []
    words = body.split()
    current: list[str] = []
    for word in words:
        candidate_words = current + [word]
        candidate = " ".join(candidate_words)
        rendered = f"{prefix}\n{candidate}" if prefix else candidate
        if current and len(rendered.encode("utf-8")) > MAX_PASSAGE_BYTES:
            body_chunk = " ".join(current)
            chunks.append(f"{prefix}\n{body_chunk}" if prefix else body_chunk)
            # A small overlap preserves facts which cross a window boundary.
            current = current[-8:] + [word]
        else:
            current = candidate_words
    if current:
        body_chunk = " ".join(current)
        chunks.append(f"{prefix}\n{body_chunk}" if prefix else body_chunk)
    return [_truncate_to_bytes(chunk, MAX_PASSAGE_BYTES) for chunk in chunks]


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _expand_query(terms: list[str]) -> set[str]:
    expanded = {term for term in terms if term not in _STOP_WORDS}
    for aliases in _ALIASES:
        if set(terms).intersection(aliases):
            expanded.update(aliases)
    return expanded


def _bm25_score(
    terms: list[str],
    query_terms: set[str],
    document_frequency: dict[str, int],
    passage_count: int,
    average_length: float,
) -> float:
    frequencies = {term: terms.count(term) for term in query_terms}
    score = 0.0
    for term, frequency in frequencies.items():
        if not frequency:
            continue
        frequency_in_documents = document_frequency[term]
        inverse_frequency = math.log(
            1 + (passage_count - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
        )
        denominator = frequency + 1.5 * (0.25 + 0.75 * len(terms) / max(average_length, 1))
        score += inverse_frequency * frequency * 2.5 / denominator
    return score


def _fuzzy_morphology_score(query_terms: list[str], passage_terms: set[str]) -> float:
    """Reward close forms such as scrubbing/scrubber without broad substring hits."""

    score = 0.0
    candidates = [term for term in passage_terms if len(term) >= 5]
    for query_term in set(query_terms):
        if len(query_term) < 5 or query_term in passage_terms:
            continue
        query_grams = _character_grams(query_term)
        best_similarity = 0.0
        for candidate in candidates:
            if candidate[0] != query_term[0]:
                continue
            candidate_grams = _character_grams(candidate)
            similarity = len(query_grams & candidate_grams) / len(
                query_grams | candidate_grams
            )
            best_similarity = max(best_similarity, similarity)
        if best_similarity >= 0.38:
            score += 2.0 * best_similarity
    return score


def _character_grams(word: str) -> set[str]:
    return {word[index : index + 3] for index in range(len(word) - 2)}


def _truncate_to_bytes(text: str, maximum: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text
    return encoded[:maximum].decode("utf-8", errors="ignore").rstrip()
