"""Bounded passage retrieval for Tool Box Phase 2 study materials."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx


STUDY_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com/study-materials"
DOCUMENT_IDS = ("1", "2", "3", "4", "5")
MAX_RESPONSE_TOKENS = 900
TARGET_RESPONSE_BYTES = 880
MAX_PASSAGE_BYTES = 420

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
    {"align", "aligned", "alignment", "calibrate", "calibrated", "calibration", "recalibrated"},
    {"sensor", "sensors", "grid", "array", "acoustic", "hydrophone", "detector"},
    {"cost", "costs", "price", "priced", "fee", "fees", "charge", "budget"},
    {"date", "day", "when", "year", "month"},
    {"duration", "long", "length", "hours", "days", "weeks", "months"},
    {"leader", "lead", "led", "director", "chair", "chief", "head", "manager", "supervisor"},
    {"begin", "began", "start", "started", "commence", "commenced", "launch", "launched"},
    {"end", "ended", "finish", "finished", "complete", "completed", "conclude", "concluded"},
    {"main", "primary", "principal", "first"},
    {"backup", "secondary", "reserve", "alternate", "alternative"},
    {"restore", "restored", "restart", "restarted", "reactivate", "reactivated", "online"},
)


def retrieve_study_passages(question: str) -> list[str]:
    """Return relevant source passages, never exceeding 900 o200k tokens."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return select_passages(question, _load_documents())


@lru_cache(maxsize=1)
def _load_documents() -> tuple[str, ...]:
    """Fetch all five small documents concurrently and retain them for the process."""

    def fetch(document_id: str) -> str:
        response = httpx.get(
            f"{STUDY_BASE_URL}/{document_id}",
            headers={"Accept": "text/markdown, text/plain;q=0.9"},
            timeout=httpx.Timeout(6.0, connect=3.0),
            follow_redirects=True,
        )
        response.raise_for_status()
        if not response.text.strip():
            raise RuntimeError(f"study material {document_id} was empty")
        return response.text

    with ThreadPoolExecutor(max_workers=len(DOCUMENT_IDS)) as executor:
        return tuple(executor.map(fetch, DOCUMENT_IDS))


def select_passages(question: str, documents: tuple[str, ...] | list[str]) -> list[str]:
    """Rank markdown passages lexically and fit the strongest ones into the limit."""

    passages = [passage for document in documents for passage in _split_markdown(document)]
    if not passages:
        raise ValueError("study materials did not contain any passages")

    query_terms = _expand_query(_terms(question))
    passage_terms = [_terms(passage) for passage in passages]
    document_frequency = {
        term: sum(term in terms for terms in passage_terms)
        for term in query_terms
    }
    average_length = sum(len(terms) for terms in passage_terms) / len(passage_terms)

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
        distinctive = [term for term in _terms(question) if term not in _STOP_WORDS]
        score += 2.5 * sum(term in term_set for term in distinctive)
        ranked.append((score, -index, passage))

    ranked.sort(reverse=True)
    selected: list[str] = []
    used_tokens = 0
    for score, _, passage in ranked:
        if score <= 0 and selected:
            break
        passage = _truncate_to_bytes(passage, MAX_PASSAGE_BYTES)
        token_count = len(passage.encode("utf-8"))
        if used_tokens + token_count > TARGET_RESPONSE_BYTES:
            continue
        selected.append(passage)
        used_tokens += token_count
        if len(selected) >= 8 or used_tokens >= 720:
            break

    if not selected:
        selected = [_truncate_to_bytes(ranked[0][2], TARGET_RESPONSE_BYTES)]
    # An encoded token always represents one or more source bytes. Keeping the
    # UTF-8 content below 900 bytes therefore guarantees fewer than 900 tokens
    # for o200k_base without downloading a tokenizer vocabulary at runtime.
    if sum(len(item.encode("utf-8")) for item in selected) > MAX_RESPONSE_TOKENS:
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


def _truncate_to_bytes(text: str, maximum: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text
    return encoded[:maximum].decode("utf-8", errors="ignore").rstrip()
