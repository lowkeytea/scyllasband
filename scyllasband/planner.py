"""Long-form planning: text records -> sentences, grouped into chains that are spoken continuously.

Consecutive sentences with the same voice, language and delivery form a chain, spoken continuously: the streaming
loop reads a chain's sentences as one stream of phones and cuts it into synthesis targets, each seeing the text around
it as context, continuing from the previous target's acoustics and decoded as part of one continuous waveform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

from .delivery import delivery_spec, resolve_delivery
from .text_normalizer import normalize_spoken_text

# A sentence ends after a terminal punctuation run (and any closing quotes/brackets) followed by whitespace.
_SENTENCE_END_RE = re.compile(r"(?:[.!?…]+|\.\.\.)[\"'”’»)\]]*\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_CLAUSE_RE = re.compile(r"[,;:—-]\s+")


@dataclass
class PlanChunk:
    chunk_id: str
    index: int
    record_index: int
    chain_id: str
    voice: str
    language: str
    delivery: dict[str, Any]
    text: str
    source_text: str
    paragraph_end: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesisPlan:
    chunks: list[PlanChunk] = field(default_factory=list)
    source_kind: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return dict(source_kind=self.source_kind, chunks=[chunk.to_dict() for chunk in self.chunks],
                    chains=len({chunk.chain_id for chunk in self.chunks}))

    def chain(self, chunk: PlanChunk) -> list[PlanChunk]:
        return [c for c in self.chunks if c.chain_id == chunk.chain_id]


def split_sentences(text: str) -> list[str]:
    """Sentences of one paragraph, each keeping its own terminal punctuation."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    out, start = [], 0
    for match in _SENTENCE_END_RE.finditer(value):
        piece = value[start:match.end()].strip()
        if any(char.isalnum() for char in piece):
            out.append(piece)
            start = match.end()
    tail = value[start:].strip()
    if any(char.isalnum() for char in tail):
        out.append(tail)
    elif tail and out:
        out[-1] = f"{out[-1]} {tail}"
    return out


def split_for_retry(text: str, *, min_share: float = 0.0) -> list[str]:
    """Split one overlong sentence in two: at the clause punctuation nearest the middle that leaves each piece at least
    ``min_share`` of the characters, else between the words nearest the middle."""
    value = text.strip()
    middle = len(value) / 2
    least = min_share * len(value)
    cuts = [m.end() for m in _CLAUSE_RE.finditer(value) if 0 < m.end() < len(value) and least <= m.end() <= len(value) - least]
    if not cuts:
        cuts = [m.end() for m in re.finditer(r"\s+", value)]
    if not cuts:
        raise ValueError("Sentence cannot be split further")
    cut = min(cuts, key=lambda position: abs(position - middle))
    return [value[:cut].strip(), value[cut:].strip()]


def plan_records(records: Sequence[Mapping[str, Any]], *, resolve_language, normalize: bool = True,
                 source_kind: str = "text") -> SynthesisPlan:
    """``records``: dicts with text, voice and optional language / delivery. ``resolve_language(voice, language)``
    returns the trained locale for the voice."""
    plan = SynthesisPlan(source_kind=source_kind)
    chain_key: tuple | None = None
    chain_serial = 0
    for record_index, record in enumerate(records):
        voice = str(record.get("voice") or "").strip()
        if not voice:
            raise ValueError(f"Record {record_index + 1} has no voice")
        language = resolve_language(voice, record.get("language"))
        delivery = resolve_delivery(record.get("delivery"))
        key = (voice, language, delivery_spec(delivery))
        if key != chain_key:
            chain_key, chain_serial = key, chain_serial + 1
        paragraphs = [p for p in _PARAGRAPH_RE.split(str(record.get("text") or "")) if p.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs):
            spoken = normalize_spoken_text(paragraph, language=language) if normalize else paragraph
            sentences = split_sentences(spoken)
            for sentence_index, sentence in enumerate(sentences):
                index = len(plan.chunks)
                plan.chunks.append(PlanChunk(
                    chunk_id=f"chunk_{index + 1:04d}", index=index, record_index=record_index, chain_id=f"chain_{chain_serial:03d}",
                    voice=voice, language=language, delivery=delivery, text=sentence, source_text=paragraph.strip(),
                    paragraph_end=sentence_index == len(sentences) - 1 and paragraph_index < len(paragraphs) - 1))
    return plan


def records_from_text(text: str, *, voice: str, language: str | None, delivery: Any = None) -> list[dict[str, Any]]:
    return [dict(text=text, voice=voice, language=language, delivery=delivery)]


# --- group-speak markup: "[voice]", "[voice:language]", "[voice:language:energy=2.3,valence=2.8]", "[language]" ----
GROUP_TAG_RE = re.compile(r"\[([-A-Za-z0-9_.,:=]+)\]")
GROUP_LANGUAGE_TAGS = frozenset({"en", "en_us", "en_gb", "es", "it", "de", "fr", "vi"})


def parse_group_lines(text: str, *, default_voice: str | None, default_language: str | None,
                      default_delivery: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        voice, language, delivery = default_voice, default_language, default_delivery
        position = 0
        for match in GROUP_TAG_RE.finditer(line):
            _append_group_row(rows, line_number, line[position:match.start()], voice, language, delivery)
            voice, language, delivery = _parse_group_tag(match.group(1), voice, language, delivery)
            position = match.end()
        _append_group_row(rows, line_number, line[position:], voice, language, delivery)
    return rows


def _parse_group_tag(label: str, voice, language, delivery):
    clean = label.strip()
    if ":" in clean:
        voice_part, language_part, *rest = clean.split(":", 2)
        voice = voice_part.strip() or voice
        language = language_part.strip() or language
        if rest and rest[0].strip():
            delivery = delivery_spec(rest[0].strip())   # a new delivery tag replaces the active request
    elif clean in GROUP_LANGUAGE_TAGS:
        language = clean
    elif clean:
        voice = clean
    return voice, language, delivery


def _append_group_row(rows, line_number, text, voice, language, delivery) -> None:
    line = text.strip()
    if not line:
        return
    if not voice:
        raise ValueError(f"Group-speak line {line_number} needs a [voice] label or --voice default")
    rows.append(dict(voice=voice, language=language, delivery=delivery, text=line))


def describe(plan: SynthesisPlan) -> Iterable[str]:
    for chunk in plan.chunks:
        yield f"{chunk.chunk_id} {chunk.chain_id} {chunk.voice}/{chunk.language} {chunk.text}"
