"""Integer durations and per-frame text events, exactly as the acoustic model was trained.

Punctuation, stress and length marks own no frames, so expanding phone ids by duration drops them. The
flow receives them per frame instead:

- boundary event: a punctuation mark's id on the frames of the ``<sil>`` that follows it (or, when that
  silence received no frames, on the first frame of the next phone); ``word_start_id`` (= phone vocab size)
  on the first phone of every word after the chunk's first word, unless punctuation already marks it;
- modifier bits: stress marks (ˈ ˌ) on the next acoustic phone, length and combining marks on the exact
  preceding phone;
- phase of each frame inside its phone and the phone's log1p duration;
- sentence type: how the sentence that the phone belongs to ends (looked up in the following context when
  the chunk ends before its sentence does).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .g2p_phrases import is_non_acoustic_phone_modifier

PUNCTUATION_PHONES = frozenset({"<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ellipsis>", "<end_stmt>",
                                "<end_question>", "<end_exclaim>"})
SENTENCE_TYPES = {"<end_stmt>": 1, "<end_question>": 2, "<end_exclaim>": 3, "<ellipsis>": 4}
SILENCE = "<sil>"


def is_punctuation_pause_phone(phone: str) -> bool:
    return phone.startswith("<pause") or phone.startswith("<end_") or phone == "<ellipsis>"


def frames_to_durations(values: Sequence[float], phones: Sequence[str], *, scale: float = 1.0) -> list[int]:
    """Rounded frames: punctuation and modifiers 0; ``<sil>`` may be 0; every other phone at least 1."""
    durations = []
    for value, phone in zip(values, phones):
        frames = int(round(max(0.0, float(value)) * float(scale)))
        if phone == SILENCE:
            pass
        elif is_non_acoustic_phone_modifier(phone) or is_punctuation_pause_phone(phone):
            frames = 0
        else:
            frames = max(1, frames)
        durations.append(frames)
    return durations


def modifier_bits(token_to_id: Mapping[str, int]) -> dict[str, int]:
    tokens = [t for t, _ in sorted(token_to_id.items(), key=lambda kv: kv[1]) if is_non_acoustic_phone_modifier(t)]
    return {token: 1 << bit for bit, token in enumerate(tokens)}


def sentence_types(phones: Sequence[str], following: Sequence[str]) -> list[int]:
    """Per phone: the type of the next sentence end at or after it, else that of the following context, else 0."""
    current = next((SENTENCE_TYPES[t] for t in following if t in SENTENCE_TYPES), 0)
    types = [0] * len(phones)
    for index in range(len(phones) - 1, -1, -1):
        current = SENTENCE_TYPES.get(phones[index], current)
        types[index] = current
    return types


def frame_events(phones: Sequence[str], durations: Sequence[int], word_starts: Sequence[int], following: Sequence[str],
                 token_to_id: Mapping[str, int], bits: Mapping[str, int]) -> dict[str, np.ndarray]:
    phones, durations = list(phones), [int(d) for d in durations]
    count, length = len(phones), int(sum(durations))
    boundary = np.zeros(length, np.int64)
    modifier = np.zeros(length, np.int64)
    phase = np.zeros(length, np.float32)
    log_duration = np.zeros(length, np.float32)
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for frames in durations:
        if frames <= 0:
            spans.append(None)
            continue
        spans.append((cursor, cursor + frames))
        phase[cursor:cursor + frames] = (np.arange(frames, dtype=np.float32) + 0.5) / float(frames)
        log_duration[cursor:cursor + frames] = math.log1p(float(frames))
        cursor += frames

    for index, phone in enumerate(phones):
        if phone in PUNCTUATION_PHONES:
            owner = index + 1   # stacked marks share the <sil> after the run; the last one names the event
            while owner < count and phones[owner] in PUNCTUATION_PHONES:
                owner += 1
            if owner >= count or phones[owner] != SILENCE:
                raise ValueError(f"Punctuation {phone!r} lacks its following <sil>")
            event = int(token_to_id[phone])
            if spans[owner] is not None:
                boundary[spans[owner][0]:spans[owner][1]] = event
            else:
                following_span = _next_segment_span(phones, spans, owner + 1)
                if following_span is not None:
                    boundary[following_span[0]] = event
        if is_non_acoustic_phone_modifier(phone):
            stress = phone in {"ˈ", "ˌ"}
            owner_span = _next_acoustic_span(phones, spans, index + 1, 1) if stress else _postfix_owner_span(phones, spans, index)
            if stress and owner_span is None:
                owner_span = _next_acoustic_span(phones, spans, index - 1, -1)
            bit = int(bits.get(phone, 0))
            if owner_span is not None and bit:
                modifier[owner_span[0]:owner_span[1]] |= bit

    word_start_id = len(token_to_id)
    for index in word_starts:
        if not 0 <= index < count:
            continue
        span = spans[index]
        if span is not None:
            if not (boundary[span[0]:span[1]] > 0).any():
                boundary[span[0]:span[1]] = word_start_id
            continue
        following_span = _next_segment_span(phones, spans, index + 1)
        if following_span is not None and boundary[following_span[0]] == 0:
            boundary[following_span[0]] = word_start_id

    sentence = np.repeat(np.asarray(sentence_types(phones, following), np.int64), np.asarray(durations, np.int64))
    return dict(expanded_boundary_event_ids=boundary, expanded_modifier_event_ids=modifier, expanded_phone_phase=phase,
                expanded_phone_log_duration=log_duration, expanded_sentence_type_ids=sentence)


def _next_segment_span(phones, spans, start):
    for index in range(max(0, start), len(spans)):
        if phones[index] == SILENCE or phones[index] in PUNCTUATION_PHONES:
            return None
        if spans[index] is not None:
            return spans[index]
    return None


def _next_acoustic_span(phones, spans, start, step):
    index = start
    while 0 <= index < len(phones):
        if phones[index] == SILENCE or phones[index] in PUNCTUATION_PHONES:
            return None
        if spans[index] is not None and not is_non_acoustic_phone_modifier(phones[index]):
            return spans[index]
        index += step
    return None


def _postfix_owner_span(phones, spans, modifier_index):
    index = modifier_index - 1
    while index >= 0 and is_non_acoustic_phone_modifier(phones[index]):
        index -= 1
    if index < 0 or phones[index] == SILENCE or phones[index] in PUNCTUATION_PHONES:
        return None
    return spans[index]
