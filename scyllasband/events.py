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
# Pause-content marks (v5 vocabularies): zero-frame tokens in front of the punctuation run before a pause, saying what the
# pause holds (<nonverbal> = sound in the gap). They reach that pause's frames as a modifier-event bit.
PAUSE_MARKS = frozenset({"<breath>", "<nonverbal>"})
SOUNDED_PAUSE = "<nonverbal>"
# "Uh-uh" / "Nuh-uh" opening a sentence (v5): the vowel, stop, break and vowel of the word, and the least frames of the
# first vowel, the break and the second vowel. The duration model gives the first vowel 1 to 3 frames and no break there,
# which is heard as "ahh" / "nah"; inside and at the end of a sentence its own lengths are right.
OPENING_BREAK = ("ʌ", "̃", "ʔ", SILENCE, "ʌ", "̃")
OPENING_BREAK_LEAD = frozenset({SILENCE, "n", "ˈ"})
OPENING_BREAK_FRAMES = ((0, 6), (3, 2), (4, 12))


def is_punctuation_pause_phone(phone: str) -> bool:
    return phone.startswith("<pause") or phone.startswith("<end_") or phone == "<ellipsis>"


def frames_to_durations(values: Sequence[float], phones: Sequence[str], *, scale: float = 1.0, pause_scale: float = 1.0,
                        min_pauses: Sequence[int] = (), min_pause_frames: float = 0.0) -> list[int]:
    """Rounded frames: punctuation and modifiers 0; ``<sil>`` may be 0; every other phone at least 1. ``pause_scale``
    multiplies ``<sil>`` frames on top of ``scale`` (pace moves pauses with the words). The ``<sil>`` at each index in
    ``min_pauses`` lasts at least ``min_pause_frames`` x ``scale`` frames."""
    durations = []
    floored = set(min_pauses)
    floor = int(round(max(0.0, float(min_pause_frames)) * float(scale)))
    for index, (value, phone) in enumerate(zip(values, phones)):
        factor = float(scale) * (float(pause_scale) if phone == SILENCE else 1.0)
        frames = int(round(max(0.0, float(value)) * factor))
        if phone == SILENCE:
            if index in floored:
                frames = max(frames, floor)
        elif is_non_acoustic_phone_modifier(phone) or is_punctuation_pause_phone(phone) or phone in PAUSE_MARKS:
            frames = 0
        else:
            frames = max(1, frames)
        durations.append(frames)
    return durations


def modifier_bits(token_to_id: Mapping[str, int]) -> dict[str, int]:
    tokens = [t for t, _ in sorted(token_to_id.items(), key=lambda kv: kv[1]) if is_non_acoustic_phone_modifier(t) or t in PAUSE_MARKS]
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
        if phone in PAUSE_MARKS:
            owner = index + 1   # the pause after the marks and punctuation that follow; none inside this target: no frames to mark
            while owner < count and (phones[owner] in PUNCTUATION_PHONES or phones[owner] in PAUSE_MARKS):
                owner += 1
            bit = int(bits.get(phone, 0))
            if bit and owner < count and phones[owner] == SILENCE and spans[owner] is not None:
                modifier[spans[owner][0]:spans[owner][1]] |= bit
            continue
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


def mark_sounded_pauses(phones: Sequence[str], *, last: bool = True, mark: str = SOUNDED_PAUSE) -> tuple[list[str], list[int]]:
    """``phones`` of one sentence (silence to silence) with ``mark`` in front of the punctuation run before every pause that
    follows punctuation, and for each ORIGINAL phone index its index in the result. The leading silence is never marked;
    the trailing one only with ``last`` (a sentence followed by more of the passage: its end pause is inside the text)."""
    phones = list(phones)
    wanted = set()
    for index in range(1, len(phones) if last else len(phones) - 1):
        if phones[index] == SILENCE and phones[index - 1] in PUNCTUATION_PHONES:
            at = index
            while at > 0 and phones[at - 1] in PUNCTUATION_PHONES:
                at -= 1
            wanted.add(at)
    out, moved = [], []
    for index, phone in enumerate(phones):
        if index in wanted:
            out.append(mark)
        moved.append(len(out))
        out.append(phone)
    return out, moved


def opening_minimums(phones: Sequence[str], durations: Sequence[int], sentence_starts: Sequence[int] = (0,)) -> list[int]:
    """``durations`` with the minimum lengths of an opening "Uh-uh" / "Nuh-uh" (``OPENING_BREAK``): where the first word
    after a sentence start (``sentence_starts``: the index of the silence in front of each sentence) is that word, with
    only the silence, an "n" and the stress mark before its first vowel."""
    out = [int(d) for d in durations]
    size = len(OPENING_BREAK)
    for start in sentence_starts:
        for at in range(start, min(start + 4, len(phones) - size + 1)):
            if not all(phone in OPENING_BREAK_LEAD for phone in phones[start:at]):
                break
            if tuple(phones[at:at + size]) == OPENING_BREAK:
                for offset, frames in OPENING_BREAK_FRAMES:
                    out[at + offset] = max(out[at + offset], frames)
                break
    return out


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
