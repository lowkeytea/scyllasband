"""Singing: a melody score to sung audio on a v4 bundle.

A score (JSON) is ``text``, ``bpm``, ``style`` and ``notes``, an ordered list of ``[label, note, beats]``: ``note`` a pitch
name with octave (``C4``, ``F#3``, ``Bb3``) or ``rest`` (label ``-``); ``label`` a word on one note, or ``word~n`` for the n-th
of consecutive parts of one word (syllables, or a melisma holding one syllable over several pitches). ``style`` picks the
delivery mode: ``pop`` (or ``sung``) and ``ballad``.

The song is rendered by phrases, the way the model was trained on clips of a few seconds:

1. :func:`parse_score` validates the score, turns notes into MIDI and beats into seconds, and groups labels into words.
2. :func:`register_shift` moves the whole song by octaves so its mean pitch sits in the voice's sung range.
3. :func:`split_phrases` cuts the song at rests of ``PHRASE_REST`` seconds or more and packs the runs between them up
   to the longest phrase the bundle can render (at most ``MAX_PHRASE_SECONDS``); a run longer than that is cut at its
   longest internal rest, else at the word boundary nearest its middle. Each phrase gets ``pad`` seconds of lead-in
   and release.
4. :func:`plan_song` gives each phrase its phones (the bundle's G2P, one group per score word), integer frames per
   phone from the words' time spans (:func:`phone_durations`: consonants share ``1 - VOWEL_SHARE`` of a word's frames,
   vowels the rest, rests and the pads become ``<sil>`` frames, so the phones sum to the phrase's frames), and the
   melody input in the layout the flow was trained on (:func:`frame_melody`: per frame ``(midi - 60) / 24``, the phase
   0..1 inside the note and 1 where a note sounds; zeros on rests).
5. :func:`sing` renders each phrase with the engine (the score's durations and melody, the delivery mode id, the
   manifest's singing sampler and steps, no delivery request, no acoustic prefix), decodes it with the singing decoder
   and places it on the song's timeline, so the rests between phrases are silence.

Singing needs a v4 bundle (``controls.delivery_modes.enabled``); see :mod:`scyllasband.contract`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np

from .engine import Engine, OverlongError, Sentence
from .events import SILENCE, is_punctuation_pause_phone
from .g2p_phrases import is_non_acoustic_phone_modifier

HOP, SAMPLE_RATE = 512, 24000
PHRASE_REST = 0.25          # seconds of rest that end a phrase
MAX_PHRASE_SECONDS = 10.0   # a phrase is never packed beyond this (nor beyond what the bundle's largest bucket holds)
PAD = 0.12                  # seconds of lead-in and release around each phrase
MIDI_CENTER, MIDI_SCALE = 60, 24
# The band, in semitones around the voice's sung centre, that the song's mean pitch is moved into by octaves. Pitch
# accuracy held within +-5 semitones of the centre and collapsed at +-8 in the register test; +7.5 sang flat.
REGISTER_BELOW, REGISTER_ABOVE = 3.0, 5.0
# Vowel symbols of the phone vocabulary (the first character of a phone decides): the same set the trainer uses.
VOWELS = frozenset("aeiouyɑɐɒæəɚɛɜɝɪɔʊʌɨʉøœɯɤɵɘɞɶ")
# Median share of a word's acoustic frames its vowels take, measured by forced alignment on the training corpus: sung
# 0.51, rap 0.42 (rap at a higher share stretched vowels and squeezed consonants).
VOWEL_SHARE = {"sung": 0.51, "ballad": 0.51, "rap": 0.42, "spoken": 0.51}
MAX_CONSONANT_FRAMES = 8
STYLE_MODES = {"pop": "sung", "sung": "sung", "ballad": "ballad", "rap": "rap"}
_NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_REST_NOTES = frozenset({"rest", "-", "r", ""})


class ScoreError(ValueError):
    """The score does not follow the format."""


@dataclass(frozen=True)
class Note:
    label: str
    midi: int | None      # None for a rest
    start: float          # seconds
    end: float

    @property
    def name(self) -> str:
        return "rest" if self.midi is None else midi_to_note(self.midi)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    parts: int = 1


@dataclass(frozen=True)
class Score:
    text: str
    bpm: float
    style: str
    mode: str                 # delivery mode: sung | ballad | rap
    notes: tuple[Note, ...]
    words: tuple[Word, ...]

    @property
    def seconds(self) -> float:
        return self.notes[-1].end if self.notes else 0.0

    @property
    def pitched(self) -> list[int]:
        return [n.midi for n in self.notes if n.midi is not None]

    def transposed(self, semitones: int) -> "Score":
        if not semitones:
            return self
        notes = tuple(Note(n.label, None if n.midi is None else n.midi + int(semitones), n.start, n.end) for n in self.notes)
        return Score(self.text, self.bpm, self.style, self.mode, notes, self.words)

    def to_dict(self) -> dict[str, Any]:
        return dict(text=self.text, bpm=self.bpm, style=self.style, mode=self.mode,
                    notes=[dict(label=n.label, note=n.name, start=round(n.start, 4), end=round(n.end, 4)) for n in self.notes],
                    words=[dict(word=w.text, start=round(w.start, 4), end=round(w.end, 4)) for w in self.words])


def note_to_midi(name: str) -> int | None:
    """``D#4`` -> 63, ``Bb3`` -> 58; None for a rest (``rest``, ``-``). Anything else raises ScoreError."""
    value = str(name or "").strip()
    if value.lower() in _REST_NOTES:
        return None
    if value[0].upper() not in _NOTE_INDEX:
        raise ScoreError(f"Unknown note {name!r}: use a pitch name with octave (C4, F#3, Bb3) or rest")
    semis, rest = _NOTE_INDEX[value[0].upper()], value[1:]
    while rest and rest[0] in "#b♯♭":
        semis += 1 if rest[0] in "#♯" else -1
        rest = rest[1:]
    try:
        octave = int(rest)
    except ValueError:
        raise ScoreError(f"Unknown note {name!r}: use a pitch name with octave (C4, F#3, Bb3) or rest") from None
    return 12 * (octave + 1) + semis


def midi_to_note(midi: int) -> str:
    return f"{_NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def parse_score(data: Mapping[str, Any]) -> Score:
    """Validate a score and give it absolute times and words.

    Words: a bare label is a word on one note (``row row row`` are three words); ``word~1 word~2 ...`` are consecutive
    parts of one word, written either as the whole word on every part (``baby~1 baby~2``, ``home~1 home~2``) or as its
    syllables (``ba~1 by~2``, joined into ``baby``). A rest ends a word. ``text``, when given, must be the words in order.
    """
    if not isinstance(data, Mapping):
        raise ScoreError("A score is a JSON object with text, bpm, style and notes")
    try:
        bpm = float(data["bpm"])
    except (KeyError, TypeError, ValueError):
        raise ScoreError("Score needs a numeric bpm") from None
    if not math.isfinite(bpm) or bpm <= 0:
        raise ScoreError("Score bpm must be positive")
    style = str(data.get("style") or "pop").strip().lower()
    if style not in STYLE_MODES:
        raise ScoreError(f"Unknown style {style!r}; choose one of {', '.join(sorted(STYLE_MODES))}")
    rows = data.get("notes")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ScoreError("Score notes must be a non-empty list of [label, note, beats]")
    beat = 60.0 / bpm
    notes: list[Note] = []
    words: list[Word] = []
    pieces: list[str] = []            # syllable spellings of the open word
    open_part = 0                     # the last part number of the open word (0: no open word)
    cursor = 0.0
    for index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise ScoreError(f"Note {index + 1} must be [label, note, beats]; got {row!r}")
        label, name, beats = str(row[0]).strip(), str(row[1]), row[2]
        try:
            beats = float(beats)
        except (TypeError, ValueError):
            raise ScoreError(f"Note {index + 1} ({label!r}) has a non-numeric length {row[2]!r}") from None
        if not math.isfinite(beats) or beats <= 0:
            raise ScoreError(f"Note {index + 1} ({label!r}) must last a positive number of beats")
        midi = note_to_midi(name)
        start, end = cursor, cursor + beats * beat
        cursor = end
        rest = midi is None or label in ("-", "")
        notes.append(Note(label if not rest else "-", None if rest else midi, start, end))
        if rest:
            open_part, pieces = 0, []
            continue
        base, _, part = label.partition("~")
        if not base:
            raise ScoreError(f"Note {index + 1} has an empty word label")
        if part and not part.isdigit():
            raise ScoreError(f"Note {index + 1}: {label!r} should be word~n with a part number")
        number = int(part) if part else 1
        if number > 1:
            if number != open_part + 1:
                raise ScoreError(f"Note {index + 1}: {label!r} does not follow part {number - 1} of its word (numbered parts "
                                 "must be consecutive, with no rest between them)")
            pieces.append(base)
            previous = words[-1]
            text = previous.text if base == pieces[0] else "".join(pieces)
            words[-1] = Word(text, previous.start, end, previous.parts + 1)
            open_part = number
            continue
        words.append(Word(base, start, end))
        pieces, open_part = [base], 1
    if not words:
        raise ScoreError("Score has no sung notes")
    text = str(data.get("text") or "").strip() or " ".join(w.text for w in words)
    expected, given = [_norm_word(w.text) for w in words], [_norm_word(t) for t in text.split()]
    if [t for t in given if t] != [t for t in expected if t]:
        raise ScoreError(f"Score text does not match the note labels: text has {len(given)} words ({' '.join(text.split()[:8])} ...), "
                         f"the notes spell {len(words)} ({' '.join(w.text for w in words[:8])} ...)")
    return Score(text=text, bpm=bpm, style=style, mode=STYLE_MODES[style], notes=tuple(notes), words=tuple(words))


def _norm_word(word: str) -> str:
    return "".join(ch for ch in str(word).lower() if ch.isalnum() or ch == "'")


def singing_modes(engine: Engine) -> list[str]:
    """The delivery modes the bundle sings in (``controls.delivery_modes.singing.modes``; every non-spoken mode with an id
    when the manifest lists none)."""
    listed = [str(m) for m in (engine.singing.get("modes") or ()) if str(m) in engine.delivery_modes]
    return listed or sorted(m for m in engine.delivery_modes if m != "spoken")


def singing_styles(engine: Engine) -> list[str]:
    """The score styles the bundle accepts, in the order of ``STYLE_MODES`` (pop before sung before ballad)."""
    modes = set(singing_modes(engine))
    return [style for style, mode in STYLE_MODES.items() if mode in modes]


# --- register ------------------------------------------------------------------------------------------------------------
def voice_sung_center(engine: Engine, voice: str) -> float | None:
    """The MIDI pitch the voice sings around. ``controls.delivery_modes.singing.voice_midi_center[voice]`` when the bundle
    gives it; else the middle of the voice's spoken pitch edges (``controls.pitch.edges``, semitones on the melody scale
    whose ``melody_center`` is C4), which sat within 3 semitones of every voice's measured sung median; None when the
    bundle has neither."""
    centers = engine.singing.get("voice_midi_center") or {}
    if voice in centers:
        return float(centers[voice])
    edges = (engine.pitch.get("edges") or {}).get(voice)
    if not edges:
        return None
    low, high = (float(e) for e in edges)
    melody_center = float(engine.pitch.get("melody_center", 16.65))
    midi_center = float(engine.singing.get("melody", {}).get("midi_center", MIDI_CENTER))
    return (low + high) / 2 - melody_center + midi_center


def register_shift(score: Score, center: float | None) -> int:
    """Octaves (as semitones: -24, -12, 0, 12, 24) that move the score's mean pitch into [center - REGISTER_BELOW,
    center + REGISTER_ABOVE], or as close as an octave step allows (ties go up); 0 without a centre."""
    pitched = score.pitched
    if center is None or not pitched:
        return 0
    mean = sum(pitched) / len(pitched)
    low, high = center - REGISTER_BELOW, center + REGISTER_ABOVE

    def distance(shift: int) -> float:
        return max(0.0, low - (mean + shift), (mean + shift) - high)
    return min((-24, -12, 0, 12, 24), key=lambda shift: (round(distance(shift), 3), -shift))


# --- phrases ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Phrase:
    start: float                 # seconds on the song's timeline where the phrase's audio begins (lead-in included)
    end: float
    notes: tuple[Note, ...]      # shifted to the phrase's own time (lead-in at 0), plus a closing rest
    words: tuple[Word, ...]
    text: str


def split_phrases(score: Score, *, phrase_rest: float = PHRASE_REST, max_seconds: float = MAX_PHRASE_SECONDS,
                  pad: float = PAD) -> list[Phrase]:
    """Phrases = runs of notes between rests of at least ``phrase_rest`` seconds, packed greedily up to ``max_seconds``
    (lead-in and release excluded); a single run longer than that is cut at its longest internal rest, else at the word
    boundary nearest its middle."""
    sung = [n for n in score.notes if n.midi is not None]
    runs: list[list[Note]] = []
    current: list[Note] = []
    for note in sung:
        if current and note.start - current[-1].end >= phrase_rest:
            runs.append(current)
            current = []
        current.append(note)
    if current:
        runs.append(current)
    word_starts = {w.start for w in score.words}

    def cut(run: list[Note]) -> list[list[Note]]:
        if run[-1].end - run[0].start <= max_seconds or len(run) < 2:
            return [run]
        gap, index = max((run[i + 1].start - run[i].end, i) for i in range(len(run) - 1))
        if gap <= 0.02:   # no rest inside: the word boundary nearest the middle
            middle = (run[0].start + run[-1].end) / 2
            candidates = [k for k in range(1, len(run)) if run[k].start in word_starts] or [len(run) // 2]
            index = min(candidates, key=lambda k: abs(run[k].start - middle)) - 1
        return cut(run[:index + 1]) + cut(run[index + 1:])
    runs = [piece for run in runs for piece in cut(run)]

    packed: list[list[Note]] = []
    current = []
    for run in runs:
        if current and run[-1].end - current[0].start > max_seconds:
            packed.append(current)
            current = []
        current += run
    if current:
        packed.append(current)

    phrases = []
    for group in packed:
        t0, t1 = group[0].start, group[-1].end
        words = tuple(Word(w.text, w.start - t0 + pad, w.end - t0 + pad, w.parts) for w in score.words
                      if w.start >= t0 - 1e-6 and w.end <= t1 + 1e-6)
        notes = [Note(n.label, n.midi, n.start - t0 + pad, n.end - t0 + pad) for n in group]
        notes.append(Note("-", None, t1 - t0 + pad, t1 - t0 + 2 * pad))
        phrases.append(Phrase(start=t0 - pad, end=t1 + pad, notes=tuple(notes), words=words, text=" ".join(w.text for w in words)))
    return phrases


# --- phones, durations and melody ------------------------------------------------------------------------------------------
def word_phones(words: Sequence[str], phonemize: Callable[..., Mapping[str, Any]], language: str) -> tuple[list[str], list[int]]:
    """Phones in the training layout for the score's words (``<sil>``, word, ``<sil>``, word, ..., ``<sil>``) and the index
    of each later word's first phone. The phrase is phonemized as one text so the G2P sees its context; when the G2P's
    words do not line up one-to-one with the score's, each word is phonemized on its own, so the count always matches."""
    text = " ".join(words)
    result = phonemize(text, language=language)
    phones, starts = _strip_punctuation(list(result["phones"]), list(result["word_starts"]))
    if len(starts) + 1 == len(words):
        return phones, starts
    phones, starts = [SILENCE], []
    for word in words:
        single = phonemize(word, language=language)
        inner = [p for p in single["phones"] if p != SILENCE and not is_punctuation_pause_phone(p)]
        if not inner:
            raise ScoreError(f"No phones for the word {word!r}")
        if len(phones) > 1:
            phones.append(SILENCE)
            starts.append(len(phones))
        phones.extend(inner)
    phones.append(SILENCE)
    return phones, starts


def _strip_punctuation(phones: list[str], word_starts: list[int]) -> tuple[list[str], list[int]]:
    """Drop punctuation phones (a score has no punctuation; the G2P may still produce some) and their silence when it
    duplicates a neighbouring one, keeping the word starts pointing at the same phones."""
    kept, starts, drop = [], [], set(word_starts)
    for index, phone in enumerate(phones):
        if is_punctuation_pause_phone(phone) or (phone == SILENCE and kept and kept[-1] == SILENCE):
            continue
        if index in drop:
            starts.append(len(kept))
        kept.append(phone)
    return kept, starts


def is_vowel(phone: str) -> bool:
    return bool(phone) and not phone.startswith("<") and phone[0] in VOWELS


def phone_durations(phones: Sequence[str], word_starts: Sequence[int], words: Sequence[Word], total_frames: int,
                    vowel_share: float, *, hop: int = HOP, sample_rate: int = SAMPLE_RATE) -> list[int]:
    """Integer latent frames per phone from the words' time spans: modifiers 0; the leading ``<sil>`` the frames before the
    first word; inside a word the consonants share ``1 - vowel_share`` of its frames equally (1 to 8 frames each) and the
    vowels the rest (a word without vowels gives its last consonant the rest); the ``<sil>`` after a word the frames up to
    the next word, or to ``total_frames`` after the last. The result sums to ``total_frames`` unless a word's phones
    outnumber its frames."""
    phones = list(phones)
    first = next((i for i, p in enumerate(phones) if p != SILENCE), len(phones))
    starts = [first, *word_starts, len(phones)]
    if len(starts) - 1 != len(words):
        raise ScoreError(f"{len(starts) - 1} phonemized words for {len(words)} score words: {[w.text for w in words]}")
    bounds = [(int(round(w.start * sample_rate / hop)), int(round(w.end * sample_rate / hop))) for w in words]
    durations = [0] * len(phones)
    for i in range(first):
        if phones[i] == SILENCE:
            durations[i] = bounds[0][0]
    for gi, (g0, g1) in enumerate(zip(starts[:-1], starts[1:])):
        f0, f1 = bounds[gi]
        acoustic = [i for i in range(g0, g1) if phones[i] != SILENCE and not is_punctuation_pause_phone(phones[i])
                    and not is_non_acoustic_phone_modifier(phones[i])]
        vowels = [i for i in acoustic if is_vowel(phones[i])]
        consonants = [i for i in acoustic if i not in vowels]
        n = max(f1 - f0, len(acoustic))
        if vowels:
            c = min(MAX_CONSONANT_FRAMES, max(1, int(round(n * (1 - vowel_share) / max(1, len(consonants))))))
        else:
            c = max(1, n // max(1, len(consonants)))
        for i in consonants:
            durations[i] = c
        left = n - c * len(consonants)
        if vowels:
            share, extra = divmod(left, len(vowels))
            for k, i in enumerate(vowels):
                durations[i] = max(1, share + (1 if k < extra else 0))
        elif consonants:
            durations[consonants[-1]] += left
        following = bounds[gi + 1][0] if gi + 1 < len(bounds) else max(f1 + 2, total_frames)
        silences = [i for i in range(g0, g1) if phones[i] == SILENCE]
        if silences:
            durations[silences[-1]] = max(0, following - (f0 + n))
    return durations


def frame_melody(notes: Sequence[Note], frames: int, *, hop: int = HOP, sample_rate: int = SAMPLE_RATE,
                 midi_center: float = MIDI_CENTER, midi_scale: float = MIDI_SCALE) -> np.ndarray:
    """[frames, 3] float32: per frame whose centre falls inside a note, ``(midi - midi_center) / midi_scale``, the phase 0..1
    of the frame inside the note and 1; zeros where no note sounds."""
    out = np.zeros((int(frames), 3), np.float32)
    centres = (np.arange(int(frames), dtype=np.float64) * hop + hop // 2) / sample_rate
    for note in notes:
        if note.midi is None or note.end <= note.start:
            continue
        inside = (centres >= note.start) & (centres < note.end)
        if not inside.any():
            continue
        out[inside, 0] = (note.midi - midi_center) / midi_scale
        out[inside, 1] = ((centres[inside] - note.start) / (note.end - note.start)).astype(np.float32)
        out[inside, 2] = 1.0
    return out


# --- the plan ----------------------------------------------------------------------------------------------------------------
@dataclass
class PhrasePlan:
    index: int
    phrase: Phrase
    sentence: Sentence
    durations: list[int]
    frames: int
    melody: np.ndarray           # [frames, 3]

    def to_dict(self) -> dict[str, Any]:
        return dict(index=self.index, start=round(self.phrase.start, 4), end=round(self.phrase.end, 4), text=self.phrase.text,
                    frames=self.frames, phones=list(self.sentence.phones), durations=list(self.durations),
                    notes=[dict(label=n.label, note=n.name, start=round(n.start, 4), end=round(n.end, 4)) for n in self.phrase.notes])


@dataclass
class SongPlan:
    score: Score                 # after the register shift
    voice: str
    language: str
    mode: str
    mode_id: int
    register_shift: int
    phrases: list[PhrasePlan] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return max((p.phrase.end for p in self.phrases), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return dict(voice=self.voice, language=self.language, mode=self.mode, mode_id=self.mode_id, register_shift=self.register_shift,
                    bpm=self.score.bpm, style=self.score.style, text=self.score.text, seconds=round(self.seconds, 3),
                    phrases=[p.to_dict() for p in self.phrases])


def plan_song(score: Score | Mapping[str, Any], engine: Engine, voice: str, language: str, *, register: bool = True,
              phrase_rest: float = PHRASE_REST, max_seconds: float | None = None, pad: float = PAD,
              vowel_share: float | None = None) -> SongPlan:
    """Phrases ready for the engine. ``engine`` supplies the bundle's G2P (with its lexicon), phone vocabulary, manifest
    and bucket limit. ``register``: shift the song by octaves into the voice's sung range. ``max_seconds``: the longest
    phrase; by default what the bundle's largest bucket holds, at most ``MAX_PHRASE_SECONDS``."""
    if not isinstance(score, Score):
        score = parse_score(score)
    if not engine.delivery_modes:
        raise ValueError("Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none")
    allowed = singing_styles(engine)
    if score.style not in allowed:
        raise ScoreError(f"Style {score.style!r} ({score.mode}) is not a singing style of this bundle; choose one of {', '.join(allowed)}")
    sample_rate = int(engine.manifest.get("audio", {}).get("sample_rate", SAMPLE_RATE))
    hop = int(engine.manifest.get("audio", {}).get("latent_hop_length", HOP))
    melody_spec = engine.singing.get("melody") or {}
    midi_center, midi_scale = float(melody_spec.get("midi_center", MIDI_CENTER)), float(melody_spec.get("midi_scale", MIDI_SCALE))
    shift = register_shift(score, voice_sung_center(engine, voice)) if register else 0
    score = score.transposed(shift)
    longest = (engine.max_frames - 2) * hop / sample_rate - 2 * pad
    max_seconds = min(MAX_PHRASE_SECONDS if max_seconds is None else float(max_seconds), longest)
    share = VOWEL_SHARE.get(score.mode, VOWEL_SHARE["sung"]) if vowel_share is None else float(vowel_share)
    plan = SongPlan(score=score, voice=voice, language=language, mode=score.mode, mode_id=engine.delivery_modes[score.mode],
                    register_shift=shift)
    for index, phrase in enumerate(split_phrases(score, phrase_rest=phrase_rest, max_seconds=max_seconds, pad=pad)):
        phones, starts = word_phones([w.text for w in phrase.words], engine.phonemize, language)
        total = int(round(phrase.notes[-1].end * sample_rate / hop)) + 2
        durations = phone_durations(phones, starts, phrase.words, total, share, hop=hop, sample_rate=sample_rate)
        frames = int(sum(durations))
        if frames > engine.max_frames:
            raise OverlongError(f"Phrase {index + 1} ({phrase.text[:40]!r}) needs {frames} latent frames, but this bundle holds at most "
                                f"{engine.max_frames}; add a rest inside it or shorten its notes", frames=frames, limit=engine.max_frames)
        melody = frame_melody(phrase.notes, frames, hop=hop, sample_rate=sample_rate, midi_center=midi_center, midi_scale=midi_scale)
        plan.phrases.append(PhrasePlan(index=index, phrase=phrase, sentence=Sentence(phones=phones, word_starts=starts),
                                       durations=durations, frames=frames, melody=melody))
    return plan


# --- rendering -----------------------------------------------------------------------------------------------------------------
def sing(engine: Engine, score: Score | Mapping[str, Any], voice: str, language: str, *, seed: int | None = None,
         steps: int | None = None, sampler: str | None = None, decoder: str | None = None, guidance: float | None = None,
         register: bool = True, progress: TextIO | None = None, **planning: Any) -> "SynthesisResult":
    """Render a score as one waveform. ``steps``, ``sampler`` and ``decoder`` default to the bundle's singing settings
    (``controls.delivery_modes.singing``); a preferred decoder the bundle does not carry falls back to vocos. ``seed``:
    phrase ``i`` draws from ``seed + i``. Extra keyword arguments go to :func:`plan_song`."""
    from .runtime import SynthesisResult
    plan = plan_song(score, engine, voice, language, register=register, **planning)
    singing = engine.singing
    steps = int(steps or singing.get("steps") or 8)
    sampler = str(sampler or singing.get("sampler") or "heun").lower()
    if sampler not in ("heun", "euler"):
        raise ValueError(f"Unsupported sampler {sampler!r}; choose heun or euler")
    preferred = str(singing.get("decoder") or "vocos")
    if decoder is None:
        decoder = preferred if engine.decoder_graphs.get(preferred) else "vocos"
    if not engine.decoder_graphs.get(decoder):
        raise ValueError(f"This bundle has no {decoder!r} decoder; it has {', '.join(k for k, v in engine.decoder_graphs.items() if v)}")
    guidance = float(singing.get("guidance", 1.0) if guidance is None else guidance)
    sample_rate = int(engine.manifest.get("audio", {}).get("sample_rate", SAMPLE_RATE))
    # No delivery request: sung clips were trained without intensity or pace, and the score sets the timing.
    delivery = (np.zeros((1, 5), np.float32), np.zeros((1, 5), bool))
    song = np.zeros(int(math.ceil(plan.seconds * sample_rate)) + sample_rate // 10, np.float32)
    rendered = []
    for item in plan.phrases:
        phrase = item.phrase
        if progress is not None:
            print(f"[scyllasband] phrase {item.index + 1}/{len(plan.phrases)} {voice}/{language} {plan.mode}: {phrase.text[:80]}",
                  file=progress, flush=True)
        rng = np.random.default_rng(None if seed is None else int(seed) + item.index)
        result = engine.synthesize_sentence(item.sentence, voice=voice, language=language, delivery=delivery, steps=steps, sampler=sampler,
                                            rng=rng, durations=item.durations, melody=item.melody, delivery_mode=plan.mode_id,
                                            guidance=guidance)
        audio = engine.decode(result.latents, voice=voice, language=language, decoder=decoder)
        offset = int(round(phrase.start * sample_rate))
        if offset < 0:
            audio, offset = audio[-offset:], 0
        count = min(audio.size, song.size - offset)
        song[offset:offset + count] += audio[:count]
        rendered.append(dict(item.to_dict(), bucket=result.bucket, seconds=round(audio.size / sample_rate, 3),
                             seed=None if seed is None else int(seed) + item.index))
    song = np.clip(song[:int(math.ceil(plan.seconds * sample_rate))], -1.0, 1.0)
    metadata = dict(plan.to_dict(), phrases=rendered, steps=steps, sampler=sampler, decoder=decoder, guidance=guidance, seed=seed,
                    graph_contract=engine.graph_contract)
    return SynthesisResult(audio=song, sample_rate=sample_rate, metadata=metadata)
