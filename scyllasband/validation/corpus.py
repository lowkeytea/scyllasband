"""Frozen multilingual validation-corpus loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "testing"
    / "long_form"
    / "suite.json"
)
DOCUMENT_KINDS = {"narrative", "dialogue_punctuation", "challenge"}
OVERLAP_STATES = {"training_overlap", "sequence_held_out", "fully_held_out"}


class ValidationCorpusError(ValueError):
    """Raised when a public validation fixture violates its frozen contract."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationCorpusError(f"Validation artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationCorpusError(f"Validation JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationCorpusError(f"Validation JSON must contain an object: {path}")
    return payload


def _resolve_inside(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationCorpusError(f"{label} escapes the suite directory: {relative}") from exc
    return candidate


def _normalized_text(path: Path) -> str:
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValidationCorpusError(f"Validation text is empty: {path}")
    return text


def _document_record(suite_root: Path, reference: object) -> dict[str, Any]:
    if isinstance(reference, str):
        metadata_path = _resolve_inside(suite_root, reference, label="document metadata")
    elif isinstance(reference, Mapping):
        metadata_value = str(reference.get("metadata") or "").strip()
        if not metadata_value:
            raise ValidationCorpusError("Document entry is missing metadata")
        metadata_path = _resolve_inside(suite_root, metadata_value, label="document metadata")
    else:
        raise ValidationCorpusError("Document entries must be metadata paths or objects")

    metadata = _read_json(metadata_path)
    if int(metadata.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValidationCorpusError(f"Unsupported document schema: {metadata_path}")
    document_id = str(metadata.get("id") or "").strip()
    if not document_id:
        raise ValidationCorpusError(f"Document id is missing: {metadata_path}")
    kind = str(metadata.get("kind") or "").strip()
    if kind not in DOCUMENT_KINDS:
        raise ValidationCorpusError(f"Unsupported document kind {kind!r}: {metadata_path}")
    languages = tuple(str(item).strip() for item in metadata.get("languages", ()) if str(item).strip())
    if not languages:
        language = str(metadata.get("language") or "").strip()
        languages = (language,) if language else ()
    if not languages:
        raise ValidationCorpusError(f"Document languages are missing: {metadata_path}")

    input_relative = str(metadata.get("input_path") or "").strip()
    if not input_relative:
        raise ValidationCorpusError(f"Document input_path is missing: {metadata_path}")
    input_path = _resolve_inside(suite_root, input_relative, label="document input")
    text = _normalized_text(input_path)
    actual_sha = sha256_bytes(text.encode("utf-8"))
    declared_sha = str(metadata.get("input_sha256") or "").strip()
    if declared_sha and declared_sha != actual_sha:
        raise ValidationCorpusError(
            f"Document hash mismatch for {document_id}: expected {declared_sha}, got {actual_sha}"
        )

    overlap = str(metadata.get("training_overlap") or "").strip()
    if overlap not in OVERLAP_STATES:
        raise ValidationCorpusError(
            f"Document {document_id} has invalid training_overlap {overlap!r}"
        )
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or not str(provenance.get("redistribution") or "").strip():
        raise ValidationCorpusError(f"Document {document_id} lacks redistribution provenance")

    replacements: list[dict[str, str]] = []
    for item in metadata.get("spoken_replacements", ()):
        if not isinstance(item, Mapping):
            raise ValidationCorpusError(f"Document {document_id} has an invalid spoken replacement")
        surface = str(item.get("surface") or "")
        spoken = str(item.get("spoken") or "")
        if not surface or not spoken or surface not in text:
            raise ValidationCorpusError(
                f"Document {document_id} spoken replacement is missing from its text: {surface!r}"
            )
        replacements.append({"surface": surface, "spoken": spoken})

    challenge_spans: list[dict[str, Any]] = []
    for item in metadata.get("challenge_spans", ()):
        if not isinstance(item, Mapping):
            raise ValidationCorpusError(f"Document {document_id} has an invalid challenge span")
        surface = str(item.get("surface") or "")
        if not surface or surface not in text:
            raise ValidationCorpusError(
                f"Document {document_id} challenge surface is missing: {surface!r}"
            )
        tags = [str(tag) for tag in item.get("tags", ()) if str(tag)]
        if not tags:
            raise ValidationCorpusError(f"Document {document_id} challenge span has no tags")
        challenge_spans.append(
            {
                "surface": surface,
                "tags": tags,
                "expected_spoken": str(item.get("expected_spoken") or surface),
                "expected_ipa": str(item.get("expected_ipa") or "") or None,
            }
        )

    spoken_reference = text
    for replacement in replacements:
        spoken_reference = spoken_reference.replace(replacement["surface"], replacement["spoken"])
    explicit_reference = metadata.get("asr_reference_text")
    if explicit_reference is not None:
        spoken_reference = unicodedata.normalize("NFC", str(explicit_reference)).strip()
    if not spoken_reference:
        raise ValidationCorpusError(f"Document {document_id} has an empty ASR reference")

    public_metadata = dict(metadata)
    public_metadata.update(
        {
            "metadata_path": metadata_path.relative_to(suite_root).as_posix(),
            "input_path": input_path.relative_to(suite_root).as_posix(),
            "input_sha256": actual_sha,
            "languages": list(languages),
            "text": text,
            "asr_reference_text": spoken_reference,
            "spoken_replacements": replacements,
            "challenge_spans": challenge_spans,
        }
    )
    return public_metadata


def load_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    suite_path = Path(path).expanduser().resolve()
    suite = _read_json(suite_path)
    if int(suite.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValidationCorpusError(f"Unsupported suite schema: {suite_path}")
    suite_id = str(suite.get("id") or "").strip()
    if not suite_id:
        raise ValidationCorpusError(f"Suite id is missing: {suite_path}")
    suite_root = suite_path.parent
    documents = [_document_record(suite_root, item) for item in suite.get("documents", ())]
    if not documents:
        raise ValidationCorpusError(f"Suite has no documents: {suite_path}")
    document_ids = [str(item["id"]) for item in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValidationCorpusError("Suite document ids must be unique")

    conditions = suite.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValidationCorpusError("Suite must declare conditions")
    condition_ids: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValidationCorpusError("Suite conditions must be objects")
        condition_id = str(condition.get("id") or "").strip()
        if not condition_id or condition_id in condition_ids:
            raise ValidationCorpusError(f"Invalid or duplicate condition id: {condition_id!r}")
        condition_ids.add(condition_id)
        ratings = condition.get("ratings", {})
        if not isinstance(ratings, Mapping):
            raise ValidationCorpusError(f"Condition {condition_id} ratings must be an object")
        for axis, rating in ratings.items():
            value = float(rating)
            if not str(axis).strip() or value < 0.0 or value > 4.0:
                raise ValidationCorpusError(f"Condition {condition_id} has invalid rating {axis}={rating}")

    public_suite = dict(suite)
    public_suite["documents"] = documents
    public_suite["suite_path"] = str(suite_path)
    public_suite["suite_root"] = str(suite_root)
    public_suite["suite_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in public_suite.items()
                if key not in {"suite_path", "suite_root", "suite_sha256"}
            }
        )
    )
    return public_suite


def validate_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    suite = load_suite(path)
    languages = sorted(
        {language for document in suite["documents"] for language in document["languages"]}
    )
    kinds = {kind: 0 for kind in sorted(DOCUMENT_KINDS)}
    for document in suite["documents"]:
        kinds[str(document["kind"])] += 1
    return {
        "suite_id": suite["id"],
        "suite_sha256": suite["suite_sha256"],
        "documents": len(suite["documents"]),
        "conditions": len(suite["conditions"]),
        "languages": languages,
        "document_kinds": kinds,
    }


def spoken_reference_for_text(document: Mapping[str, Any], text: str) -> str:
    output = str(text)
    for replacement in document.get("spoken_replacements", ()):
        surface = str(replacement.get("surface") or "")
        spoken = str(replacement.get("spoken") or "")
        if surface and spoken:
            output = output.replace(surface, spoken)
    return output
