from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


AFFECT_AXES_V1 = ("calm", "joy", "anger", "sadness", "sarcasm", "questioning")
AFFECT_AXES_V2 = ("calm", "joy", "anger", "sadness", "sarcasm", "whisper")
SUPPORTED_AFFECT_AXES = (*AFFECT_AXES_V1, "whisper")


DEFAULT_CHUNK_FIELDS = (
    "text",
    "voice",
    "language",
    "emotion",
    "affect",
    "affect_guidance_scale",
    "emotion_guidance",
    "boundary_before",
    "boundary_after",
    "boundary_before_id",
    "boundary_after_id",
    "phone_source",
    "phones",
    "phone_ids",
    "predicted_durations",
    "predicted_latent_frames",
    "fixed_latent_frames",
    "prefix_frames_used",
    "reference_key",
    "reference_mask",
    "native_reference_mask",
    "fallback_reference_mask",
)

DEFAULT_TOP_LEVEL_FIELDS = (
    "mode",
    "sample_rate",
    "chunk_count",
    "sampler",
    "steps",
    "speed",
    "affect_guidance_scale",
    "emotion_guidance",
    "guidance_null_reference",
)


@dataclass(frozen=True)
class MetadataMismatch:
    path: str
    left: Any
    right: Any

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "left": self.left, "right": self.right}


def load_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    return payload


def compare_metadata_files(
    left_path: str | Path,
    right_path: str | Path,
    *,
    left_label: str = "left",
    right_label: str = "right",
    chunk_fields: tuple[str, ...] = DEFAULT_CHUNK_FIELDS,
    top_level_fields: tuple[str, ...] = DEFAULT_TOP_LEVEL_FIELDS,
    max_mismatches: int | None = None,
) -> dict[str, Any]:
    left_file = Path(left_path)
    right_file = Path(right_path)
    return compare_long_form_metadata(
        load_metadata(left_file),
        load_metadata(right_file),
        left_label=left_label,
        right_label=right_label,
        left_base_dir=left_file.parent,
        right_base_dir=right_file.parent,
        chunk_fields=chunk_fields,
        top_level_fields=top_level_fields,
        max_mismatches=max_mismatches,
    )


def compare_long_form_metadata(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
    left_base_dir: str | Path | None = None,
    right_base_dir: str | Path | None = None,
    chunk_fields: tuple[str, ...] = DEFAULT_CHUNK_FIELDS,
    top_level_fields: tuple[str, ...] = DEFAULT_TOP_LEVEL_FIELDS,
    max_mismatches: int | None = None,
) -> dict[str, Any]:
    normalized_left = normalize_long_form_metadata(left, base_dir=left_base_dir)
    normalized_right = normalize_long_form_metadata(right, base_dir=right_base_dir)
    mismatches: list[MetadataMismatch] = []

    for field in top_level_fields:
        _compare_value(
            mismatches,
            field,
            normalized_left.get(field),
            normalized_right.get(field),
        )

    left_chunks = normalized_left.get("chunks") or []
    right_chunks = normalized_right.get("chunks") or []
    _compare_value(mismatches, "chunks.length", len(left_chunks), len(right_chunks))
    for index, (left_chunk, right_chunk) in enumerate(zip(left_chunks, right_chunks)):
        for field in chunk_fields:
            _compare_value(
                mismatches,
                f"chunks[{index}].{field}",
                left_chunk.get(field),
                right_chunk.get(field),
            )

    shown_mismatches = mismatches
    if max_mismatches is not None and max_mismatches >= 0:
        shown_mismatches = mismatches[:max_mismatches]
    return {
        "status": "ok" if not mismatches else "mismatch",
        "left_label": left_label,
        "right_label": right_label,
        "left": _metadata_summary(normalized_left),
        "right": _metadata_summary(normalized_right),
        "mismatch_count": len(mismatches),
        "shown_mismatch_count": len(shown_mismatches),
        "truncated": len(shown_mismatches) != len(mismatches),
        "mismatches": [mismatch.to_dict() for mismatch in shown_mismatches],
    }


def normalize_long_form_metadata(
    payload: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(base_dir) if base_dir is not None else None
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("metadata must contain a chunks list")

    normalized_chunks = [
        _normalize_chunk(chunk, index=index, top=payload, base_dir=base)
        for index, chunk in enumerate(chunks)
        if isinstance(chunk, dict)
    ]
    return {
        "mode": _normalize_mode(payload.get("mode")),
        "backend": _scalar(payload.get("backend")),
        "sample_rate": _int_or_none(payload.get("sample_rate")),
        "chunk_count": _int_or_none(payload.get("chunk_count")) or len(normalized_chunks),
        "sampler": _scalar(payload.get("sampler")),
        "steps": _int_or_none(payload.get("steps")),
        "speed": _float_or_none(payload.get("speed")),
        "affect_guidance_scale": _float_or_none(payload.get("affect_guidance_scale")),
        "emotion_guidance": _normalize_guidance(payload.get("emotion_guidance")),
        "guidance_null_reference": _bool_or_none(payload.get("guidance_null_reference")),
        "chunks": normalized_chunks,
    }


def _normalize_chunk(
    chunk: dict[str, Any],
    *,
    index: int,
    top: dict[str, Any],
    base_dir: Path | None,
) -> dict[str, Any]:
    summary = _load_chunk_summary(chunk, base_dir=base_dir)
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    request = metadata.get("request")
    if not isinstance(request, dict):
        request = {}
    prepared = metadata.get("prepared_inputs")
    if not isinstance(prepared, dict):
        prepared = {}
    duration = metadata.get("duration_expansion")
    if not isinstance(duration, dict):
        duration = {}
    g2p = metadata.get("g2p")
    if not isinstance(g2p, dict):
        g2p = {}

    return {
        "index": _int_or_none(chunk.get("index")) if "index" in chunk else index,
        "text": _scalar(chunk.get("text")),
        "voice": _first_scalar(
            chunk.get("voice"),
            summary.get("voice"),
            metadata.get("voice_id"),
            request.get("resolved_voice"),
            top.get("voice"),
        ),
        "language": _first_scalar(
            chunk.get("language"),
            summary.get("language"),
            metadata.get("language"),
            request.get("resolved_language"),
            top.get("language"),
        ),
        "emotion": _first_scalar(
            chunk.get("emotion"),
            summary.get("emotion"),
            metadata.get("emotion"),
            request.get("resolved_emotion"),
            top.get("emotion"),
        ),
        "affect": _normalize_affect(
            _first_present(
                chunk.get("affect"),
                summary.get("affect_values"),
                metadata.get("affect_values"),
                metadata.get("affect_vector"),
                request.get("affect_values"),
                request.get("affect_vector"),
                prepared.get("affect_vector"),
                request.get("affect"),
                top.get("affect"),
            ),
            axes=_first_present(
                chunk.get("affect_axes"),
                summary.get("affect_axes"),
                metadata.get("affect_axes"),
                request.get("affect_axes"),
                top.get("affect_axes"),
            ),
        ),
        "affect_guidance_scale": _float_or_none(
            _first_present(
                chunk.get("affect_guidance_scale"),
                summary.get("affect_guidance_scale"),
                metadata.get("affect_guidance_scale"),
                request.get("affect_guidance_scale"),
                top.get("affect_guidance_scale"),
            )
        ),
        "emotion_guidance": _normalize_guidance(
            _first_present(
                chunk.get("emotion_guidance"),
                summary.get("emotion_guidance"),
                metadata.get("emotion_guidance"),
                request.get("emotion_guidance"),
                top.get("emotion_guidance"),
            )
        ),
        "boundary_before": _first_scalar(
            chunk.get("boundary_before"),
            metadata.get("boundary_before"),
            metadata.get("request_boundary_before"),
            request.get("boundary_before"),
        ),
        "boundary_after": _first_scalar(
            chunk.get("boundary_after"),
            metadata.get("boundary_after"),
            metadata.get("request_boundary_after"),
            request.get("boundary_after"),
        ),
        "boundary_before_id": _int_or_none(_first_present(chunk.get("boundary_before_id"), metadata.get("boundary_before_id"), prepared.get("boundary_before_id"))),
        "boundary_after_id": _int_or_none(_first_present(chunk.get("boundary_after_id"), metadata.get("boundary_after_id"), prepared.get("boundary_after_id"))),
        "split_reason": _scalar(chunk.get("split_reason")),
        "starts_sentence": _bool_or_none(chunk.get("starts_sentence")),
        "ends_sentence": _bool_or_none(chunk.get("ends_sentence")),
        "pause_after_ms": _int_or_none(chunk.get("pause_after_ms")),
        "phone_source": _first_scalar(summary.get("phone_source"), metadata.get("phone_source"), g2p.get("phone_source")),
        "phones": _list_or_none(_first_present(summary.get("phones"), metadata.get("phones"), prepared.get("phones"))),
        "phone_ids": _int_list_or_none(
            _first_present(
                summary.get("phone_ids"),
                metadata.get("phone_ids"),
                prepared.get("active_phone_ids"),
            )
        ),
        "predicted_durations": _int_list_or_none(
            _first_present(
                summary.get("predicted_durations"),
                metadata.get("predicted_durations"),
                duration.get("predicted_durations"),
            )
        ),
        "predicted_latent_frames": _int_or_none(
            _first_present(
                summary.get("predicted_latent_frames"),
                metadata.get("predicted_latent_frames"),
                duration.get("predicted_latent_frames"),
                chunk.get("latent_frames"),
            )
        ),
        "fixed_latent_frames": _int_or_none(
            _first_present(
                metadata.get("fixed_latent_frames"),
                prepared.get("fixed_latent_frames"),
                duration.get("fixed_latent_frames"),
                chunk.get("preflight_fixed_latent_frames"),
            )
        ),
        "prefix_frames_used": _int_or_none(
            _first_present(
                chunk.get("prefix_frames_used"),
                summary.get("prefix_frames_used"),
                metadata.get("prefix_frames_used"),
                prepared.get("prefix_frames_used"),
            )
        ),
        "reference_key": _first_scalar(chunk.get("reference_key"), summary.get("reference_key"), metadata.get("reference_key"), prepared.get("reference_key")),
        "reference_mask": _float_or_none(_first_present(summary.get("reference_mask"), metadata.get("reference_mask"), prepared.get("reference_mask"))),
        "native_reference_mask": _float_or_none(
            _first_present(
                summary.get("native_reference_mask"),
                metadata.get("native_reference_mask"),
                prepared.get("native_reference_mask"),
            )
        ),
        "fallback_reference_mask": _float_or_none(
            _first_present(
                summary.get("fallback_reference_mask"),
                metadata.get("fallback_reference_mask"),
                prepared.get("fallback_reference_mask"),
            )
        ),
    }


def _load_chunk_summary(chunk: dict[str, Any], *, base_dir: Path | None) -> dict[str, Any]:
    path_value = chunk.get("summary_path") or chunk.get("dir")
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    if base_dir is not None and not path.is_absolute():
        direct = base_dir / path
        if direct.exists():
            path = direct
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _compare_value(mismatches: list[MetadataMismatch], path: str, left: Any, right: Any) -> None:
    if left is None and right is None:
        return
    if left != right:
        mismatches.append(MetadataMismatch(path=path, left=left, right=right))


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    chunks = metadata.get("chunks") or []
    return {
        "mode": metadata.get("mode"),
        "backend": metadata.get("backend"),
        "sample_rate": metadata.get("sample_rate"),
        "chunk_count": metadata.get("chunk_count"),
        "chunks": len(chunks),
    }


def _normalize_mode(value: Any) -> str | None:
    if value == "native_chunked":
        return "chunked"
    return _scalar(value)


def _normalize_guidance(value: Any) -> list[dict[str, Any]] | None:
    value = _first_present(value)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        terms: list[dict[str, Any]] = []
        for raw in value.split(","):
            part = raw.strip()
            if not part:
                continue
            if ":" in part:
                emotion, scale = part.split(":", 1)
            else:
                emotion, scale = part, "1.0"
            try:
                scale_value = float(scale.strip())
            except ValueError:
                scale_value = scale.strip()
            terms.append({"emotion": emotion.strip(), "scale": scale_value})
        return terms or None
    if isinstance(value, list):
        terms = []
        for item in value:
            if not isinstance(item, dict):
                continue
            term = {"emotion": _scalar(item.get("emotion"))}
            if "emotion_id" in item:
                term["emotion_id"] = _int_or_none(item.get("emotion_id"))
            if "scale" in item:
                term["scale"] = _float_or_none(item.get("scale"))
            terms.append(term)
        return terms or None
    return None


def _normalize_affect(
    value: Any,
    *,
    axes: Any = None,
) -> dict[str, float] | Any:
    if isinstance(value, dict):
        normalized: dict[str, float] = {}
        for axis in SUPPORTED_AFFECT_AXES:
            number = _float_or_none(value.get(axis))
            if number is not None:
                normalized[axis] = number
        return normalized or value
    axis_order = (
        tuple(str(item) for item in axes)
        if isinstance(axes, (list, tuple))
        else AFFECT_AXES_V1
    )
    if axis_order not in {AFFECT_AXES_V1, AFFECT_AXES_V2}:
        axis_order = AFFECT_AXES_V1
    if isinstance(value, (list, tuple)) and len(value) == len(axis_order):
        numbers = [_float_or_none(item) for item in value]
        if all(number is not None for number in numbers):
            return {axis: float(number) for axis, number in zip(axis_order, numbers)}
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_scalar(*values: Any) -> str | None:
    for value in values:
        scalar = _scalar(value)
        if scalar:
            return scalar
    return None


def _scalar(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    return None


def _list_or_none(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return list(value)
    return None


def _int_list_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    output: list[int] = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is None:
            return None
        output.append(parsed)
    return output
