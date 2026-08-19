"""Deterministic routing into Scylla's Band source-reference libraries.

This module is intentionally NumPy-only so the public ONNX/LiteRT runners and
the training-side PyTorch diagnostic path can share one routing contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence
import math

import numpy as np


REFERENCE_ROUTING_VERSION = "full_library_axis_interpolation_v1"
REFERENCE_ROUTING_V4_VERSION = "native_exact_primary_relative_residual_v4"
REFERENCE_PACK_V4_SCHEMA = "scyllasband_reference_pack_v4"
REFERENCE_PACK_V4_AXES = ("calm", "joy", "anger", "sadness", "whisper")
REFERENCE_PACK_V4_IDENTITY_DIM = 512
REFERENCE_PACK_V4_PROSODY_DIM = 32


@dataclass(frozen=True)
class RoutedReference:
    style: np.ndarray
    prosody: np.ndarray
    key: str
    contributing_axes: tuple[str, ...]
    candidate_counts: dict[str, int]

@dataclass(frozen=True)
class RoutedReferenceV4:
    identity: np.ndarray
    prosody_baseline: np.ndarray
    prosody_delta: np.ndarray
    prosody_feature_mask: np.ndarray
    prosody_confidence: float
    baseline_index: int
    prototype_index: int | None
    route_kind: str
    reference_indices: tuple[int, ...]
    identity_window_indices: tuple[int, ...]


class ReferencePackV4Error(ValueError):
    """Raised when a schema-v4 pack cannot satisfy the routing contract."""




_BASE_AFFECT_ALIASES = {
    "calm": {"calm", "neutral", "friendly"},
    "joy": {"joy", "happy", "excited"},
    "anger": {"anger", "angry", "frustrated"},
    "sadness": {"sadness", "sad", "unsure"},
    "sarcasm": {"sarcasm", "sarcastic"},
    "whisper": {"whisper", "whispered"},
}


def route_affect_reference(
    pack: Mapping[str, np.ndarray],
    *,
    language: str,
    axes: Sequence[str],
    values: Sequence[float] | np.ndarray,
) -> RoutedReference | None:
    """Interpolate native source references for a affect request.

    The neutral library centroid anchors zero affect. Each active axis adds the
    centroid of exact source references tagged for that affect or delivery.
    Weights are barycentric, so mixed emotions remain bounded by observed
    reference representations rather than extrapolating beyond them.
    """

    identifiers = pack.get("reference_ids")
    styles = pack.get("reference_style_embeddings")
    prosodies = pack.get("reference_prosody_stats")
    languages = pack.get("reference_languages")
    base_affects = pack.get("reference_base_affects")
    delivery_modes = pack.get("reference_delivery_modes")
    overlay_tags = pack.get("reference_overlay_tags_json")
    required = (identifiers, styles, prosodies, languages, base_affects)
    if any(value is None for value in required):
        return None

    style_matrix = np.asarray(styles, dtype=np.float32)
    prosody_matrix = np.asarray(prosodies, dtype=np.float32)
    if style_matrix.ndim != 2 or prosody_matrix.ndim != 2:
        return None
    row_count = min(style_matrix.shape[0], prosody_matrix.shape[0])
    language_rows = _string_rows(languages, row_count)
    affect_rows = _string_rows(base_affects, row_count)
    delivery_rows = _string_rows(delivery_modes, row_count)
    overlay_rows = _overlay_rows(overlay_tags, row_count)
    requested_language = _normalized(language)
    language_mask = np.asarray(
        [value == requested_language for value in language_rows], dtype=np.bool_
    )
    if not bool(language_mask.any()):
        return None

    axis_names = tuple(str(axis) for axis in axes)
    value_array = np.asarray(values, dtype=np.float32).reshape(-1)
    if value_array.size != len(axis_names):
        raise ValueError(
            f"Reference routing received {value_array.size} values for {len(axis_names)} axes"
        )
    value_array = np.clip(np.nan_to_num(value_array), 0.0, 1.0)

    neutral_mask = language_mask & np.asarray(
        [value == "neutral" for value in affect_rows], dtype=np.bool_
    )
    if not bool(neutral_mask.any()):
        neutral_mask = language_mask & np.asarray(
            [value in {"calm", "friendly"} for value in affect_rows],
            dtype=np.bool_,
        )
    if not bool(neutral_mask.any()):
        neutral_mask = language_mask
    neutral_style = style_matrix[:row_count][neutral_mask].mean(axis=0)
    neutral_prosody = prosody_matrix[:row_count][neutral_mask].mean(axis=0)

    weighted_styles: list[np.ndarray] = []
    weighted_prosodies: list[np.ndarray] = []
    weights: list[float] = []
    contributing: list[str] = []
    counts: dict[str, int] = {}
    neutral_weight = max(0.0, 1.0 - float(value_array.max(initial=0.0)))
    if neutral_weight > 0.0 or not bool(np.any(value_array > 0.0)):
        weighted_styles.append(neutral_style)
        weighted_prosodies.append(neutral_prosody)
        weights.append(max(neutral_weight, 1.0 if not bool(np.any(value_array > 0.0)) else 0.0))

    for axis, weight in zip(axis_names, value_array.tolist(), strict=True):
        if weight <= 0.0:
            continue
        aliases = _BASE_AFFECT_ALIASES.get(_normalized(axis), {_normalized(axis)})
        mask_values = []
        for base_affect, delivery, tags in zip(
            affect_rows, delivery_rows, overlay_rows, strict=True
        ):
            tokens = {base_affect, delivery, *tags}
            mask_values.append(bool(tokens & aliases))
        axis_mask = language_mask & np.asarray(mask_values, dtype=np.bool_)
        count = int(axis_mask.sum())
        counts[str(axis)] = count
        if count <= 0:
            continue
        weighted_styles.append(style_matrix[:row_count][axis_mask].mean(axis=0))
        weighted_prosodies.append(prosody_matrix[:row_count][axis_mask].mean(axis=0))
        weights.append(float(weight))
        contributing.append(str(axis))

    if not weights:
        return None
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= max(float(weight_array.sum()), 1e-8)
    style = np.sum(np.stack(weighted_styles) * weight_array[:, None], axis=0)
    prosody = np.sum(np.stack(weighted_prosodies) * weight_array[:, None], axis=0)
    active = ",".join(
        f"{axis}={float(value):.3f}"
        for axis, value in zip(axis_names, value_array.tolist(), strict=True)
        if value > 0.0
    ) or "neutral"
    return RoutedReference(
        style=np.asarray(style, dtype=np.float32),
        prosody=np.asarray(prosody, dtype=np.float32),
        key=f"{requested_language}|affect_route|{active}",
        contributing_axes=tuple(contributing),
        candidate_counts=counts,
    )


def _normalized(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _string_rows(value: np.ndarray | None, count: int) -> list[str]:
    if value is None:
        return [""] * count
    rows = np.asarray(value).reshape(-1).tolist()
    return [_normalized(rows[index]) if index < len(rows) else "" for index in range(count)]


def _overlay_rows(value: np.ndarray | None, count: int) -> list[set[str]]:
    if value is None:
        return [set() for _ in range(count)]
    rows = np.asarray(value).reshape(-1).tolist()
    output: list[set[str]] = []
    for index in range(count):
        raw = rows[index] if index < len(rows) else "[]"
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = []
        output.append({_normalized(item) for item in parsed or () if _normalized(item)})
    return output
def route_reference_pack_v4(
    pack: Mapping[str, np.ndarray],
    *,
    language: str,
    affect_values: Sequence[float] | np.ndarray,
    affect_mask: Sequence[float] | np.ndarray,
) -> RoutedReferenceV4:
    """Resolve identity, native baseline, and an optional exact residual.

    This is the inference-only form of the training router. Runtime inference
    has no sample to self-exclude, so every qualified identity window and each
    prepared baseline or prototype member remains eligible.
    """

    schema = str(np.asarray(pack.get("schema_version", "")).item())
    if schema != REFERENCE_PACK_V4_SCHEMA:
        raise ReferencePackV4Error(
            f"expected {REFERENCE_PACK_V4_SCHEMA!r}, got {schema!r}"
        )
    axes = tuple(
        str(item)
        for item in np.asarray(pack.get("affect_axis_order", ())).reshape(-1)
    )
    if axes != REFERENCE_PACK_V4_AXES:
        raise ReferencePackV4Error(
            f"expected affect axes {REFERENCE_PACK_V4_AXES!r}, got {axes!r}"
        )
    values = np.asarray(affect_values, dtype=np.float32).reshape(-1)
    masks = np.asarray(affect_mask, dtype=np.float32).reshape(-1)
    if values.shape != (5,) or masks.shape != (5,):
        raise ReferencePackV4Error(
            "v4 routing requires five affect values and masks"
        )

    languages = np.asarray(pack["baseline_languages"]).astype(str)
    matches = np.flatnonzero(languages == str(language))
    if not matches.size:
        raise ReferencePackV4Error(
            f"no native prosody baseline for {language!r}"
        )
    baseline_index = int(matches[0])
    global_location = np.asarray(
        pack["prosody_global_locations"], dtype=np.float32
    )
    global_scale = np.asarray(
        pack["prosody_global_scales"], dtype=np.float32
    )
    baseline_raw = np.asarray(
        pack["baseline_locations"], dtype=np.float32
    )[baseline_index]
    baseline_mask = np.asarray(
        pack["baseline_feature_masks"], dtype=np.float32
    )[baseline_index]
    baseline_confidence = float(
        np.asarray(pack["baseline_confidence"])[baseline_index]
    )
    if global_location.shape != (REFERENCE_PACK_V4_PROSODY_DIM,):
        raise ReferencePackV4Error(
            "v4 pack has an invalid prosody global location"
        )
    if (
        global_scale.shape != (REFERENCE_PACK_V4_PROSODY_DIM,)
        or np.any(global_scale <= 0.0)
    ):
        raise ReferencePackV4Error(
            "v4 pack has an invalid prosody global scale"
        )
    baseline = (
        (baseline_raw - global_location) / global_scale
    ) * baseline_mask

    selected_prototype: int | None = None
    candidate = np.flatnonzero(
        (
            np.asarray(pack["prototype_languages"]).astype(str)
            == str(language)
        )
        & (
            np.asarray(
                pack["prototype_baseline_indices"], dtype=np.int64
            )
            == baseline_index
        )
    )
    if candidate.size:
        centers = np.asarray(
            pack["prototype_affect_centers"], dtype=np.float32
        )[candidate]
        center_masks = np.asarray(
            pack["prototype_affect_masks"], dtype=np.float32
        )[candidate]
        target_mask = masks > 0.5
        mask_exact = np.all(
            (center_masks > 0.5) == target_mask.reshape(1, -1),
            axis=1,
        )
        value_exact = np.all(
            np.isclose(
                centers, values.reshape(1, -1), atol=1e-6
            )
            | ~target_mask.reshape(1, -1),
            axis=1,
        )
        exact = np.flatnonzero(mask_exact & value_exact)
        if exact.size:
            if exact.size > 1:
                confidence = np.asarray(
                    pack["prototype_confidence"], dtype=np.float32
                )[candidate[exact]]
                support = np.asarray(
                    pack["prototype_distinct_source_counts"],
                    dtype=np.int64,
                )[candidate[exact]]
                exact = exact[
                    np.lexsort(
                        (candidate[exact], -support, -confidence)
                    )
                ]
            selected_prototype = int(candidate[int(exact[0])])

    delta = np.zeros(
        REFERENCE_PACK_V4_PROSODY_DIM, dtype=np.float32
    )
    feature_mask = baseline_mask.copy()
    confidence = baseline_confidence
    references = tuple(
        int(index)
        for index in np.asarray(
            pack["baseline_reference_indices"]
        )[baseline_index].tolist()
        if int(index) >= 0
    )
    route_kind = "global_affect_only"
    if selected_prototype is not None:
        prototype_references = tuple(
            int(index)
            for index in np.asarray(
                pack["prototype_reference_indices"]
            )[selected_prototype].tolist()
            if int(index) >= 0
        )
        if prototype_references:
            delta = np.asarray(
                pack["prototype_prosody_deltas"], dtype=np.float32
            )[selected_prototype]
            prototype_mask = np.asarray(
                pack["prototype_feature_masks"], dtype=np.float32
            )[selected_prototype]
            feature_mask = baseline_mask * prototype_mask
            delta = delta * feature_mask
            confidence = float(
                np.asarray(
                    pack["prototype_confidence"], dtype=np.float32
                )[selected_prototype]
            )
            references = prototype_references
            route_kind = "native_exact"
        else:
            selected_prototype = None

    identity_embeddings = np.asarray(
        pack["identity_embeddings"], dtype=np.float32
    )
    identity_weights = np.asarray(
        pack["identity_embedding_weights"], dtype=np.float32
    ).reshape(-1)
    identity_references = np.asarray(
        pack["identity_window_reference_indices"], dtype=np.int64
    ).reshape(-1)
    if (
        identity_embeddings.ndim != 2
        or identity_embeddings.shape[1]
        != REFERENCE_PACK_V4_IDENTITY_DIM
    ):
        raise ReferencePackV4Error(
            "v4 pack has invalid identity embeddings"
        )
    if (
        identity_embeddings.shape[0] == 0
        or identity_weights.shape
        != (identity_embeddings.shape[0],)
        or identity_references.shape
        != (identity_embeddings.shape[0],)
    ):
        raise ReferencePackV4Error(
            "v4 pack has invalid identity weights or indices"
        )
    weight_sum = float(identity_weights.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ReferencePackV4Error(
            "v4 pack identity weights must have positive mass"
        )
    identity_weights = identity_weights / weight_sum
    identity = np.sum(
        identity_embeddings * identity_weights[:, None], axis=0
    )
    norm = float(np.linalg.norm(identity))
    if norm > 1e-12:
        identity /= norm
    outputs = (identity, baseline, delta, feature_mask)
    if any(not np.isfinite(value).all() for value in outputs):
        raise ReferencePackV4Error(
            "v4 routing produced non-finite features"
        )
    return RoutedReferenceV4(
        identity=identity.astype(np.float32),
        prosody_baseline=baseline.astype(np.float32),
        prosody_delta=delta.astype(np.float32),
        prosody_feature_mask=feature_mask.astype(np.float32),
        prosody_confidence=float(
            np.clip(confidence, 0.0, 1.0)
        ),
        baseline_index=baseline_index,
        prototype_index=selected_prototype,
        route_kind=route_kind,
        reference_indices=references,
        identity_window_indices=tuple(
            range(identity_embeddings.shape[0])
        ),
    )
