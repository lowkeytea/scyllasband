"""Bundle contract: what a Scylla's Band model bundle must contain for this runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "1.0.0"
GRAPH_CONTRACT = "scyllasband_measured_delivery_v2"
DURATION_INPUTS = ("span_phone_ids", "span_segment_ids", "span_mask", "voice_id", "language_id", "delivery_values", "delivery_present")
CONTEXT_INPUTS = ("span_context_phone_ids", "span_context_segment_ids", "span_context_mask")
EVENT_INPUTS = ("expanded_boundary_event_ids", "expanded_modifier_event_ids", "expanded_phone_phase", "expanded_phone_log_duration",
                "expanded_sentence_type_ids")
VECTOR_INPUTS = ("noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values", "delivery_present", "latent_mask",
                 "span_context_hidden", "prefix_latents", "prefix_mask", *EVENT_INPUTS)
VOCODER_INPUTS = ("latents", "latent_mask", "voice_id", "language_id", "emotion_id")
COMPONENT_INPUTS = {"g2p": ("text",), "duration_predictor": DURATION_INPUTS, "vector_context_encoder": CONTEXT_INPUTS}
REQUIRED_ASSETS = ("phone_vocab", "voice_index", "language_index", "g2p_config", "g2p_tokenizer", "g2p_language_map")
BACKEND_FORMATS = {"onnx": "onnx", "litert": "litert"}


class BundleError(ValueError):
    pass


def load_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / "manifest.json"
    if not path.is_file():
        raise BundleError(f"No manifest.json in {bundle_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bundle_layout(bundle_dir: str | Path) -> dict[str, Any]:
    """Raise BundleError when the bundle cannot run on this runtime; return a short summary otherwise."""
    bundle_dir = Path(bundle_dir)
    manifest = load_manifest(bundle_dir)
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise BundleError(f"Bundle contract_version {manifest.get('contract_version')!r} != {CONTRACT_VERSION!r}")
    controls = manifest.get("controls") or {}
    if controls.get("graph_input_contract") != GRAPH_CONTRACT:
        raise BundleError(f"Bundle graph contract {controls.get('graph_input_contract')!r} is not supported by this runtime "
                          f"(expected {GRAPH_CONTRACT!r}). Download the current model: python -m scyllasband download --yes")
    for key in REQUIRED_ASSETS:
        rel = (manifest.get("assets") or {}).get(key)
        if not rel or not (bundle_dir / rel).is_file():
            raise BundleError(f"Bundle asset {key!r} is missing ({rel})")
    components = manifest.get("components") or {}
    buckets = (controls.get("target_buckets") or {}).get("buckets") or []
    if not buckets:
        raise BundleError("Bundle declares no latent-frame buckets")
    expected = dict(COMPONENT_INPUTS)
    for bucket in buckets:
        expected[str(bucket["vector_estimator"])] = VECTOR_INPUTS
        expected[str(bucket["vocoder"])] = VOCODER_INPUTS
    backends = sorted({fmt for spec in components.values() for fmt in (spec.get("artifacts") or {}) if fmt in BACKEND_FORMATS})
    if not backends:
        raise BundleError("Bundle components declare no onnx or litert artifacts")
    for name, inputs in expected.items():
        spec = components.get(name)
        if spec is None:
            raise BundleError(f"Bundle component {name!r} is missing")
        if tuple(spec.get("inputs") or ()) != inputs:
            raise BundleError(f"Bundle component {name!r} inputs {spec.get('inputs')} differ from the contract {list(inputs)}")
        for backend in backends:
            artifact = (spec.get("artifacts") or {}).get(backend)
            if not artifact or not (bundle_dir / artifact["path"]).is_file():
                raise BundleError(f"Bundle component {name!r} has no {backend} artifact at {artifact and artifact.get('path')}")
    voices = manifest.get("voices") or []
    if not voices or any(not v.get("languages") for v in voices):
        raise BundleError("Bundle voices must each declare their languages")
    return dict(bundle_dir=str(bundle_dir), backends=backends, voices=[v["id"] for v in voices],
                buckets=sorted(int(b["latent_frames"]) for b in buckets), release=_release(controls))


def _release(controls: Mapping[str, Any]) -> str | None:
    delivery = controls.get("delivery") or {}
    return delivery.get("release_id")
