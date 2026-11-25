"""Bundle contract: what a Scylla's Band model bundle must contain for this runtime.

Three graph contracts are accepted:

- ``scyllasband_measured_controls_v3``: the spoken model. The flow (``vector_estimator[_N]``) takes ``VECTOR_INPUTS``.
- ``scyllasband_measured_controls_v4``: the same graphs plus singing. The flow takes ``VECTOR_INPUTS_V4``, which adds the
  ``delivery_mode_id`` input (0 spoken, 1 sung, 2 rap, 3 ballad; ``controls.delivery_modes``), and each bucket may name an
  optional second decoder ``vocoder_bigvgan[_N]`` with the vocoder's inputs (``controls.decoders``).
- ``scyllasband_measured_controls_v5`` (also written ``..._v5_1``): identity vectors and style rows. The duration, pitch and flow
  graphs take the voice's identity vector (the ``voice_identity`` asset) and a style row: ``style_id`` on the duration
  graph, ``delivery_mode_id`` on the pitch and flow graphs. The rows are the bundle's own table
  (``controls.delivery_modes.ids``); ``controls.delivery_modes.duration_styles_trained`` (and ``pitch_styles_trained``)
  list the ones speech may use, ``controls_absent.styles`` those of them that take neither intensity nor pace.
  The pitch graphs also take the contour rendered so far as a prefix. A bucket may name any decoder that
  ``controls.decoders`` declares.

``CONTRACT_VERSION`` stays ``1.0.0``: it names the manifest layout (assets, components, buckets, controls), which v4 bundles
keep; what changed is the graph contract, which the manifest declares separately and the runtime reads per bundle. A v3
bundle runs exactly as before (the mode input is never fed); a v4 bundle is always fed the mode input, 0 for speech.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

CONTRACT_VERSION = "1.0.0"
GRAPH_CONTRACT = "scyllasband_measured_controls_v3"
GRAPH_CONTRACT_V4 = "scyllasband_measured_controls_v4"
GRAPH_CONTRACT_V5 = "scyllasband_measured_controls_v5"
GRAPH_CONTRACTS_V5 = (GRAPH_CONTRACT_V5, "scyllasband_measured_controls_v5_1")   # one contract under either name
GRAPH_CONTRACTS = (GRAPH_CONTRACT, GRAPH_CONTRACT_V4, *GRAPH_CONTRACTS_V5)
DURATION_INPUTS = ("span_phone_ids", "span_segment_ids", "span_mask", "voice_id", "language_id", "delivery_values", "delivery_present")
CONTEXT_INPUTS = ("span_context_phone_ids", "span_context_segment_ids", "span_context_mask")
EVENT_INPUTS = ("expanded_boundary_event_ids", "expanded_modifier_event_ids", "expanded_phone_phase", "expanded_phone_log_duration",
                "expanded_sentence_type_ids")
VECTOR_INPUTS = ("noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values", "delivery_present", "latent_mask",
                 "span_context_hidden", "prefix_latents", "prefix_mask", *EVENT_INPUTS, "expanded_melody")
DELIVERY_MODE_INPUT = "delivery_mode_id"
VECTOR_INPUTS_V4 = (*VECTOR_INPUTS, DELIVERY_MODE_INPUT)
PITCH_INPUTS = ("noise", "time", "expanded_phone_ids", "voice_id", "language_id", "delivery_values", "delivery_present", "latent_mask",
                "span_context_hidden", *EVENT_INPUTS)
# v5: the duration, pitch and flow graphs take the voice's identity vector (``identity_reference``
# f32 [1, dim], from the bundle's ``voice_identity`` asset) and a style row (int64 [1]: ``style_id`` on the duration graph,
# ``delivery_mode_id`` on the pitch and flow graphs), and the pitch graphs take the contour rendered so far as a prefix
# (``prefix_latents`` f32 [1, 2, frames], ``prefix_mask`` bool [1, frames]).
IDENTITY_INPUT = "identity_reference"
STYLE_INPUT = "style_id"
DURATION_INPUTS_V5 = (*DURATION_INPUTS, IDENTITY_INPUT, STYLE_INPUT)
VECTOR_INPUTS_V5 = (*VECTOR_INPUTS, IDENTITY_INPUT, DELIVERY_MODE_INPUT)
PITCH_INPUTS_V5 = (*PITCH_INPUTS, "prefix_latents", "prefix_mask", IDENTITY_INPUT, DELIVERY_MODE_INPUT)
VOCODER_INPUTS = ("latents", "latent_mask", "voice_id", "language_id", "emotion_id")
COMPONENT_INPUTS = {"g2p": ("text",), "duration_predictor": DURATION_INPUTS, "vector_context_encoder": CONTEXT_INPUTS,
                    "pitch_context_encoder": CONTEXT_INPUTS}
OPTIONAL_BUCKET_COMPONENTS = {"vocoder_bigvgan": VOCODER_INPUTS}   # bucket key -> inputs; validated only when a bucket names one
REQUIRED_ASSETS = ("phone_vocab", "voice_index", "language_index", "g2p_config", "g2p_tokenizer", "g2p_language_map")
BACKEND_FORMATS = {"onnx": "onnx", "litert": "litert", "coreml": "coreml", "coreai": "coreai"}


class BundleError(ValueError):
    pass


def load_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / "manifest.json"
    if not path.is_file():
        raise BundleError(f"No manifest.json in {bundle_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def vector_inputs(graph_contract: str | None) -> tuple[str, ...]:
    """The flow's inputs under a graph contract: v4 adds the delivery mode id (v3 has none), v5 the identity vector."""
    if graph_contract in GRAPH_CONTRACTS_V5:
        return VECTOR_INPUTS_V5
    return VECTOR_INPUTS_V4 if graph_contract == GRAPH_CONTRACT_V4 else VECTOR_INPUTS


def validate_bundle_layout(bundle_dir: str | Path) -> dict[str, Any]:
    """Raise BundleError when the bundle cannot run on this runtime; return a short summary otherwise."""
    bundle_dir = Path(bundle_dir)
    manifest = load_manifest(bundle_dir)
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise BundleError(f"Bundle contract_version {manifest.get('contract_version')!r} != {CONTRACT_VERSION!r}")
    controls = manifest.get("controls") or {}
    graph_contract = controls.get("graph_input_contract")
    if graph_contract not in GRAPH_CONTRACTS:   # v2 bundles: use a runtime from before v3
        raise BundleError(f"Bundle graph contract {graph_contract!r} is not supported by this runtime "
                          f"(expected one of {', '.join(GRAPH_CONTRACTS)}). Download the current model: python -m scyllasband download --yes")
    for key in REQUIRED_ASSETS:
        rel = (manifest.get("assets") or {}).get(key)
        if not rel or not (bundle_dir / rel).is_file():
            raise BundleError(f"Bundle asset {key!r} is missing ({rel})")
    components = manifest.get("components") or {}
    buckets = (controls.get("target_buckets") or {}).get("buckets") or []
    if not buckets:
        raise BundleError("Bundle declares no latent-frame buckets")
    expected = dict(COMPONENT_INPUTS)
    v5 = graph_contract in GRAPH_CONTRACTS_V5
    optional_keys = dict(OPTIONAL_BUCKET_COMPONENTS)
    if v5:
        expected["duration_predictor"] = DURATION_INPUTS_V5
        _validate_v5(bundle_dir, manifest)
        # Any decoder the manifest declares may be named by a bucket under its component's base name.
        for spec in (controls.get("decoders") or {}).values():
            if str(spec.get("component") or "vocoder") != "vocoder":
                optional_keys[str(spec["component"])] = VOCODER_INPUTS
    optional: dict[str, tuple[str, ...]] = {}
    for bucket in buckets:
        expected[str(bucket["vector_estimator"])] = vector_inputs(graph_contract)
        expected[str(bucket["pitch_estimator"])] = PITCH_INPUTS_V5 if v5 else PITCH_INPUTS
        expected[str(bucket["vocoder"])] = VOCODER_INPUTS
        for key, inputs in optional_keys.items():
            if bucket.get(key):
                optional[str(bucket[key])] = inputs
    backends = sorted({fmt for spec in components.values() for fmt in (spec.get("artifacts") or {}) if fmt in BACKEND_FORMATS})
    if not backends:
        raise BundleError(f"Bundle components declare no {', '.join(BACKEND_FORMATS)} artifacts")
    for name, inputs in expected.items():
        spec = components.get(name)
        if spec is None:
            raise BundleError(f"Bundle component {name!r} is missing")
        if tuple(spec.get("inputs") or ()) != inputs:
            raise BundleError(f"Bundle component {name!r} inputs {spec.get('inputs')} differ from the contract {list(inputs)}")
        for backend in backends:
            artifact = (spec.get("artifacts") or {}).get(backend)
            if not artifact or not (bundle_dir / artifact["path"]).exists():   # Core ML/Core AI assets are directories
                raise BundleError(f"Bundle component {name!r} has no {backend} artifact at {artifact and artifact.get('path')}")
    # An optional component (a second decoder) need not exist for every backend; the artifacts it declares must be there.
    for name, inputs in optional.items():
        spec = components.get(name)
        if spec is None:
            raise BundleError(f"Bundle component {name!r} is named by a bucket but missing")
        if tuple(spec.get("inputs") or ()) != inputs:
            raise BundleError(f"Bundle component {name!r} inputs {spec.get('inputs')} differ from the contract {list(inputs)}")
        for backend, artifact in (spec.get("artifacts") or {}).items():
            if backend in BACKEND_FORMATS and not (bundle_dir / artifact["path"]).exists():
                raise BundleError(f"Bundle component {name!r} has no {backend} artifact at {artifact.get('path')}")
    voices = manifest.get("voices") or []
    if not voices or any(not v.get("languages") for v in voices):
        raise BundleError("Bundle voices must each declare their languages")
    return dict(bundle_dir=str(bundle_dir), backends=backends, voices=[v["id"] for v in voices],
                buckets=sorted(int(b["latent_frames"]) for b in buckets), release=_release(controls), graph_contract=graph_contract,
                singing=bool((controls.get("delivery_modes") or {}).get("enabled")))


def _validate_v5(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    """What a v5 bundle declares beside its graphs: an identity vector for every voice, the style table with the rows
    speech may use, and the pause mark it asks for."""
    controls, assets = manifest.get("controls") or {}, manifest.get("assets") or {}
    rel = assets.get("voice_identity")
    if not rel or not (bundle_dir / rel).is_file():
        raise BundleError(f"Bundle asset 'voice_identity' is missing ({rel})")
    table = json.loads((bundle_dir / rel).read_text(encoding="utf-8"))
    try:
        dim, vectors = int(table["dim"]), dict(table["voices"])
    except (KeyError, TypeError, ValueError):
        raise BundleError("Bundle asset 'voice_identity' must hold dim and voices {voice: vector}") from None
    for voice in manifest.get("voices") or []:
        if len(vectors.get(voice.get("id")) or ()) != dim:
            raise BundleError(f"Bundle asset 'voice_identity' has no {dim}-value vector for voice {voice.get('id')!r}")
    modes = controls.get("delivery_modes") or {}
    ids = modes.get("ids") if modes.get("enabled") else None
    if not isinstance(ids, Mapping) or "spoken" not in ids:
        raise BundleError("A v5 bundle must enable controls.delivery_modes and give its style rows (ids) with a 'spoken' row")
    trained = modes.get("duration_styles_trained")
    if not isinstance(trained, list) or "spoken" not in trained or any(style not in ids for style in trained):
        raise BundleError("A v5 bundle must list the styles its duration graph was trained on "
                          "(controls.delivery_modes.duration_styles_trained: rows of ids, 'spoken' among them)")
    marks = controls.get("pause_marks")
    if not isinstance(marks, Mapping) or marks.get("default") not in ("all", "none"):
        raise BundleError("A v5 bundle must state controls.pause_marks.default (all or none)")
    if marks["default"] == "all":
        vocab = json.loads((bundle_dir / assets["phone_vocab"]).read_text(encoding="utf-8"))
        if marks.get("token") not in dict(vocab.get("token_to_id", vocab)):
            raise BundleError(f"Bundle pause mark {marks.get('token')!r} is not in the phone vocabulary")


def _release(controls: Mapping[str, Any]) -> str | None:
    delivery = controls.get("delivery") or {}
    return delivery.get("release_id")
