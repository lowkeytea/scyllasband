"""LiteRT graph runner for Scylla's Band bundles."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Mapping

import numpy as np

from .contract import (
    ScyllasBandBundleManifest,
    validate_duration_hierarchy_sampling_contract,
    validate_duration_pause_presence_contract,
    validate_vector_timing_conditioning_contract,
)
from .g2p_phrases import (
    BOUNDARY_PHONE_TOKENS,
    DEFAULT_G2P_PHRASE_MAX_CHARS,
    g2p_phrase_segments,
    is_non_acoustic_phone_modifier,
    punctuation_phone_token,
)
from .text_normalizer import SPOKEN_TEXT_NORMALIZER_SHA256
from .reference_routing import (
    REFERENCE_ROUTING_VERSION,
    REFERENCE_ROUTING_V4_VERSION,
    route_affect_reference,
    route_reference_pack_v4,
)


@dataclass(frozen=True)
class LiteRTSynthesisResult:
    audio: np.ndarray
    metadata: dict[str, Any]
    latents: np.ndarray | None = None


@dataclass(frozen=True)
class _EmotionGuidanceTerm:
    emotion: str
    emotion_id: int
    scale: float


@dataclass(frozen=True)
class _AffectFeatures:
    enabled: bool
    axes: tuple[str, ...]
    values: np.ndarray
    condition_mask: np.ndarray
    requested: object
    preset: str | None


@dataclass(frozen=True)
class _ReferenceFeatures:
    style: np.ndarray
    prosody: np.ndarray
    identity: np.ndarray
    prosody_baseline: np.ndarray
    prosody_delta: np.ndarray
    prosody_feature_mask: np.ndarray
    prosody_confidence: np.ndarray
    mask: np.ndarray
    native_mask: float
    fallback_mask: float
    key: str | None
    path: str | None


@dataclass(frozen=True)
class _VectorTimingFeatures:
    boundary_event_ids: np.ndarray
    modifier_event_ids: np.ndarray
    phone_phase: np.ndarray
    phone_log_duration: np.ndarray
    unrepresented_zero_frame_modifier_events: int


PROSODY_DROP_INDICES = (0, 14)
PROSODY_LOG1P_INDICES = (1, 2, 5, 6, 7, 8, 9, 10, 11, 13)
_FRONTEND_CACHE_SIZE = 256
_WORD_BOUNDARY_MARKER = "<word_boundary>"


class LiteRTRunner:
    backend_name = "litert"

    """Execute the fixed-shape LiteRT Scylla's Band graph chain.

    This module is public-runtime code. It deliberately consumes only the
    Scylla's Band bundle contract and bundle assets; it does not import scyllasband-trainer.
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        manifest: ScyllasBandBundleManifest,
        *,
        sessions: Mapping[str, Any] | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = manifest
        self.phone_to_id = _load_token_to_id(self.bundle_dir / manifest.assets["phone_vocab"])
        self.voice_to_id = _load_index(self.bundle_dir / manifest.assets["voice_index"])
        self.language_to_id = _load_index(self.bundle_dir / manifest.assets["language_index"])
        emotion_asset = manifest.assets.get("emotion_index") or manifest.assets.get("emotions")
        self.emotion_to_id = (
            _load_index(self.bundle_dir / emotion_asset) if emotion_asset else {"neutral": 0}
        )
        self.g2p_config = _load_json(self.bundle_dir / manifest.assets["g2p_config"])
        self.g2p_tokenizer = _load_json(self.bundle_dir / manifest.assets["g2p_tokenizer"])
        self.g2p_language_map = _load_json(self.bundle_dir / manifest.assets["g2p_language_map"])
        normalization_asset = manifest.assets.get("g2p_normalization")
        g2p_normalization = (
            _load_optional_json(self.bundle_dir / normalization_asset)
            if normalization_asset
            else None
        ) or {}
        # Optional bundle-declared remap for punctuation phone tokens whose
        # embeddings the acoustic model never (or barely) trained on. A bundle
        # that introduces typed boundary tokens without matching training data
        # can point them back at a well-trained token (usually <sil>) instead of
        # injecting an untrained embedding into the middle of a phone sequence.
        self.g2p_punctuation_token_remap = {
            str(key): str(value)
            for key, value in dict(
                g2p_normalization.get("punctuation_token_remap") or {}
            ).items()
        }
        self.g2p_punctuation_token_remap_scope = str(
            g2p_normalization.get("punctuation_token_remap_scope")
            or "all_boundaries"
        )
        if self.g2p_punctuation_token_remap_scope not in {
            "all_boundaries",
            "continuation_only",
        }:
            raise ValueError(
                "Unsupported punctuation_token_remap_scope: "
                f"{self.g2p_punctuation_token_remap_scope!r}"
            )
        override_asset = manifest.assets.get("g2p_pronunciation_overrides")
        self.g2p_pronunciation_overrides = (
            _load_optional_json(self.bundle_dir / override_asset) if override_asset else {}
        )
        declared_normalizer_sha = str(g2p_normalization.get("contract_sha256") or "")
        if declared_normalizer_sha and declared_normalizer_sha != SPOKEN_TEXT_NORMALIZER_SHA256:
            raise ValueError(
                "Bundle spoken-text normalizer contract does not match this runtime: "
                f"bundle={declared_normalizer_sha} runtime={SPOKEN_TEXT_NORMALIZER_SHA256}"
            )
        self.export_status = _load_optional_json(self.bundle_dir / "export_status.json")
        controls = dict(getattr(self.manifest, "controls", {}) or {})
        self.duration_pause_presence_config = (
            validate_duration_pause_presence_contract(self.manifest)
        )
        self.duration_pause_presence_enabled = bool(
            self.duration_pause_presence_config.get("enabled", False)
        )
        self.duration_pause_presence_threshold_probability = float(
            self.duration_pause_presence_config.get(
                "threshold_probability",
                0.5,
            )
        )
        self.duration_hierarchy_config = (
            validate_duration_hierarchy_sampling_contract(self.manifest)
        )
        self.duration_hierarchy_enabled = bool(
            self.duration_hierarchy_config.get("enabled", False)
        )
        self.vector_timing_config = validate_vector_timing_conditioning_contract(
            self.manifest,
            self.bundle_dir,
        )
        self.vector_timing_enabled = bool(
            self.vector_timing_config.get("enabled", False)
        )
        affect_config = controls.get("affect", {})
        self.affect_config = (
            dict(affect_config) if isinstance(affect_config, Mapping) else {}
        )
        self.affect_enabled = bool(self.affect_config.get("enabled", False))
        self.affect_axes = tuple(str(item) for item in self.affect_config.get("axes", ()))
        self.affect_presets = dict(self.affect_config.get("presets", {}) or {})
        self.affect_legacy_presets = dict(
            self.affect_config.get("legacy_presets", {}) or {}
        )
        self.affect_partial_defaults = {
            str(axis): float(value)
            for axis, value in dict(
                self.affect_config.get("partial_defaults", {}) or {}
            ).items()
        }
        self.affect_axis_minimums = {
            str(axis): float(value)
            for axis, value in dict(
                self.affect_config.get("axis_minimums", {}) or {}
            ).items()
        }
        self.affect_axis_maximums = {
            str(axis): float(value)
            for axis, value in dict(
                self.affect_config.get("axis_maximums", {}) or {}
            ).items()
        }
        word_boundaries = controls.get("word_boundaries", {})
        if not isinstance(word_boundaries, Mapping):
            word_boundaries = {}
        self.word_boundary_config = dict(word_boundaries)
        self.word_boundary_enabled = bool(
            self.word_boundary_config.get("enabled", False)
        )
        self.word_boundary_output_symbol = str(
            self.word_boundary_config.get("g2p_output_symbol", " ")
        )
        self.word_boundary_duration_phone = str(
            self.word_boundary_config.get("duration_phone", "<sil>")
        )
        self.word_boundary_presence_threshold_frames = float(
            self.word_boundary_config.get("presence_threshold_frames", 0.5)
        )
        if self.word_boundary_enabled:
            if self.word_boundary_duration_phone not in self.phone_to_id:
                raise ValueError(
                    "Bundle word-boundary duration phone is absent from the phone "
                    f"vocabulary: {self.word_boundary_duration_phone!r}"
                )
            if self.word_boundary_presence_threshold_frames <= 0.0:
                raise ValueError(
                    "Bundle word-boundary presence threshold must be positive"
                )
        punctuation_silence = controls.get("punctuation_silence", {})
        if not isinstance(punctuation_silence, Mapping):
            punctuation_silence = {}
        self.punctuation_silence_target = str(
            punctuation_silence.get("target")
            or self.g2p_config.get("punctuation_silence_target")
            or "merge_into_punctuation"
        ).strip().lower()
        (
            self.punctuation_pause_floor_table_ms,
            self.punctuation_pause_floor_table_sha256,
        ) = _load_punctuation_pause_floor_table(punctuation_silence)
        fixed_shapes = dict(self.export_status.get("fixed_shapes", {}) or controls.get("fixed_shapes", {}))
        self.g2p_text_tokens = int(fixed_shapes.get("g2p_text_tokens") or self.g2p_config.get("fixed_text_tokens") or 512)
        self.g2p_segment_config = _g2p_segment_config_for_fixed_text(
            self.g2p_config,
            tokenizer=self.g2p_tokenizer,
            fixed_text_tokens=self.g2p_text_tokens,
        )
        self.phone_frames = int(fixed_shapes.get("phone_frames") or 256)
        self.latent_frames = int(fixed_shapes.get("latent_frames") or 256)
        self.target_bucket_frames = _target_bucket_frames(controls, self.latent_frames)
        self.latent_dim = int(self.manifest.audio.latent_dim)
        self.reference_config = dict(controls.get("reference_packs", {}) or {})
        self.reference_schema_version = int(
            self.reference_config.get("schema_version") or 0
        )
        self.reference_style_dim = int(
            self.reference_config["style_dim"]
            if "style_dim" in self.reference_config
            else 128
        )
        self.reference_prosody_dim = int(
            self.reference_config["prosody_dim"]
            if "prosody_dim" in self.reference_config
            else 32
        )
        self.reference_identity_dim = int(
            self.reference_config.get("identity_dim") or 512
        )
        self.reference_baseline_dim = int(
            self.reference_config.get("prosody_baseline_dim")
            or self.reference_prosody_dim
        )
        self.reference_delta_dim = int(
            self.reference_config.get("prosody_delta_dim")
            or self.reference_prosody_dim
        )
        self.reference_fallback_weight = float(
            self.reference_config.get("fallback_weight") or 0.25
        )
        routing = self.reference_config.get("affect_routing")
        self.reference_affect_routing = (
            dict(routing) if isinstance(routing, Mapping) else {}
        )
        self.prefix_config = dict(controls.get("prefix_conditioning", {}) or {})
        self.prefix_max_frames = max(1, int(self.prefix_config.get("max_frames") or 64))
        self._voice_pack_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._reference_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._sessions: dict[str, Any] = dict(sessions or {})
        self._g2p_cache: OrderedDict[tuple[str, str, str, str], dict[str, Any]] = OrderedDict()
        self._g2p_word_cache: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        self._duration_cache: OrderedDict[
            tuple[Any, ...],
            tuple[
                tuple[float, ...],
                tuple[float, ...] | None,
                tuple[tuple[float, float, float], ...] | None,
            ],
        ] = OrderedDict()
        self._g2p_lock = threading.RLock()
        self._duration_lock = threading.RLock()

    def synthesize(
        self,
        request: Any,
        *,
        text: str | None,
        language: str,
    ) -> LiteRTSynthesisResult:
        if request.speed <= 0.0:
            raise ValueError("speed must be positive")
        phone_result = self._resolve_phones(request, text=text, language=language)
        phones = phone_result["phones"]
        if len(phones) > self.phone_frames:
            raise ValueError(
                f"Phone sequence has {len(phones)} phones, but this LiteRT bundle "
                f"supports at most {self.phone_frames}. Split the text into shorter chunks."
            )
        unknown = sorted({phone for phone in phones if phone not in self.phone_to_id})
        if unknown:
            preview = ", ".join(unknown[:20])
            raise ValueError(f"Phone sequence contains symbols outside the Scylla's Band phone vocabulary: {preview}")

        voice_index = self._lookup(self.voice_to_id, request.voice_id, "voice")
        language_index = self._lookup(self.language_to_id, language, "language")
        affect_features = self._resolve_affect(request)
        affect_guidance_scale = self._resolve_affect_guidance_scale(request, affect_features)
        guidance_terms = (
            []
            if affect_features.enabled
            else self._parse_emotion_guidance(getattr(request, "emotion_guidance", None))
        )
        emotion_name, emotion_index = self._resolve_emotion(request, guidance_terms)
        reference_features = self._reference_features(
            voice_id=str(request.voice_id),
            language=language,
            emotion=emotion_name,
            affect=affect_features,
        )
        phone_ids = [int(self.phone_to_id[phone]) for phone in phones]
        boundary_before_id, boundary_after_id = self._boundary_ids(request, phone_result)
        duration_scale = 1.0 / float(request.speed)
        (
            durations,
            punctuation_floor_audit,
            pause_presence_audit,
            duration_hierarchy_audit,
        ) = self._predict_durations(
            phone_ids,
            phones=phones,
            punctuation_events=list(phone_result.get("g2p_punctuation_events") or []),
            word_boundary_candidate_indices=set(
                int(item)
                for item in phone_result.get(
                    "g2p_word_boundary_candidate_indices", ()
                )
            ),
            language=language,
            voice_index=voice_index,
            language_index=language_index,
            emotion_index=emotion_index,
            affect_features=affect_features,
            affect_guidance_scale=affect_guidance_scale,
            guidance_terms=guidance_terms,
            reference_features=reference_features,
            guidance_null_reference=bool(getattr(request, "guidance_null_reference", True)),
            boundary_before_id=boundary_before_id,
            boundary_after_id=boundary_after_id,
            duration_scale=duration_scale,
            request=request,
        )
        phone_result["punctuation_duration_floors"] = punctuation_floor_audit
        phone_result["duration_pause_presence"] = pause_presence_audit
        phone_result["duration_hierarchy_sampling"] = duration_hierarchy_audit
        word_boundary_candidates = set(
            int(item)
            for item in phone_result.get(
                "g2p_word_boundary_candidate_indices", ()
            )
        )
        phone_result["g2p_word_boundary_durations"] = [
            {"phone_index": index, "frames": int(durations[index])}
            for index in sorted(word_boundary_candidates)
            if 0 <= index < len(durations)
        ]
        latent_length = int(sum(durations))
        if latent_length <= 0:
            raise ValueError("Predicted zero latent frames; cannot synthesize")
        fixed_latent_frames = self._latent_bucket_frames(latent_length)
        if latent_length > fixed_latent_frames:
            raise ValueError(
                f"Predicted {latent_length} latent frames, but this LiteRT bundle "
                f"supports at most {fixed_latent_frames}. Increase --speed or split the text."
            )
        vector_component = self._bucket_component_name("vector_estimator", fixed_latent_frames)
        vocoder_component = self._bucket_component_name("vocoder", fixed_latent_frames)

        expanded_phone_ids = _expand_phone_lists_to_length(phone_ids, durations, latent_length)
        vector_timing = _expanded_vector_timing_features(
            phones=phones,
            durations=durations,
            word_boundary_candidate_indices=word_boundary_candidates,
            punctuation_events=list(
                phone_result.get("g2p_punctuation_events") or []
            ),
            config=self.vector_timing_config,
        )
        if vector_timing is not None:
            phone_result["vector_timing_unrepresented_zero_frame_modifier_events"] = (
                vector_timing.unrepresented_zero_frame_modifier_events
            )
        vector_context_phone_ids = [
            int(phone_id)
            for index, (phone_id, duration) in enumerate(
                zip(phone_ids, durations)
            )
            if index not in word_boundary_candidates or int(duration) > 0
        ]
        span_context_hidden = self._span_context_hidden(
            request,
            target_phone_ids=vector_context_phone_ids,
            language=language,
            component_name=vector_component,
        )
        latents = self._sample_latents(
            expanded_phone_ids,
            latent_length=latent_length,
            fixed_latent_frames=fixed_latent_frames,
            vector_component=vector_component,
            span_context_hidden=span_context_hidden,
            vector_timing=vector_timing,
            voice_index=voice_index,
            language_index=language_index,
            emotion_index=emotion_index,
            affect_features=affect_features,
            affect_guidance_scale=affect_guidance_scale,
            guidance_terms=guidance_terms,
            reference_features=reference_features,
            guidance_null_reference=bool(getattr(request, "guidance_null_reference", True)),
            emotion_embed_scale=float(getattr(request, "emotion_embed_scale", 1.0)),
            prefix_latents=getattr(request, "prefix_latents", None),
            boundary_before_id=boundary_before_id,
            boundary_after_id=boundary_after_id,
            steps=int(request.steps),
            sampler=str(request.sampler),
            seed=request.seed,
            noise_scale=float(request.temperature),
        )
        affect_guidance_active = affect_features.enabled and affect_guidance_scale != 1.0
        guidance_branch_count = 2 if affect_guidance_active else (1 + len(guidance_terms) if guidance_terms else 1)
        guidance_vector_batched = bool(
            guidance_terms
            and not affect_guidance_active
            and self._supports_batched_guidance(vector_component, guidance_branch_count)
        )
        sampler_evaluations = int(request.steps) * (2 if str(request.sampler) == "heun" else 1)
        audio = self._run_vocoder(
            latents,
            component_name=vocoder_component,
            latent_length=latent_length,
            voice_index=voice_index,
            language_index=language_index,
            emotion_index=emotion_index,
            reference_features=reference_features,
        )
        trim_samples = max(1, (latent_length * 2 - 1) * int(self.manifest.audio.hop_length))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)[:trim_samples]
        audio = np.clip(audio, -1.0, 1.0)
        metadata = {
            "backend": self.backend_name,
            "bundle_dir": str(self.bundle_dir),
            "voice_id": request.voice_id,
            "voice_index": voice_index,
            "language": language,
            "language_index": language_index,
            "emotion": emotion_name,
            "emotion_id": emotion_index,
            "emotion_guidance": _emotion_guidance_metadata(guidance_terms),
            "emotion_guidance_null_weight": _emotion_guidance_null_weight(guidance_terms),
            "affect_guidance_scale": affect_guidance_scale,
            "affect_guidance_null_weight": 1.0 - affect_guidance_scale,
            "affect_guidance_reference_retained": bool(affect_features.enabled),
            "guidance_null_reference": bool(getattr(request, "guidance_null_reference", True)),
            "emotion_embed_scale": float(getattr(request, "emotion_embed_scale", 1.0)),
            "reference_key": reference_features.key,
            "reference_pack_path": reference_features.path,
            "reference_mask": float(reference_features.mask[0]),
            "native_reference_mask": float(reference_features.native_mask),
            "fallback_reference_mask": float(reference_features.fallback_mask),
            "prefix_frames_used": int(self._prefix_inputs(getattr(request, "prefix_latents", None))[1].sum()),
            "boundary_before_id": boundary_before_id,
            "boundary_after_id": boundary_after_id,
            "text": text or "",
            "phone_source": phone_result["phone_source"],
            "phones": phones,
            "phone_ids": phone_ids,
            "predicted_durations": durations,
            "predicted_latent_frames": latent_length,
            "predicted_mel_frames": latent_length * 2,
            "trimmed_audio_samples": int(audio.size),
            "fixed_latent_frames": fixed_latent_frames,
            "target_bucket_latent_frames": fixed_latent_frames,
            "vector_component": vector_component,
            "vocoder_component": vocoder_component,
            "steps": int(request.steps),
            "sampler": str(request.sampler),
            "vector_guidance_branches": guidance_branch_count,
            "vector_guidance_batched": guidance_vector_batched,
            "vector_model_invocations": sampler_evaluations * (
                1 if guidance_vector_batched else guidance_branch_count
            ),
            "speed": float(request.speed),
            "duration_scale": duration_scale,
            "min_sentence_pause_ms": float(getattr(request, "min_sentence_pause_ms", 0.0)),
            "min_clause_pause_ms": float(getattr(request, "min_clause_pause_ms", 0.0)),
            "punctuation_pause_floor_table_sha256": self.punctuation_pause_floor_table_sha256,
            "seed": request.seed,
            "noise_scale": float(request.temperature),
        }
        metadata.update(self._affect_metadata(affect_features))
        metadata.update({key: value for key, value in phone_result.items() if key not in {"phones", "phone_source"}})
        return LiteRTSynthesisResult(
            audio=audio,
            metadata=metadata,
            latents=np.asarray(latents[:, :, :latent_length], dtype=np.float32).copy(),
        )

    def session_status(self) -> dict[str, Any]:
        """Report process-local component sessions without creating new ones."""
        return {
            "backend": self.backend_name,
            "session_count": len(self._sessions),
            "sessions": sorted(self._sessions),
        }

    def estimate_latent_frames(
        self,
        request: Any,
        *,
        text: str | None,
        language: str,
    ) -> dict[str, Any]:
        if request.speed <= 0.0:
            raise ValueError("speed must be positive")
        phone_result = self._resolve_phones(request, text=text, language=language)
        phones = phone_result["phones"]
        if len(phones) > self.phone_frames:
            raise ValueError(
                f"Phone sequence has {len(phones)} phones, but this LiteRT bundle "
                f"supports at most {self.phone_frames}. Split the text into shorter chunks."
            )
        unknown = sorted({phone for phone in phones if phone not in self.phone_to_id})
        if unknown:
            preview = ", ".join(unknown[:20])
            raise ValueError(f"Phone sequence contains symbols outside the Scylla's Band phone vocabulary: {preview}")

        voice_index = self._lookup(self.voice_to_id, request.voice_id, "voice")
        language_index = self._lookup(self.language_to_id, language, "language")
        affect_features = self._resolve_affect(request)
        affect_guidance_scale = self._resolve_affect_guidance_scale(request, affect_features)
        guidance_terms = (
            []
            if affect_features.enabled
            else self._parse_emotion_guidance(getattr(request, "emotion_guidance", None))
        )
        emotion_name, emotion_index = self._resolve_emotion(request, guidance_terms)
        reference_features = self._reference_features(
            voice_id=str(request.voice_id),
            language=language,
            emotion=emotion_name,
            affect=affect_features,
        )
        phone_ids = [int(self.phone_to_id[phone]) for phone in phones]
        boundary_before_id, boundary_after_id = self._boundary_ids(request, phone_result)
        duration_scale = 1.0 / float(request.speed)
        (
            durations,
            punctuation_floor_audit,
            pause_presence_audit,
            duration_hierarchy_audit,
        ) = self._predict_durations(
            phone_ids,
            phones=phones,
            punctuation_events=list(phone_result.get("g2p_punctuation_events") or []),
            word_boundary_candidate_indices=set(
                int(item)
                for item in phone_result.get(
                    "g2p_word_boundary_candidate_indices", ()
                )
            ),
            language=language,
            voice_index=voice_index,
            language_index=language_index,
            emotion_index=emotion_index,
            affect_features=affect_features,
            affect_guidance_scale=affect_guidance_scale,
            guidance_terms=guidance_terms,
            reference_features=reference_features,
            guidance_null_reference=bool(getattr(request, "guidance_null_reference", True)),
            boundary_before_id=boundary_before_id,
            boundary_after_id=boundary_after_id,
            duration_scale=duration_scale,
            request=request,
        )
        phone_result["punctuation_duration_floors"] = punctuation_floor_audit
        phone_result["duration_pause_presence"] = pause_presence_audit
        phone_result["duration_hierarchy_sampling"] = duration_hierarchy_audit
        word_boundary_candidates = set(
            int(item)
            for item in phone_result.get(
                "g2p_word_boundary_candidate_indices", ()
            )
        )
        phone_result["g2p_word_boundary_durations"] = [
            {"phone_index": index, "frames": int(durations[index])}
            for index in sorted(word_boundary_candidates)
            if 0 <= index < len(durations)
        ]
        latent_length = int(sum(durations))
        fixed_latent_frames = self._latent_bucket_frames(latent_length) if latent_length > 0 else self.target_bucket_frames[0]
        metadata = {
            "backend": self.backend_name,
            "bundle_dir": str(self.bundle_dir),
            "voice_id": request.voice_id,
            "voice_index": voice_index,
            "language": language,
            "language_index": language_index,
            "emotion": emotion_name,
            "emotion_id": emotion_index,
            "emotion_guidance": _emotion_guidance_metadata(guidance_terms),
            "emotion_guidance_null_weight": _emotion_guidance_null_weight(guidance_terms),
            "affect_guidance_scale": affect_guidance_scale,
            "affect_guidance_null_weight": 1.0 - affect_guidance_scale,
            "affect_guidance_reference_retained": bool(affect_features.enabled),
            "guidance_null_reference": bool(getattr(request, "guidance_null_reference", True)),
            "emotion_embed_scale": float(getattr(request, "emotion_embed_scale", 1.0)),
            "reference_key": reference_features.key,
            "reference_pack_path": reference_features.path,
            "reference_mask": float(reference_features.mask[0]),
            "native_reference_mask": float(reference_features.native_mask),
            "fallback_reference_mask": float(reference_features.fallback_mask),
            "boundary_before_id": boundary_before_id,
            "boundary_after_id": boundary_after_id,
            "text": text or "",
            "phone_source": phone_result["phone_source"],
            "phones": phones,
            "phone_ids": phone_ids,
            "predicted_durations": durations,
            "predicted_latent_frames": latent_length,
            "predicted_mel_frames": latent_length * 2,
            "fixed_latent_frames": fixed_latent_frames,
            "target_bucket_latent_frames": fixed_latent_frames,
            "speed": float(request.speed),
            "duration_scale": duration_scale,
            "min_sentence_pause_ms": float(getattr(request, "min_sentence_pause_ms", 0.0)),
            "min_clause_pause_ms": float(getattr(request, "min_clause_pause_ms", 0.0)),
            "punctuation_pause_floor_table_sha256": self.punctuation_pause_floor_table_sha256,
        }
        metadata.update(self._affect_metadata(affect_features))
        metadata.update({key: value for key, value in phone_result.items() if key not in {"phones", "phone_source"}})
        return metadata

    def phonemize(
        self,
        text: str,
        *,
        language: str,
        boundary_before: str | None = None,
        boundary_after: str | None = None,
    ) -> dict[str, Any]:
        cache_key = (
            str(text or ""),
            self._g2p_language(language),
            _normalize_boundary_before(boundary_before),
            _normalize_boundary_after(boundary_after),
        )
        with self._g2p_lock:
            cached = self._g2p_cache.get(cache_key)
            if cached is not None:
                self._g2p_cache.move_to_end(cache_key)
                return copy.deepcopy(cached)
            result = self._phonemize_uncached(
                text,
                language=language,
                boundary_before=boundary_before,
                boundary_after=boundary_after,
            )
            self._g2p_cache[cache_key] = copy.deepcopy(result)
            self._g2p_cache.move_to_end(cache_key)
            while len(self._g2p_cache) > _FRONTEND_CACHE_SIZE:
                self._g2p_cache.popitem(last=False)
            return result

    def _phonemize_uncached(
        self,
        text: str,
        *,
        language: str,
        boundary_before: str | None = None,
        boundary_after: str | None = None,
    ) -> dict[str, Any]:
        model_language = self._g2p_language(language)
        segments = _g2p_text_segments(text, config=self.g2p_segment_config)
        pause_phone = "<sil>" if "<sil>" in self.phone_to_id else None
        before = _normalize_boundary_before(boundary_before)
        after = _normalize_boundary_after(boundary_after)
        phones: list[str] = []
        predictions: list[dict[str, Any]] = []
        boundary_tokens: list[str] = []
        source_boundary_tokens: list[str] = []
        punctuation_events: list[dict[str, Any]] = []
        word_boundary_candidate_indices: list[int] = []
        pronunciation_overrides: list[dict[str, Any]] = []
        terminal_tail_repairs: list[dict[str, Any]] = []
        leading_context_phone = _context_phone_for_boundary(before, self.phone_to_id, self.g2p_config)
        if leading_context_phone is not None:
            phones.append(leading_context_phone)
            boundary_tokens.append(leading_context_phone)
        if pause_phone is not None and _insert_leading_silence(before):
            phones.append(pause_phone)
        for index, segment in enumerate(segments):
            prediction = self._predict_g2p_segment(segment, language=model_language)
            predictions.append(prediction)
            pronunciation_overrides.extend(
                dict(item) for item in prediction.get("pronunciation_overrides", [])
            )
            terminal_tail_repairs.extend(
                dict(item) for item in prediction.get("terminal_tail_repairs", [])
            )
            predicted_phones = [
                str(phone) for phone in prediction.get("phones", ())
            ]
            for predicted_index, phone in enumerate(predicted_phones):
                if phone == _WORD_BOUNDARY_MARKER:
                    if (
                        self.word_boundary_enabled
                        and _is_internal_word_boundary_marker(
                            predicted_phones, predicted_index
                        )
                    ):
                        candidate_index = len(phones)
                        phones.append(self.word_boundary_duration_phone)
                        word_boundary_candidate_indices.append(candidate_index)
                    continue
                if phone not in _SILENCE_PHONES:
                    phones.append(phone)
            source_boundary_phone = _boundary_phone_for_segment(segment, self.phone_to_id)
            boundary_phone = source_boundary_phone
            has_following_segment = index < len(segments) - 1
            remap_at_position = (
                self.g2p_punctuation_token_remap_scope == "all_boundaries"
                or has_following_segment
            )
            if (
                source_boundary_phone is not None
                and remap_at_position
                and source_boundary_phone in self.g2p_punctuation_token_remap
            ):
                remapped = self.g2p_punctuation_token_remap[source_boundary_phone]
                boundary_phone = remapped if remapped in self.phone_to_id else None
            if boundary_phone is not None:
                phone_index = len(phones)
                phones.append(boundary_phone)
                boundary_tokens.append(boundary_phone)
                if source_boundary_phone is not None:
                    source_boundary_tokens.append(source_boundary_phone)
                    punctuation_events.append(
                        {
                            "segment_index": index,
                            "phone_index": phone_index,
                            "source_phone": source_boundary_phone,
                            "emitted_phone": boundary_phone,
                            "remapped": source_boundary_phone != boundary_phone,
                            "has_following_segment": has_following_segment,
                        }
                    )
                if (
                    self.punctuation_silence_target == "explicit_silence"
                    and pause_phone is not None
                    and boundary_phone != pause_phone
                ):
                    phones.append(pause_phone)
            elif has_following_segment:
                internal_phone = _continuation_phone(
                    "clause_continue",
                    self.phone_to_id,
                    self.g2p_config,
                    fallback_pause=pause_phone,
                )
                if internal_phone is not None:
                    phones.append(internal_phone)
                    boundary_tokens.append(internal_phone)
                    if self.punctuation_silence_target == "explicit_silence" and pause_phone is not None:
                        phones.append(pause_phone)
        trailing_phone = _trailing_boundary_phone(after, self.phone_to_id, self.g2p_config)
        if trailing_phone is not None and not _ends_in_boundary_phone(phones):
            phones.append(trailing_phone)
            boundary_tokens.append(trailing_phone)
        if not phones:
            raise ValueError("G2P produced no usable Scylla's Band phone symbols")
        confidences = [float(item.get("confidence", 0.0)) for item in predictions]
        result = {
            "phones": phones,
            "phone_source": "g2p",
            "g2p_prediction_text": " | ".join(str(item.get("prediction_text", "")) for item in predictions),
            "g2p_prediction_confidence": min(confidences) if confidences else 0.0,
            "g2p_segment_confidences": confidences,
            "g2p_segments": segments,
            "g2p_inserted_pause_tokens": pause_phone is not None,
            "g2p_boundary_tokens": boundary_tokens,
            "g2p_source_boundary_tokens": source_boundary_tokens,
            "g2p_punctuation_events": punctuation_events,
            "g2p_word_boundary_candidates_enabled": self.word_boundary_enabled,
            "g2p_word_boundary_candidate_indices": (
                word_boundary_candidate_indices
            ),
            "punctuation_silence_target": self.punctuation_silence_target,
            "boundary_before": before,
            "boundary_after": after,
            "context_phone_tokens_enabled": _emit_context_phone_tokens(self.g2p_config),
        }
        if pronunciation_overrides:
            result["pronunciation_overrides"] = pronunciation_overrides
        if terminal_tail_repairs:
            result["g2p_terminal_tail_repairs"] = terminal_tail_repairs
        return result

    def _resolve_phones(self, request: Any, *, text: str | None, language: str) -> dict[str, Any]:
        explicit = request.explicit_phones
        if isinstance(explicit, str):
            phones = [item for item in explicit.split() if item]
        elif explicit:
            phones = [str(item) for item in explicit if str(item)]
        else:
            phones = []
        if phones:
            return {"phones": phones, "phone_source": "explicit"}
        if not text or not text.strip():
            raise ValueError("Raw text synthesis requires text or explicit_phones")
        result = self.phonemize(
            text,
            language=language,
            boundary_before=getattr(request, "boundary_before", None),
            boundary_after=getattr(request, "boundary_after", None),
        )
        result["g2p_input_text"] = text
        return result

    def _predict_g2p_segment(self, text: str, *, language: str) -> dict[str, Any]:
        prediction = self._predict_g2p_raw(text, language=language)
        prediction = self._apply_g2p_pronunciation_overrides(prediction, text=text, language=language)
        return self._repair_g2p_terminal_tail(prediction, text=text, language=language)

    def _predict_g2p_raw(self, text: str, *, language: str) -> dict[str, Any]:
        encoded = self._encode_g2p_text(text, language=language)
        logits = self._session("g2p").invoke({0: encoded[np.newaxis, :]})
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float32)[0]
        return self._decode_g2p_logits(logits)

    def _repair_g2p_terminal_tail(
        self,
        prediction: dict[str, Any],
        *,
        text: str,
        language: str,
    ) -> dict[str, Any]:
        """Trim short CTC tail hallucinations after a correctly decoded final word."""
        word = _terminal_g2p_word(text)
        if not word:
            return prediction
        cache_key = (str(language), word)
        isolated = self._g2p_word_cache.get(cache_key)
        if isolated is None:
            isolated_prediction = self._predict_g2p_raw(word, language=language)
            isolated_prediction = self._apply_g2p_pronunciation_overrides(
                isolated_prediction,
                text=word,
                language=language,
            )
            isolated = tuple(str(phone) for phone in isolated_prediction.get("phones", []))
            self._g2p_word_cache[cache_key] = isolated
            self._g2p_word_cache.move_to_end(cache_key)
            while len(self._g2p_word_cache) > _FRONTEND_CACHE_SIZE:
                self._g2p_word_cache.popitem(last=False)
        else:
            self._g2p_word_cache.move_to_end(cache_key)
        repaired, removed = _trim_g2p_terminal_artifacts(
            [str(phone) for phone in prediction.get("phones", [])],
            list(isolated),
        )
        if not removed:
            return prediction
        updated = dict(prediction)
        updated["phones"] = repaired
        updated["prediction_symbols"] = repaired
        updated["prediction_text"] = " ".join(repaired).strip()
        updated["terminal_tail_repairs"] = [
            {
                "word": word,
                "phones": list(isolated),
                "removed": removed,
            }
        ]
        return updated

    def _encode_g2p_text(self, text: str, *, language: str) -> np.ndarray:
        token_to_id = {str(key): int(value) for key, value in self.g2p_tokenizer["text_symbols"].items()}
        language_token = f"<{language}>"
        if language_token not in token_to_id:
            raise ValueError(f"G2P language {language!r} is not supported by this bundle")
        char_repeats = max(1, int(self.g2p_tokenizer.get("char_repeats", 1)))
        lowercase = bool(self.g2p_tokenizer.get("lowercase", True))
        work = str(text or "")
        if lowercase:
            work = work.lower()
        sequence = [int(token_to_id[language_token])]
        emitted_chars = 0
        for char in work:
            token_id = token_to_id.get(char)
            if token_id is None:
                continue
            sequence.extend([int(token_id)] * char_repeats)
            emitted_chars += 1
        sequence.append(int(token_to_id.get("<end>", self.g2p_tokenizer.get("text_pad_index", 0))))
        if emitted_chars <= 0:
            raise ValueError("G2P text has no characters supported by this bundle")
        if len(sequence) > self.g2p_text_tokens:
            raise ValueError(
                f"G2P input encodes to {len(sequence)} tokens, but this LiteRT bundle "
                f"supports at most {self.g2p_text_tokens}. Split the text into shorter chunks."
            )
        out = np.full((self.g2p_text_tokens,), int(self.g2p_tokenizer.get("text_pad_index", 0)), dtype=np.int64)
        out[: len(sequence)] = np.asarray(sequence, dtype=np.int64)
        return out

    def _decode_g2p_logits(self, logits: np.ndarray) -> dict[str, Any]:
        phoneme_symbols = {
            int(key): str(value)
            for key, value in dict(self.g2p_tokenizer["phoneme_symbols"]).items()
        }
        pad_index = int(self.g2p_tokenizer.get("phoneme_pad_index", 0))
        end_index = int(self.g2p_tokenizer.get("phoneme_end_index", 0))
        probs = _softmax(np.asarray(logits, dtype=np.float32), axis=-1)
        argmax = np.argmax(probs, axis=-1).astype(np.int64)
        phones: list[str] = []
        emitted_probs: list[float] = []
        previous: int | None = None
        for frame, token_id_raw in enumerate(argmax.tolist()):
            token_id = int(token_id_raw)
            if previous == token_id:
                continue
            previous = token_id
            if token_id == pad_index:
                continue
            if token_id == end_index:
                break
            symbol = phoneme_symbols.get(token_id)
            if (
                self.word_boundary_enabled
                and symbol == self.word_boundary_output_symbol
            ):
                phones.append(_WORD_BOUNDARY_MARKER)
                emitted_probs.append(float(probs[frame, token_id]))
                continue
            if _skip_g2p_output_symbol(symbol):
                continue
            phones.append(str(symbol))
            emitted_probs.append(float(probs[frame, token_id]))
        confidence = _prob_product(emitted_probs)
        return {
            "phones": phones,
            "prediction_text": " ".join(phones).strip(),
            "prediction_symbols": phones,
            "confidence": confidence,
        }

    def _apply_g2p_pronunciation_overrides(
        self,
        prediction: dict[str, Any],
        *,
        text: str,
        language: str,
    ) -> dict[str, Any]:
        overrides = _pronunciation_overrides_for_language(
            self.g2p_pronunciation_overrides,
            language=language,
        )
        if not overrides:
            return prediction
        words = _words_for_pronunciation_overrides(text)
        if not words:
            return prediction
        phones = [str(phone) for phone in prediction.get("phones", [])]
        applied: list[dict[str, Any]] = []
        for word, spec in sorted(overrides.items()):
            if word not in words:
                continue
            target = _override_phone_list(spec, key="phones")
            source = _override_phone_list(spec, key="replace")
            if not target or not source:
                continue
            phones, count = _replace_phone_subsequence(phones, source, target)
            if count <= 0:
                continue
            applied.append(
                {
                    "word": word,
                    "replace": source,
                    "phones": target,
                    "count": count,
                }
            )
        if not applied:
            return prediction
        updated = dict(prediction)
        updated["phones"] = phones
        updated["prediction_symbols"] = phones
        updated["prediction_text"] = " ".join(phones).strip()
        updated["pronunciation_overrides"] = applied
        return updated

    def _predict_durations(
        self,
        phone_ids: list[int],
        *,
        phones: list[str],
        punctuation_events: list[Mapping[str, Any]],
        word_boundary_candidate_indices: set[int],
        language: str,
        voice_index: int,
        language_index: int,
        emotion_index: int,
        affect_features: _AffectFeatures,
        affect_guidance_scale: float,
        guidance_terms: list[_EmotionGuidanceTerm],
        reference_features: _ReferenceFeatures,
        guidance_null_reference: bool,
        boundary_before_id: int,
        boundary_after_id: int,
        duration_scale: float,
        request: Any,
    ) -> tuple[
        list[int],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        if affect_features.enabled and affect_guidance_scale != 1.0:
            (
                null_values,
                null_presence_logits,
                null_quantiles,
            ) = self._predict_duration_values(
                phone_ids,
                voice_index=voice_index,
                language_index=language_index,
                emotion_index=emotion_index,
                affect_features=affect_features,
                emotion_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                boundary_before_id=boundary_before_id,
                boundary_after_id=boundary_after_id,
                affect_condition_scale=0.0,
            )
            (
                conditioned_values,
                conditioned_presence_logits,
                conditioned_quantiles,
            ) = self._predict_duration_values(
                phone_ids,
                voice_index=voice_index,
                language_index=language_index,
                emotion_index=emotion_index,
                affect_features=affect_features,
                emotion_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                boundary_before_id=boundary_before_id,
                boundary_after_id=boundary_after_id,
                affect_condition_scale=1.0,
            )
            null_array = np.asarray(null_values, dtype=np.float32)
            conditioned_array = np.asarray(conditioned_values, dtype=np.float32)
            values = np.maximum(
                null_array + float(affect_guidance_scale) * (conditioned_array - null_array),
                0.0,
            ).tolist()
            pause_presence_logits = _blend_optional_numpy_logits(
                null_presence_logits,
                conditioned_presence_logits,
                null_weight=1.0 - float(affect_guidance_scale),
                term_weight=float(affect_guidance_scale),
            )
            duration_quantiles = _blend_optional_numpy_quantiles(
                null_quantiles,
                conditioned_quantiles,
                null_weight=1.0 - float(affect_guidance_scale),
                term_weight=float(affect_guidance_scale),
            )
        elif guidance_terms:
            null_reference_scale = 0.0 if guidance_null_reference else 1.0
            (
                values,
                pause_presence_logits,
                duration_quantiles,
            ) = self._predict_duration_values(
                phone_ids,
                voice_index=voice_index,
                language_index=language_index,
                emotion_index=emotion_index,
                affect_features=affect_features,
                emotion_condition_scale=0.0,
                reference_features=reference_features,
                reference_condition_scale=null_reference_scale,
                boundary_before_id=boundary_before_id,
                boundary_after_id=boundary_after_id,
            )
            blended = _emotion_guidance_null_weight(guidance_terms) * np.asarray(values, dtype=np.float32)
            blended_presence = (
                _emotion_guidance_null_weight(guidance_terms)
                * np.asarray(pause_presence_logits, dtype=np.float32)
                if pause_presence_logits is not None
                else None
            )
            blended_quantiles = (
                _emotion_guidance_null_weight(guidance_terms)
                * np.asarray(duration_quantiles, dtype=np.float32)
                if duration_quantiles is not None
                else None
            )
            for term in guidance_terms:
                (
                    term_values,
                    term_presence_logits,
                    term_quantiles,
                ) = self._predict_duration_values(
                    phone_ids,
                    voice_index=voice_index,
                    language_index=language_index,
                    emotion_index=int(term.emotion_id),
                    affect_features=affect_features,
                    emotion_condition_scale=1.0,
                    reference_features=reference_features,
                    reference_condition_scale=1.0,
                    boundary_before_id=boundary_before_id,
                    boundary_after_id=boundary_after_id,
                )
                blended = blended + float(term.scale) * np.asarray(term_values, dtype=np.float32)
                if blended_presence is not None:
                    if term_presence_logits is None:
                        raise RuntimeError(
                            "Duration guidance branches disagree on "
                            "pause-presence output"
                        )
                    blended_presence = (
                        blended_presence
                        + float(term.scale)
                        * np.asarray(term_presence_logits, dtype=np.float32)
                    )
                elif term_presence_logits is not None:
                    raise RuntimeError(
                        "Duration guidance branches disagree on "
                        "pause-presence output"
                    )
                if blended_quantiles is not None:
                    if term_quantiles is None:
                        raise RuntimeError(
                            "Duration guidance branches disagree on quantile output"
                        )
                    blended_quantiles = (
                        blended_quantiles
                        + float(term.scale)
                        * np.asarray(term_quantiles, dtype=np.float32)
                    )
                elif term_quantiles is not None:
                    raise RuntimeError(
                        "Duration guidance branches disagree on quantile output"
                    )
            values = np.maximum(blended, 0.0).tolist()
            pause_presence_logits = (
                blended_presence.tolist()
                if blended_presence is not None
                else None
            )
            duration_quantiles = (
                np.sort(np.maximum(blended_quantiles, 0.0), axis=-1).tolist()
                if blended_quantiles is not None
                else None
            )
        else:
            (
                values,
                pause_presence_logits,
                duration_quantiles,
            ) = self._predict_duration_values(
                phone_ids,
                voice_index=voice_index,
                language_index=language_index,
                emotion_index=emotion_index,
                affect_features=affect_features,
                emotion_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                boundary_before_id=boundary_before_id,
                boundary_after_id=boundary_after_id,
            )
        durations = _frames_to_durations(
            values,
            phones=phones,
            scale=duration_scale,
            min_phone_frames=1,
            zero_duration_punctuation=(
                self.punctuation_silence_target == "explicit_silence"
            ),
            word_boundary_candidate_indices=(
                set()
                if self.duration_pause_presence_enabled
                else word_boundary_candidate_indices
            ),
            word_boundary_presence_threshold_frames=(
                self.word_boundary_presence_threshold_frames
            ),
        )
        requested_hierarchy_mode = str(
            getattr(request, "duration_hierarchy_mode", "default") or "default"
        ).strip().lower()
        if requested_hierarchy_mode in {"", "auto", "default"}:
            hierarchy_mode = str(
                self.duration_hierarchy_config.get("default_mode") or "p50"
            )
        else:
            hierarchy_mode = requested_hierarchy_mode
        if hierarchy_mode not in {"p50", "sampled"}:
            raise ValueError(
                "duration_hierarchy_mode must be default, sampled, or p50"
            )
        if hierarchy_mode == "sampled" and not self.duration_hierarchy_enabled:
            raise ValueError(
                "duration_hierarchy_mode=sampled requires a hierarchy-enabled bundle"
            )
        pause_presence_audit: list[dict[str, Any]] = []
        if hierarchy_mode == "sampled":
            if duration_quantiles is None:
                raise RuntimeError(
                    "Duration hierarchy contract is active but the graph returned no quantiles"
                )
            seed_contract = dict(self.duration_hierarchy_config.get("seed") or {})
            request_seed = getattr(request, "seed", None)
            hierarchy_seed = (
                int(request_seed)
                if request_seed is not None
                else int(seed_contract.get("missing_request_seed", 0))
            )
            (
                durations,
                pause_presence_audit,
                duration_hierarchy_audit,
            ) = _apply_duration_hierarchy_sampling(
                durations,
                duration_quantiles=duration_quantiles,
                pause_presence_logits=pause_presence_logits,
                phones=phones,
                punctuation_events=punctuation_events,
                word_boundary_candidate_indices=word_boundary_candidate_indices,
                language=language,
                voice=str(getattr(request, "voice_id", "") or ""),
                seed=hierarchy_seed,
                duration_scale=duration_scale,
                config=self.duration_hierarchy_config,
            )
        elif self.duration_pause_presence_enabled:
            if pause_presence_logits is None:
                raise RuntimeError(
                    "Duration pause-presence contract is active but the graph "
                    "returned no logits"
                )
            durations, pause_presence_audit = _apply_duration_pause_presence(
                durations,
                pause_presence_logits=pause_presence_logits,
                phones=phones,
                punctuation_events=punctuation_events,
                word_boundary_candidate_indices=word_boundary_candidate_indices,
                threshold_probability=(
                    self.duration_pause_presence_threshold_probability
                ),
            )
            duration_hierarchy_audit = {
                "schema_version": "scyllasband_duration_hierarchy_sampling_v1",
                "enabled": bool(self.duration_hierarchy_enabled),
                "mode": "p50",
            }
        else:
            duration_hierarchy_audit = {
                "schema_version": "scyllasband_duration_hierarchy_sampling_v1",
                "enabled": bool(self.duration_hierarchy_enabled),
                "mode": "p50",
            }
        semantic_phones: dict[int, str] = {}
        ellipsis_counts = iter(_ellipsis_dot_counts_from_request(request))
        punctuation_repeat_counts: dict[int, int] = {}
        for event in punctuation_events:
            try:
                phone_index = int(event.get("phone_index"))
            except (TypeError, ValueError):
                continue
            source_phone = str(event.get("source_phone") or "")
            if 0 <= phone_index < len(phones) and source_phone:
                semantic_phones[phone_index] = source_phone
                if source_phone == "<ellipsis>":
                    punctuation_repeat_counts[phone_index] = next(
                        ellipsis_counts, 3
                    )
        floor_audit: list[dict[str, Any]] = []
        floored = _apply_punctuation_duration_floors(
            durations,
            phones=phones,
            semantic_phones=semantic_phones,
            punctuation_repeat_counts=punctuation_repeat_counts,
            sentence_frames=_pause_ms_to_latent_frames(
                getattr(request, "min_sentence_pause_ms", 0.0),
                sample_rate=int(self.manifest.audio.sample_rate),
                latent_hop_length=int(self.manifest.audio.latent_hop_length),
            ),
            clause_frames=_pause_ms_to_latent_frames(
                getattr(request, "min_clause_pause_ms", 0.0),
                sample_rate=int(self.manifest.audio.sample_rate),
                latent_hop_length=int(self.manifest.audio.latent_hop_length),
            ),
            calibrated_frames={
                phone: _pause_ms_to_latent_frames(
                    self.punctuation_pause_floor_table_ms.get(
                        f"{language}|{phone}",
                        0.0,
                    ),
                    sample_rate=int(self.manifest.audio.sample_rate),
                    latent_hop_length=int(self.manifest.audio.latent_hop_length),
                )
                for phone in _ZERO_DURATION_PUNCTUATION_PHONES
            },
            floor_terminal=_request_has_following_chunk(request),
            audit=floor_audit,
        )
        gated_by_index = {
            int(item["phone_index"]): int(item["duration_after_gate"])
            for item in pause_presence_audit
        }
        for item in floor_audit:
            try:
                index = int(
                    item.get("target_phone_index", item.get("phone_index"))
                )
            except (TypeError, ValueError):
                continue
            if gated_by_index.get(index) == 0 and int(
                floored[index] if 0 <= index < len(floored) else 0
            ) > 0:
                item["overrides_pause_presence_absence"] = True
        return (
            floored,
            floor_audit,
            pause_presence_audit,
            duration_hierarchy_audit,
        )

    def _predict_duration_values(
        self,
        phone_ids: list[int],
        *,
        voice_index: int,
        language_index: int,
        emotion_index: int,
        affect_features: _AffectFeatures,
        emotion_condition_scale: float,
        reference_features: _ReferenceFeatures,
        reference_condition_scale: float,
        boundary_before_id: int,
        boundary_after_id: int,
        affect_condition_scale: float = 1.0,
    ) -> tuple[
        list[float],
        list[float] | None,
        list[list[float]] | None,
    ]:
        cache_key = (
            tuple(int(item) for item in phone_ids),
            int(voice_index),
            int(language_index),
            int(emotion_index),
            tuple(float(value) for value in affect_features.values.reshape(-1)),
            tuple(
                float(value) for value in affect_features.condition_mask.reshape(-1)
            ),
            float(affect_condition_scale),
            float(emotion_condition_scale),
            reference_features.key,
            reference_features.path,
            float(reference_features.mask[0]),
            float(reference_condition_scale),
            int(boundary_before_id),
            int(boundary_after_id),
        )
        with self._duration_lock:
            cached = self._duration_cache.get(cache_key)
            if cached is not None:
                self._duration_cache.move_to_end(cache_key)
                cached_values, cached_presence, cached_quantiles = cached
                return (
                    list(cached_values),
                    list(cached_presence)
                    if cached_presence is not None
                    else None,
                    [list(row) for row in cached_quantiles]
                    if cached_quantiles is not None
                    else None,
                )
            padded = np.zeros((1, self.phone_frames), dtype=np.int64)
            mask = np.zeros((1, self.phone_frames), dtype=np.bool_)
            padded[0, : len(phone_ids)] = np.asarray(phone_ids, dtype=np.int64)
            mask[0, : len(phone_ids)] = True
            input_names = self._component_input_names("duration_predictor")
            args: dict[int, np.ndarray] = (
                {}
                if input_names
                else {
                    0: padded,
                    1: np.asarray([voice_index], dtype=np.int64),
                    2: np.asarray([language_index], dtype=np.int64),
                    3: np.asarray([emotion_index], dtype=np.int64),
                    4: np.asarray([boundary_before_id], dtype=np.int64),
                    5: np.asarray([boundary_after_id], dtype=np.int64),
                    6: mask,
                }
            )
            self._add_optional_arg(args, "duration_predictor", "phone_ids", padded)
            self._add_optional_arg(
                args, "duration_predictor", "voice_id", np.asarray([voice_index], dtype=np.int64)
            )
            self._add_optional_arg(
                args, "duration_predictor", "language_id", np.asarray([language_index], dtype=np.int64)
            )
            self._add_optional_arg(
                args, "duration_predictor", "emotion_id", np.asarray([emotion_index], dtype=np.int64)
            )
            self._add_optional_arg(
                args, "duration_predictor", "affect_values", affect_features.values
            )
            self._add_optional_arg(
                args,
                "duration_predictor",
                "boundary_before_id",
                np.asarray([boundary_before_id], dtype=np.int64),
            )
            self._add_optional_arg(
                args,
                "duration_predictor",
                "boundary_after_id",
                np.asarray([boundary_after_id], dtype=np.int64),
            )
            self._add_optional_arg(args, "duration_predictor", "phone_mask", mask)
            self._add_optional_arg(
                args,
                "duration_predictor",
                "emotion_condition_mask",
                np.asarray([emotion_condition_scale], dtype=np.float32),
            )
            self._add_optional_arg(
                args,
                "duration_predictor",
                "affect_condition_mask",
                affect_features.condition_mask * float(affect_condition_scale),
            )
            self._add_reference_args(args, "duration_predictor", reference_features, reference_condition_scale)
            session = self._session("duration_predictor")
            if self.duration_pause_presence_enabled or self.duration_hierarchy_enabled:
                if not hasattr(session, "invoke_all"):
                    raise RuntimeError(
                        "Duration multi-output contract requires a "
                        "multi-output runtime session"
                    )
                outputs = list(session.invoke_all(args))
                expected_output_count = (
                    1
                    + int(self.duration_pause_presence_enabled)
                    + int(self.duration_hierarchy_enabled)
                )
                if len(outputs) != expected_output_count:
                    raise RuntimeError(
                        "Duration graph output count does not match its bundle contract"
                    )
            else:
                outputs = [session.invoke(args)]
            output = outputs[0]
            values = tuple(
                float(item)
                for item in np.asarray(output, dtype=np.float32).reshape(-1)[: len(phone_ids)]
            )
            presence_values = (
                tuple(
                    float(item)
                    for item in np.asarray(
                        outputs[1],
                        dtype=np.float32,
                    ).reshape(-1)[: len(phone_ids)]
                )
                if self.duration_pause_presence_enabled
                else None
            )
            quantile_index = 1 + int(self.duration_pause_presence_enabled)
            quantile_values: tuple[tuple[float, float, float], ...] | None = None
            if self.duration_hierarchy_enabled:
                quantile_array = np.asarray(
                    outputs[quantile_index],
                    dtype=np.float32,
                )
                if (
                    quantile_array.ndim != 3
                    or quantile_array.shape[0] != 1
                    or quantile_array.shape[2] != 3
                    or quantile_array.shape[1] < len(phone_ids)
                ):
                    raise RuntimeError(
                        "Duration quantiles must have shape [1, phone_frames, 3]"
                    )
                quantile_values = tuple(
                    tuple(float(item) for item in quantile_array[0, index, :])
                    for index in range(len(phone_ids))
                )
                for row in quantile_values:
                    if (
                        not all(math.isfinite(item) and item >= 0.0 for item in row)
                        or not row[0] <= row[1] <= row[2]
                    ):
                        raise RuntimeError(
                            "Duration quantiles must be finite, nonnegative, and monotonic"
                        )
            if (
                self.duration_pause_presence_enabled
                and (
                    presence_values is None
                    or len(presence_values) != len(values)
                    or len(values) != len(phone_ids)
                )
            ):
                raise RuntimeError(
                    "Duration and pause-presence outputs must align with active phones"
                )
            self._duration_cache[cache_key] = (
                values,
                presence_values,
                quantile_values,
            )
            self._duration_cache.move_to_end(cache_key)
            while len(self._duration_cache) > _FRONTEND_CACHE_SIZE:
                self._duration_cache.popitem(last=False)
            return (
                list(values),
                list(presence_values)
                if presence_values is not None
                else None,
                [list(row) for row in quantile_values]
                if quantile_values is not None
                else None,
            )

    def _sample_latents(
        self,
        expanded_phone_ids: list[int],
        *,
        latent_length: int,
        fixed_latent_frames: int,
        vector_component: str,
        span_context_hidden: np.ndarray | None,
        vector_timing: _VectorTimingFeatures | None,
        voice_index: int,
        language_index: int,
        emotion_index: int,
        affect_features: _AffectFeatures,
        affect_guidance_scale: float,
        guidance_terms: list[_EmotionGuidanceTerm],
        reference_features: _ReferenceFeatures,
        guidance_null_reference: bool,
        emotion_embed_scale: float,
        prefix_latents: Any,
        boundary_before_id: int,
        boundary_after_id: int,
        steps: int,
        sampler: str,
        seed: int | None,
        noise_scale: float,
    ) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be positive")
        sampler = str(sampler or "euler").lower()
        if sampler not in {"euler", "heun"}:
            raise ValueError(f"Unsupported sampler {sampler!r}; expected 'euler' or 'heun'")
        rng = np.random.default_rng(seed)
        latents = rng.standard_normal((1, self.latent_dim, fixed_latent_frames)).astype(np.float32)
        latents *= float(noise_scale)
        latent_mask = np.zeros((1, fixed_latent_frames), dtype=np.bool_)
        latent_mask[0, :latent_length] = True
        latents *= latent_mask[:, np.newaxis, :].astype(np.float32)
        expanded = np.zeros((1, fixed_latent_frames), dtype=np.int64)
        expanded[0, :latent_length] = np.asarray(expanded_phone_ids, dtype=np.int64)
        vector_timing = _pad_vector_timing_features(
            vector_timing,
            fixed_latent_frames=fixed_latent_frames,
        )
        voice = np.asarray([voice_index], dtype=np.int64)
        language = np.asarray([language_index], dtype=np.int64)
        emotion = np.asarray([emotion_index], dtype=np.int64)
        boundary_before = np.asarray([boundary_before_id], dtype=np.int64)
        boundary_after = np.asarray([boundary_after_id], dtype=np.int64)
        prefix_values, prefix_mask = self._prefix_inputs(prefix_latents, component_name=vector_component)
        mask_float = latent_mask[:, np.newaxis, :].astype(np.float32)
        dt = 1.0 / float(steps)
        for step in range(steps):
            if sampler == "euler":
                time_value = (float(step) + 0.5) / float(steps)
                velocity = self._guided_vector_velocity(
                    latents,
                    time_value,
                    expanded,
                    vector_component,
                    span_context_hidden,
                    vector_timing,
                    voice,
                    language,
                    emotion,
                    affect_features,
                    boundary_before,
                    boundary_after,
                    latent_mask,
                    affect_guidance_scale,
                    guidance_terms,
                    reference_features,
                    guidance_null_reference,
                    emotion_embed_scale,
                    prefix_values,
                    prefix_mask,
                )
                latents = (latents + dt * velocity) * mask_float
            else:
                start_time = float(step) / float(steps)
                end_time = float(step + 1) / float(steps)
                start_velocity = self._guided_vector_velocity(
                    latents,
                    start_time,
                    expanded,
                    vector_component,
                    span_context_hidden,
                    vector_timing,
                    voice,
                    language,
                    emotion,
                    affect_features,
                    boundary_before,
                    boundary_after,
                    latent_mask,
                    affect_guidance_scale,
                    guidance_terms,
                    reference_features,
                    guidance_null_reference,
                    emotion_embed_scale,
                    prefix_values,
                    prefix_mask,
                )
                predicted = (latents + dt * start_velocity) * mask_float
                end_velocity = self._guided_vector_velocity(
                    predicted,
                    end_time,
                    expanded,
                    vector_component,
                    span_context_hidden,
                    vector_timing,
                    voice,
                    language,
                    emotion,
                    affect_features,
                    boundary_before,
                    boundary_after,
                    latent_mask,
                    affect_guidance_scale,
                    guidance_terms,
                    reference_features,
                    guidance_null_reference,
                    emotion_embed_scale,
                    prefix_values,
                    prefix_mask,
                )
                latents = (latents + 0.5 * dt * (start_velocity + end_velocity)) * mask_float
        return latents.astype(np.float32)

    def _guided_vector_velocity(
        self,
        latents: np.ndarray,
        time_value: float,
        expanded: np.ndarray,
        vector_component: str,
        span_context_hidden: np.ndarray | None,
        vector_timing: _VectorTimingFeatures | None,
        voice: np.ndarray,
        language: np.ndarray,
        emotion: np.ndarray,
        affect_features: _AffectFeatures,
        boundary_before: np.ndarray,
        boundary_after: np.ndarray,
        latent_mask: np.ndarray,
        affect_guidance_scale: float,
        guidance_terms: list[_EmotionGuidanceTerm],
        reference_features: _ReferenceFeatures,
        guidance_null_reference: bool,
        emotion_embed_scale: float,
        prefix_latents: np.ndarray,
        prefix_mask: np.ndarray,
    ) -> np.ndarray:
        if affect_features.enabled and affect_guidance_scale != 1.0:
            null_velocity = self._vector_velocity(
                latents,
                time_value,
                expanded,
                vector_component,
                span_context_hidden,
                voice,
                language,
                emotion,
                affect_features,
                boundary_before,
                boundary_after,
                latent_mask,
                vector_timing=vector_timing,
                emotion_condition_scale=emotion_embed_scale,
                affect_condition_scale=0.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                prefix_latents=prefix_latents,
                prefix_mask=prefix_mask,
            )
            conditioned_velocity = self._vector_velocity(
                latents,
                time_value,
                expanded,
                vector_component,
                span_context_hidden,
                voice,
                language,
                emotion,
                affect_features,
                boundary_before,
                boundary_after,
                latent_mask,
                vector_timing=vector_timing,
                emotion_condition_scale=emotion_embed_scale,
                affect_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                prefix_latents=prefix_latents,
                prefix_mask=prefix_mask,
            )
            return np.asarray(
                null_velocity + float(affect_guidance_scale) * (conditioned_velocity - null_velocity),
                dtype=np.float32,
            )
        if not guidance_terms:
            return self._vector_velocity(
                latents,
                time_value,
                expanded,
                vector_component,
                span_context_hidden,
                voice,
                language,
                emotion,
                affect_features,
                boundary_before,
                boundary_after,
                latent_mask,
                vector_timing=vector_timing,
                emotion_condition_scale=emotion_embed_scale,
                affect_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                prefix_latents=prefix_latents,
                prefix_mask=prefix_mask,
            )
        if self._supports_batched_guidance(vector_component, 1 + len(guidance_terms)):
            return self._batched_guided_vector_velocity(
                latents,
                time_value,
                expanded,
                vector_component,
                span_context_hidden,
                vector_timing,
                voice,
                language,
                emotion,
                affect_features,
                boundary_before,
                boundary_after,
                latent_mask,
                guidance_terms,
                reference_features,
                guidance_null_reference,
                emotion_embed_scale,
                prefix_latents,
                prefix_mask,
            )
        null_reference_scale = 0.0 if guidance_null_reference else 1.0
        velocity = self._vector_velocity(
            latents,
            time_value,
            expanded,
            vector_component,
            span_context_hidden,
            voice,
            language,
            emotion,
            affect_features,
            boundary_before,
            boundary_after,
            latent_mask,
            vector_timing=vector_timing,
            emotion_condition_scale=0.0,
            affect_condition_scale=1.0,
            reference_features=reference_features,
            reference_condition_scale=null_reference_scale,
            prefix_latents=prefix_latents,
            prefix_mask=prefix_mask,
        )
        blended = _emotion_guidance_null_weight(guidance_terms) * velocity
        for term in guidance_terms:
            term_velocity = self._vector_velocity(
                latents,
                time_value,
                expanded,
                vector_component,
                span_context_hidden,
                voice,
                language,
                np.asarray([int(term.emotion_id)], dtype=np.int64),
                affect_features,
                boundary_before,
                boundary_after,
                latent_mask,
                vector_timing=vector_timing,
                emotion_condition_scale=emotion_embed_scale,
                affect_condition_scale=1.0,
                reference_features=reference_features,
                reference_condition_scale=1.0,
                prefix_latents=prefix_latents,
                prefix_mask=prefix_mask,
            )
            blended = blended + float(term.scale) * term_velocity
        return np.asarray(blended, dtype=np.float32)

    def _supports_batched_guidance(
        self,
        component_name: str,
        branch_count: int,
    ) -> bool:
        del component_name, branch_count
        return False

    def _batched_guided_vector_velocity(
        self,
        latents: np.ndarray,
        time_value: float,
        expanded: np.ndarray,
        vector_component: str,
        span_context_hidden: np.ndarray | None,
        vector_timing: _VectorTimingFeatures | None,
        voice: np.ndarray,
        language: np.ndarray,
        emotion: np.ndarray,
        affect_features: _AffectFeatures,
        boundary_before: np.ndarray,
        boundary_after: np.ndarray,
        latent_mask: np.ndarray,
        guidance_terms: list[_EmotionGuidanceTerm],
        reference_features: _ReferenceFeatures,
        guidance_null_reference: bool,
        emotion_embed_scale: float,
        prefix_latents: np.ndarray,
        prefix_mask: np.ndarray,
    ) -> np.ndarray:
        branch_count = 1 + len(guidance_terms)
        null_emotion = int(np.asarray(emotion).reshape(-1)[0])
        emotion_ids = np.asarray(
            [null_emotion, *(int(term.emotion_id) for term in guidance_terms)],
            dtype=np.int64,
        )
        emotion_scales = np.asarray(
            [0.0, *(float(emotion_embed_scale) for _ in guidance_terms)],
            dtype=np.float32,
        )
        reference_scales = np.asarray(
            [0.0 if guidance_null_reference else 1.0, *(1.0 for _ in guidance_terms)],
            dtype=np.float32,
        )
        args: dict[int, np.ndarray] = {
            0: _repeat_batch(latents, branch_count),
            1: np.full((branch_count,), float(time_value), dtype=np.float32),
            2: _repeat_batch(expanded, branch_count),
            3: _repeat_batch(voice, branch_count),
            4: _repeat_batch(language, branch_count),
            5: emotion_ids,
            6: _repeat_batch(boundary_before, branch_count),
            7: _repeat_batch(boundary_after, branch_count),
            8: _repeat_batch(latent_mask, branch_count),
        }
        self._add_optional_arg(args, vector_component, "emotion_condition_mask", emotion_scales)
        self._add_optional_arg(
            args,
            vector_component,
            "reference_style",
            _repeat_batch(reference_features.style, branch_count),
        )
        self._add_optional_arg(
            args,
            vector_component,
            "reference_prosody",
            _repeat_batch(reference_features.prosody, branch_count),
        )
        self._add_optional_arg(
            args,
            vector_component,
            "reference_mask",
            _repeat_batch(reference_features.mask, branch_count),
        )
        self._add_optional_arg(
            args,
            vector_component,
            "reference_condition_mask",
            reference_scales,
        )
        self._add_optional_arg(
            args,
            vector_component,
            "prefix_latents",
            _repeat_batch(prefix_latents, branch_count),
        )
        self._add_optional_arg(
            args,
            vector_component,
            "prefix_mask",
            _repeat_batch(prefix_mask, branch_count),
        )
        if span_context_hidden is not None:
            self._add_optional_arg(
                args,
                vector_component,
                "span_context_hidden",
                _repeat_batch(span_context_hidden, branch_count),
            )
        if vector_timing is not None:
            for name, value in _vector_timing_items(vector_timing):
                self._add_optional_arg(
                    args,
                    vector_component,
                    name,
                    _repeat_batch(value, branch_count),
                )
        velocities = np.asarray(self._session(vector_component).invoke(args), dtype=np.float32)
        if velocities.ndim != 3 or int(velocities.shape[0]) != branch_count:
            raise RuntimeError(
                f"Batched vector estimator returned shape {tuple(velocities.shape)}; "
                f"expected [{branch_count}, C, T]"
            )
        weights = np.asarray(
            [_emotion_guidance_null_weight(guidance_terms), *(term.scale for term in guidance_terms)],
            dtype=np.float32,
        )
        return np.sum(
            velocities * weights[:, np.newaxis, np.newaxis],
            axis=0,
            keepdims=True,
            dtype=np.float32,
        )

    def _vector_velocity(
        self,
        latents: np.ndarray,
        time_value: float,
        expanded: np.ndarray,
        component_name: str,
        span_context_hidden: np.ndarray | None,
        voice: np.ndarray,
        language: np.ndarray,
        emotion: np.ndarray,
        affect_features: _AffectFeatures,
        boundary_before: np.ndarray,
        boundary_after: np.ndarray,
        latent_mask: np.ndarray,
        *,
        vector_timing: _VectorTimingFeatures | None,
        emotion_condition_scale: float,
        affect_condition_scale: float,
        reference_features: _ReferenceFeatures,
        reference_condition_scale: float,
        prefix_latents: np.ndarray,
        prefix_mask: np.ndarray,
    ) -> np.ndarray:
        input_names = self._component_input_names(component_name)
        args: dict[int, np.ndarray] = (
            {}
            if input_names
            else {
                0: latents.astype(np.float32),
                1: np.asarray([float(time_value)], dtype=np.float32),
                2: expanded,
                3: voice,
                4: language,
                5: emotion,
                6: boundary_before,
                7: boundary_after,
                8: latent_mask,
            }
        )
        self._add_optional_arg(args, component_name, "noise", latents.astype(np.float32))
        self._add_optional_arg(
            args, component_name, "time", np.asarray([float(time_value)], dtype=np.float32)
        )
        self._add_optional_arg(args, component_name, "expanded_phone_ids", expanded)
        self._add_optional_arg(args, component_name, "voice_id", voice)
        self._add_optional_arg(args, component_name, "language_id", language)
        self._add_optional_arg(args, component_name, "emotion_id", emotion)
        self._add_optional_arg(args, component_name, "affect_values", affect_features.values)
        self._add_optional_arg(args, component_name, "boundary_before_id", boundary_before)
        self._add_optional_arg(args, component_name, "boundary_after_id", boundary_after)
        self._add_optional_arg(args, component_name, "latent_mask", latent_mask)
        self._add_optional_arg(args, component_name, "emotion_condition_mask", np.asarray([emotion_condition_scale], dtype=np.float32))
        self._add_optional_arg(
            args,
            component_name,
            "affect_condition_mask",
            affect_features.condition_mask * float(affect_condition_scale),
        )
        self._add_reference_args(args, component_name, reference_features, reference_condition_scale)
        self._add_optional_arg(args, component_name, "prefix_latents", prefix_latents)
        self._add_optional_arg(args, component_name, "prefix_mask", prefix_mask)
        if span_context_hidden is not None:
            self._add_optional_arg(args, component_name, "span_context_hidden", span_context_hidden)
        if vector_timing is not None:
            for name, value in _vector_timing_items(vector_timing):
                self._add_optional_arg(args, component_name, name, value)
        output = self._session(component_name).invoke(args)
        return np.asarray(output, dtype=np.float32)

    def _run_vocoder(
        self,
        latents: np.ndarray,
        *,
        component_name: str,
        latent_length: int,
        voice_index: int,
        language_index: int,
        emotion_index: int,
        reference_features: _ReferenceFeatures,
    ) -> np.ndarray:
        input_names = self._component_input_names(component_name)
        if not input_names:
            args: dict[int, np.ndarray] = {
                0: latents.astype(np.float32),
                1: np.asarray([voice_index], dtype=np.int64),
                2: np.asarray([language_index], dtype=np.int64),
                3: np.asarray([emotion_index], dtype=np.int64),
            }
        else:
            args = {}
            self._add_optional_arg(args, component_name, "latents", latents.astype(np.float32))
            latent_frames = int(latents.shape[-1]) if np.asarray(latents).ndim >= 3 else int(self.latent_frames)
            latent_mask = np.zeros((1, latent_frames), dtype=np.bool_)
            latent_mask[0, : max(0, min(int(latent_length), latent_frames))] = True
            self._add_optional_arg(args, component_name, "latent_mask", latent_mask)
            self._add_optional_arg(args, component_name, "voice_id", np.asarray([voice_index], dtype=np.int64))
            self._add_optional_arg(args, component_name, "language_id", np.asarray([language_index], dtype=np.int64))
            self._add_optional_arg(args, component_name, "emotion_id", np.asarray([emotion_index], dtype=np.int64))
            self._add_reference_args(args, component_name, reference_features, 1.0)
        output = self._session(component_name).invoke(args)
        return np.asarray(output, dtype=np.float32)

    def _add_reference_args(
        self,
        args: dict[int, np.ndarray],
        component_name: str,
        reference_features: _ReferenceFeatures,
        reference_condition_scale: float,
    ) -> None:
        self._add_optional_arg(args, component_name, "reference_style", reference_features.style)

        self._add_optional_arg(
            args,
            component_name,
            "identity_reference",
            reference_features.identity,
        )
        self._add_optional_arg(
            args,
            component_name,
            "identity_reference_mask",
            reference_features.mask * float(reference_condition_scale),
        )
        self._add_optional_arg(
            args,
            component_name,
            "prosody_baseline",
            reference_features.prosody_baseline,
        )
        self._add_optional_arg(
            args,
            component_name,
            "prosody_delta",
            reference_features.prosody_delta,
        )
        self._add_optional_arg(
            args,
            component_name,
            "prosody_feature_mask",
            reference_features.prosody_feature_mask,
        )
        self._add_optional_arg(
            args,
            component_name,
            "prosody_confidence",
            reference_features.prosody_confidence
            * float(reference_condition_scale),
        )
        self._add_optional_arg(args, component_name, "reference_prosody", reference_features.prosody)
        self._add_optional_arg(args, component_name, "reference_mask", reference_features.mask)
        self._add_optional_arg(
            args,
            component_name,
            "reference_condition_mask",
            np.asarray([reference_condition_scale], dtype=np.float32),
        )

    def _session(self, component_name: str) -> Any:
        if component_name not in self._sessions:
            self._sessions[component_name] = _LiteRTSession(self._component_path(component_name))
        return self._sessions[component_name]

    def _component_path(self, component_name: str) -> Path:
        component = self.manifest.components[component_name]
        artifact = component.artifacts.get("litert")
        if artifact is None:
            raise RuntimeError(f"Bundle component {component_name!r} has no LiteRT artifact")
        return self.bundle_dir / artifact.path

    def _latent_bucket_frames(self, latent_length: int) -> int:
        for frames in self.target_bucket_frames:
            if int(latent_length) <= int(frames):
                return int(frames)
        return int(self.target_bucket_frames[-1])

    def _bucket_component_name(self, base: str, latent_frames: int) -> str:
        frames = int(latent_frames)
        if frames == int(self.latent_frames):
            return base
        candidate = f"{base}_{frames}"
        if candidate in self.manifest.components:
            return candidate
        raise RuntimeError(
            f"Bundle target bucket {frames} requires component {candidate!r}; "
            "refusing to run the default fixed-shape component"
        )

    def _component_input_names(self, component_name: str) -> tuple[str, ...]:
        component = self.manifest.components.get(component_name)
        return tuple(component.inputs) if component is not None else ()

    def _component_supports_input(self, component_name: str, input_name: str) -> bool:
        return input_name in self._component_input_names(component_name)

    def _optional_input_index(self, component_name: str, input_name: str) -> int | None:
        inputs = self._component_input_names(component_name)
        try:
            return inputs.index(input_name)
        except ValueError:
            return None

    def _add_optional_arg(
        self,
        args: dict[int, np.ndarray],
        component_name: str,
        input_name: str,
        value: np.ndarray,
    ) -> None:
        index = self._optional_input_index(component_name, input_name)
        if index is not None:
            args[int(index)] = value

    def _reference_features(
        self,
        *,
        voice_id: str,
        language: str,
        emotion: str,
        affect: _AffectFeatures | None = None,
    ) -> _ReferenceFeatures:
        if not self._reference_inputs_enabled():
            return self._empty_reference_features()
        pack_dir = self.manifest.assets.get("voice_packs")
        if not pack_dir:
            return self._empty_reference_features()
        path = self.bundle_dir / pack_dir / f"{voice_id}.npz"
        if not path.is_file():
            return self._empty_reference_features(path=str(path))
        pack = self._load_voice_pack(path)

        if self.reference_schema_version == 4:
            if affect is None or not affect.enabled:
                raise ValueError(
                    "Schema-v4 reference packs require resolved five-axis affect values"
                )
            expected_version = str(
                self.reference_affect_routing.get("version") or ""
            )
            if expected_version != REFERENCE_ROUTING_V4_VERSION:
                raise ValueError(
                    "Unsupported schema-v4 affect-reference routing version: "
                    f"{expected_version!r}"
                )
            routed_v4 = route_reference_pack_v4(
                pack,
                language=language,
                affect_values=affect.values.reshape(-1),
                # Public requests resolve every axis, including the omitted-affect
                # neutral preset. This matches the training-side public route.
                affect_mask=np.ones((len(affect.axes),), dtype=np.float32),
            )
            if routed_v4.identity.shape != (self.reference_identity_dim,):
                raise ValueError(
                    "Schema-v4 identity width does not match the bundle manifest"
                )
            for label, value, width in (
                (
                    "prosody baseline",
                    routed_v4.prosody_baseline,
                    self.reference_baseline_dim,
                ),
                (
                    "prosody delta",
                    routed_v4.prosody_delta,
                    self.reference_delta_dim,
                ),
                (
                    "prosody feature mask",
                    routed_v4.prosody_feature_mask,
                    self.reference_prosody_dim,
                ),
            ):
                if value.shape != (width,):
                    raise ValueError(
                        f"Schema-v4 {label} width does not match the bundle manifest"
                    )
            prototype = (
                "none"
                if routed_v4.prototype_index is None
                else str(routed_v4.prototype_index)
            )
            return _ReferenceFeatures(
                style=np.zeros(
                    (1, self.reference_style_dim), dtype=np.float32
                ),
                prosody=np.zeros(
                    (1, self.reference_prosody_dim), dtype=np.float32
                ),
                identity=routed_v4.identity.reshape(1, -1).astype(np.float32),
                prosody_baseline=routed_v4.prosody_baseline.reshape(
                    1, -1
                ).astype(np.float32),
                prosody_delta=routed_v4.prosody_delta.reshape(
                    1, -1
                ).astype(np.float32),
                prosody_feature_mask=routed_v4.prosody_feature_mask.reshape(
                    1, -1
                ).astype(np.float32),
                prosody_confidence=np.asarray(
                    [routed_v4.prosody_confidence], dtype=np.float32
                ),
                mask=np.asarray([1.0], dtype=np.float32),
                native_mask=float(
                    routed_v4.route_kind.startswith("native")
                ),
                fallback_mask=float(
                    routed_v4.route_kind.startswith("global_affect_only")
                ),
                key=(
                    f"{language}|v4|{routed_v4.route_kind}|"
                    f"baseline={routed_v4.baseline_index}|prototype={prototype}"
                ),
                path=str(path),
            )
        if (
            affect is not None
            and affect.enabled
            and bool(self.reference_affect_routing.get("enabled", False))
        ):
            expected_version = str(
                self.reference_affect_routing.get("version") or ""
            )
            if expected_version != REFERENCE_ROUTING_VERSION:
                raise ValueError(
                    "Unsupported affect-reference routing version: "
                    f"{expected_version!r}"
                )
            routed = route_affect_reference(
                pack,
                language=language,
                axes=affect.axes,
                values=affect.values.reshape(-1),
            )
            if routed is not None:
                style_mean, style_std, prosody_mean, prosody_std = (
                    self._reference_normalization_stats()
                )
                return _ReferenceFeatures(
                    style=_normalize_style(
                        routed.style, style_mean, style_std
                    ).reshape(1, -1).astype(np.float32),
                    prosody=_normalize_prosody(
                        routed.prosody, prosody_mean, prosody_std
                    ).reshape(1, -1).astype(np.float32),

                    identity=np.zeros(
                        (1, self.reference_identity_dim), dtype=np.float32
                    ),
                    prosody_baseline=np.zeros(
                        (1, self.reference_baseline_dim), dtype=np.float32
                    ),
                    prosody_delta=np.zeros(
                        (1, self.reference_delta_dim), dtype=np.float32
                    ),
                    prosody_feature_mask=np.zeros(
                        (1, self.reference_prosody_dim), dtype=np.float32
                    ),
                    prosody_confidence=np.zeros((1,), dtype=np.float32),
                    mask=np.asarray([1.0], dtype=np.float32),
                    native_mask=1.0,
                    fallback_mask=0.0,
                    key=routed.key,
                    path=str(path),
                )
        suffix = self._select_reference_suffix(pack, language=language, emotion=emotion)
        if suffix is None:
            return self._empty_reference_features(path=str(path))
        reference_mask = _mask_value(pack.get(f"reference_mask__{suffix}"))
        native_mask = _mask_value(pack.get(f"native_reference_mask__{suffix}"))
        fallback_mask = _mask_value(pack.get(f"fallback_reference_mask__{suffix}"))
        if reference_mask <= 0.0:
            return self._empty_reference_features(path=str(path))
        style = _array_or_zeros(pack.get(f"style_embedding__{suffix}"), self.reference_style_dim)
        prosody = _array_or_zeros(pack.get(f"prosody_stats__{suffix}"), self.reference_prosody_dim)
        style_mean, style_std, prosody_mean, prosody_std = self._reference_normalization_stats()
        style = _normalize_style(style, style_mean, style_std)
        prosody = _normalize_prosody(prosody, prosody_mean, prosody_std)
        effective_mask = _effective_reference_mask(
            reference_mask,
            native_reference_mask=native_mask,
            fallback_reference_mask=fallback_mask,
            fallback_weight=self.reference_fallback_weight,
        )
        return _ReferenceFeatures(
            style=style.reshape(1, -1).astype(np.float32),
            prosody=prosody.reshape(1, -1).astype(np.float32),

            identity=np.zeros(
                (1, self.reference_identity_dim), dtype=np.float32
            ),
            prosody_baseline=np.zeros(
                (1, self.reference_baseline_dim), dtype=np.float32
            ),
            prosody_delta=np.zeros(
                (1, self.reference_delta_dim), dtype=np.float32
            ),
            prosody_feature_mask=np.zeros(
                (1, self.reference_prosody_dim), dtype=np.float32
            ),
            prosody_confidence=np.zeros((1,), dtype=np.float32),
            mask=np.asarray([effective_mask], dtype=np.float32),
            native_mask=float(native_mask),
            fallback_mask=float(fallback_mask),
            key=suffix.replace("__", "|", 1),
            path=str(path),
        )

    def _span_context_hidden(
        self,
        request: Any,
        *,
        target_phone_ids: list[int],
        language: str,
        component_name: str,
    ) -> np.ndarray | None:
        if not self._component_supports_input(component_name, "span_context_hidden"):
            return None
        span_config = self._span_context_config()
        if span_config and not bool(span_config.get("enabled", True)):
            hidden_size = int(
                span_config.get("hidden_size")
                or self.export_status.get("exported_components", {})
                .get("vector_estimator", {})
                .get("inputs", {})
                .get("span_context_hidden_size")
                or 512
            )
            return np.zeros((1, max(1, hidden_size)), dtype=np.float32)
        if "vector_context_encoder" not in self.manifest.components:
            raise RuntimeError(
                f"Bundle component {component_name!r} requires span_context_hidden, "
                "but vector_context_encoder is missing"
            )
        phone_ids, segment_ids, mask = self._span_context_inputs(
            request,
            target_phone_ids=target_phone_ids,
            language=language,
        )
        hidden = self._session("vector_context_encoder").invoke({0: phone_ids, 1: segment_ids, 2: mask})
        return np.asarray(hidden, dtype=np.float32)

    def _span_context_inputs(
        self,
        request: Any,
        *,
        target_phone_ids: list[int],
        language: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_phones = max(1, int(self._span_context_config().get("context_max_phones") or 768))
        before_ids = self._span_context_text_phone_ids(
            getattr(request, "context_before", None),
            language=language,
        )
        after_ids = self._span_context_text_phone_ids(
            getattr(request, "context_after", None),
            language=language,
        )
        target_ids = [int(item) for item in target_phone_ids]
        ids, segments = _balanced_span_context_ids(
            before_ids,
            target_ids,
            after_ids,
            max_phones=max_phones,
        )
        phone_ids = np.zeros((1, max_phones), dtype=np.int64)
        segment_ids = np.zeros((1, max_phones), dtype=np.int64)
        mask = np.zeros((1, max_phones), dtype=np.bool_)
        if ids:
            length = min(max_phones, len(ids))
            phone_ids[0, :length] = np.asarray(ids[:length], dtype=np.int64)
            segment_ids[0, :length] = np.asarray(segments[:length], dtype=np.int64)
            mask[0, :length] = True
        return phone_ids, segment_ids, mask

    def _span_context_text_phone_ids(self, text: Any, *, language: str) -> list[int]:
        value = str(text or "").strip()
        if not value:
            return []
        try:
            result = self.phonemize(
                value,
                language=language,
                boundary_before="chunk_continue",
                boundary_after="chunk_continue",
            )
        except ValueError:
            return []
        word_boundary_candidates = {
            int(item)
            for item in result.get(
                "g2p_word_boundary_candidate_indices", ()
            )
        }
        return [
            int(self.phone_to_id[phone])
            for index, phone in enumerate(result.get("phones", []))
            if (
                index not in word_boundary_candidates
                and str(phone) in self.phone_to_id
            )
        ]

    def _span_context_config(self) -> dict[str, Any]:
        controls = dict(getattr(self.manifest, "controls", {}) or {})
        config = controls.get("span_conditioning")
        return dict(config) if isinstance(config, Mapping) else {}

    def _reference_inputs_enabled(self) -> bool:
        return any(
            self._component_supports_input(component, "reference_style")
            or self._component_supports_input(
                component, "identity_reference"
            )
            for component in ("duration_predictor", "vector_estimator", "vocoder")
        )

    def _empty_reference_features(self, *, path: str | None = None) -> _ReferenceFeatures:
        return _ReferenceFeatures(
            style=np.zeros((1, self.reference_style_dim), dtype=np.float32),
            prosody=np.zeros((1, self.reference_prosody_dim), dtype=np.float32),

            identity=np.zeros(
                (1, self.reference_identity_dim), dtype=np.float32
            ),
            prosody_baseline=np.zeros(
                (1, self.reference_baseline_dim), dtype=np.float32
            ),
            prosody_delta=np.zeros(
                (1, self.reference_delta_dim), dtype=np.float32
            ),
            prosody_feature_mask=np.zeros(
                (1, self.reference_prosody_dim), dtype=np.float32
            ),
            prosody_confidence=np.zeros((1,), dtype=np.float32),
            mask=np.zeros((1,), dtype=np.float32),
            native_mask=0.0,
            fallback_mask=0.0,
            key=None,
            path=path,
        )

    def _load_voice_pack(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._voice_pack_cache.get(path)
        if cached is not None:
            return cached
        with np.load(str(path)) as payload:
            arrays = {str(key): np.asarray(payload[key]) for key in payload.files}
        self._voice_pack_cache[path] = arrays
        return arrays

    def _select_reference_suffix(self, pack: Mapping[str, np.ndarray], *, language: str, emotion: str) -> str | None:
        languages = [str(language)]
        if language == "en_gb":
            languages.append("en_us")
        elif language == "en":
            languages.extend(["en_us", "en_gb"])
        emotions = [str(emotion)]
        if emotion != "neutral":
            emotions.append("neutral")
        for lang in languages:
            for emo in emotions:
                suffix = f"{lang}__{emo}"
                if _mask_value(pack.get(f"reference_mask__{suffix}")) > 0.0:
                    return suffix
        return None

    def _reference_normalization_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._reference_stats is not None:
            return self._reference_stats
        style_rows: list[np.ndarray] = []
        prosody_rows: list[np.ndarray] = []
        pack_dir = self.manifest.assets.get("voice_packs")
        npz_dir = self.bundle_dir / pack_dir if pack_dir else None
        if npz_dir is not None and npz_dir.is_dir():
            for path in sorted(npz_dir.glob("*.npz")):
                pack = self._load_voice_pack(path)
                for key, value in pack.items():
                    if not key.startswith("reference_mask__") or _mask_value(value) <= 0.0:
                        continue
                    suffix = key[len("reference_mask__") :]
                    style_rows.append(_array_or_zeros(pack.get(f"style_embedding__{suffix}"), self.reference_style_dim))
                    prosody_rows.append(_transform_prosody(_array_or_zeros(pack.get(f"prosody_stats__{suffix}"), self.reference_prosody_dim)))
        style_mean, style_std = _standardize_rows(style_rows, self.reference_style_dim)
        prosody_mean, prosody_std = _standardize_rows(prosody_rows, self.reference_prosody_dim)
        self._reference_stats = (style_mean, style_std, prosody_mean, prosody_std)
        return self._reference_stats

    def _prefix_inputs(self, prefix_latents: Any, component_name: str = "vector_estimator") -> tuple[np.ndarray, np.ndarray]:
        frames = self.prefix_max_frames
        latents = np.zeros((1, self.latent_dim, frames), dtype=np.float32)
        mask = np.zeros((1, frames), dtype=np.bool_)
        if prefix_latents is None or not self._component_supports_input(component_name, "prefix_latents"):
            return latents, mask
        value = np.asarray(prefix_latents, dtype=np.float32)
        if value.ndim == 2:
            value = value[np.newaxis, :, :]
        if value.ndim != 3 or int(value.shape[1]) != self.latent_dim:
            return latents, mask
        keep = min(frames, int(value.shape[-1]))
        if keep <= 0:
            return latents, mask
        latents[:, :, -keep:] = value[:, :, -keep:]
        mask[:, -keep:] = True
        return latents, mask

    def _boundary_ids(self, request: Any, phone_result: Mapping[str, Any]) -> tuple[int, int]:
        before = phone_result.get("boundary_before") or getattr(request, "boundary_before", None)
        after = phone_result.get("boundary_after") or getattr(request, "boundary_after", None)
        before = _normalize_boundary_before(before)
        after = _normalize_boundary_after(after)
        return _boundary_before_id(before, self.g2p_config), _boundary_after_id(after, self.g2p_config)

    def _lookup(self, values: Mapping[str, int], key: str, label: str) -> int:
        try:
            return int(values[key])
        except KeyError as exc:
            options = ", ".join(sorted(values))
            raise ValueError(f"Unknown {label} {key!r}; available: {options}") from exc

    def _resolve_emotion(
        self,
        request: Any,
        guidance_terms: list[_EmotionGuidanceTerm],
    ) -> tuple[str, int]:
        if self.affect_enabled:
            return "neutral", int(self.emotion_to_id.get("neutral", 0))
        if guidance_terms:
            primary = max(guidance_terms, key=lambda term: term.scale)
            return primary.emotion, int(primary.emotion_id)
        raw = getattr(request, "emotion", None)
        if raw is None or not str(raw).strip():
            raw = getattr(request, "style_id", None)
        emotion = _normalize_emotion(raw)
        return emotion, self._lookup(self.emotion_to_id, emotion, "emotion")

    def _resolve_affect(self, request: Any) -> _AffectFeatures:
        raw = getattr(request, "affect", None)
        raw_emotion = str(getattr(request, "emotion", None) or "").strip()
        if not self.affect_enabled:
            if raw is not None and str(raw).strip():
                raise ValueError("This bundle does not support continuous affect conditioning")
            return _AffectFeatures(
                enabled=False,
                axes=(),
                values=np.zeros((1, 0), dtype=np.float32),
                condition_mask=np.zeros((1,), dtype=np.float32),
                requested=None,
                preset=None,
            )
        if getattr(request, "emotion_guidance", None):
            raise ValueError(
                "Legacy categorical emotion guidance is not supported by this continuous-affect bundle"
            )
        if raw is not None and raw_emotion and raw_emotion.lower() not in {"default", "neutral"}:
            raise ValueError("Pass either emotion axis values or a legacy emotion preset, not both")

        preset: str | None = None
        requested: object = raw
        spec: object = raw
        if spec is None or (isinstance(spec, str) and not spec.strip()):
            if raw_emotion and raw_emotion.lower() not in {"default", "neutral"}:
                legacy = self.affect_legacy_presets.get(raw_emotion)
                if legacy is None:
                    raise ValueError(
                        f"Emotion {raw_emotion!r} has no declared preset in this bundle"
                    )
                spec = legacy
                requested = raw_emotion
            else:
                spec = self.affect_config.get("default_preset")
                requested = None

        if isinstance(spec, str):
            value = spec.strip()
            if "=" not in value:
                if value in self.affect_presets:
                    preset = value
                    spec = self.affect_presets[value]
                elif value in self.affect_legacy_presets:
                    preset = value
                    spec = self.affect_legacy_presets[value]
                    if isinstance(spec, str):
                        preset = str(spec)
                        spec = self.affect_presets.get(preset)
                else:
                    options = ", ".join(sorted(self.affect_presets)) or "none"
                    raise ValueError(f"Unknown emotion preset {value!r}; available: {options}")
            else:
                parsed: dict[str, float] = {}
                for part in value.split(","):
                    name, separator, raw_value = part.strip().partition("=")
                    if not separator or not name.strip() or not raw_value.strip():
                        raise ValueError(
                            f"Invalid emotion term {part!r}; expected axis=value"
                        )
                    axis = name.strip().lower()
                    if axis in parsed:
                        raise ValueError(f"Duplicate emotion axis {axis!r}")
                    parsed[axis] = float(raw_value)
                spec = parsed

        values_by_axis: dict[str, float]
        partial_request = False
        if isinstance(spec, Mapping):
            values_by_axis = {str(key).strip().lower(): float(value) for key, value in spec.items()}
            partial_request = True
        elif isinstance(spec, (list, tuple)):
            if len(spec) != len(self.affect_axes):
                raise ValueError(
                    f"Emotion preset has {len(spec)} values; expected {len(self.affect_axes)}"
                )
            values_by_axis = {
                axis: float(value) for axis, value in zip(self.affect_axes, spec, strict=True)
            }
        else:
            raise ValueError("Emotion must be a preset, axis=value string, or axis mapping")
        unknown = sorted(set(values_by_axis) - set(self.affect_axes))
        if unknown:
            raise ValueError(f"Unknown emotion axis: {', '.join(unknown)}")
        vector = np.zeros((1, len(self.affect_axes)), dtype=np.float32)
        if partial_request:
            for index, axis in enumerate(self.affect_axes):
                vector[0, index] = float(self.affect_partial_defaults.get(axis, 0.0))
        for index, axis in enumerate(self.affect_axes):
            if axis not in values_by_axis:
                continue
            value = float(values_by_axis[axis])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"Emotion value for {axis!r} must be finite and within [0, 1]")
            minimum = float(self.affect_axis_minimums.get(axis, 0.0))
            maximum = float(self.affect_axis_maximums.get(axis, 1.0))
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Emotion value for {axis!r} must be within [{minimum}, {maximum}]"
                )
            vector[0, index] = value
        for index, axis in enumerate(self.affect_axes):
            value = float(vector[0, index])
            minimum = float(self.affect_axis_minimums.get(axis, 0.0))
            maximum = float(self.affect_axis_maximums.get(axis, 1.0))
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Emotion value for {axis!r} must be within [{minimum}, {maximum}]"
                )
        return _AffectFeatures(
            enabled=True,
            axes=self.affect_axes,
            values=vector,
            condition_mask=np.ones(
                vector.shape
                if int(self.affect_config.get("axis_order_version") or 0) == 3
                else (1,),
                dtype=np.float32,
            ),
            requested=requested,
            preset=preset,
        )

    def _resolve_affect_guidance_scale(
        self,
        request: Any,
        affect: _AffectFeatures,
    ) -> float:
        scale = float(getattr(request, "affect_guidance_scale", 1.0))
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("emotion_scale must be finite and non-negative")
        if not affect.enabled and scale != 1.0:
            raise ValueError("emotion_scale requires a continuous-affect bundle")
        return scale

    def _affect_metadata(self, affect: _AffectFeatures) -> dict[str, Any]:
        if not affect.enabled:
            return {"affect_enabled": False}
        return {
            "affect_enabled": True,
            "affect_axes": list(affect.axes),
            "affect_axis_order_version": self.affect_config.get("axis_order_version"),
            "affect_requested": affect.requested,
            "affect_preset": affect.preset,
            "affect_preset_version": self.affect_config.get("preset_version"),
            "affect_values": {
                axis: float(affect.values[0, index])
                for index, axis in enumerate(affect.axes)
            },
            "affect_vector": [float(value) for value in affect.values.reshape(-1)],
            "affect_condition_mask": (
                float(affect.condition_mask[0])
                if affect.condition_mask.size == 1
                else [
                    float(value)
                    for value in affect.condition_mask.reshape(-1)
                ]
            ),
        }

    def _parse_emotion_guidance(self, spec: str | None) -> list[_EmotionGuidanceTerm]:
        if spec is None or not str(spec).strip():
            return []
        terms: list[_EmotionGuidanceTerm] = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            name, _, raw_scale = part.partition(":")
            emotion = _normalize_emotion(name)
            scale = float(raw_scale) if raw_scale.strip() else 1.0
            if not math.isfinite(scale):
                raise ValueError(f"Invalid emotion guidance scale: {raw_scale!r}")
            if scale < 0.0:
                raise ValueError(f"Emotion guidance scale must be non-negative: {raw_scale!r}")
            terms.append(
                _EmotionGuidanceTerm(
                    emotion=emotion,
                    emotion_id=self._lookup(self.emotion_to_id, emotion, "emotion"),
                    scale=scale,
                )
            )
        return terms

    def _g2p_language(self, language: str) -> str:
        normalized = str(language).strip().lower().replace("-", "_")
        return str(self.g2p_language_map.get(normalized, normalized))


class _LiteRTSession:
    def __init__(self, model_path: Path) -> None:
        interpreter_cls = _load_interpreter_class()
        self.interpreter = interpreter_cls(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.inputs = list(self.interpreter.get_input_details())
        self.outputs = list(self.interpreter.get_output_details())
        self._inputs_by_arg = {_input_arg_index(detail): detail for detail in self.inputs}

    def invoke(self, args: Mapping[int, np.ndarray]) -> np.ndarray:
        return self.invoke_all(args)[0]

    def invoke_all(self, args: Mapping[int, np.ndarray]) -> list[np.ndarray]:
        missing = sorted(set(self._inputs_by_arg) - set(args))
        if missing:
            raise RuntimeError(f"Missing LiteRT input arg(s): {missing}")
        for arg_index, value in args.items():
            detail = self._inputs_by_arg[int(arg_index)]
            array = np.asarray(value, dtype=detail["dtype"])
            expected_shape = tuple(int(item) for item in detail["shape"])
            if tuple(array.shape) != expected_shape:
                raise RuntimeError(
                    f"LiteRT input args_{arg_index} expected shape {expected_shape}, got {tuple(array.shape)}"
                )
            self.interpreter.set_tensor(int(detail["index"]), array)
        self.interpreter.invoke()
        return [
            self.interpreter.get_tensor(int(output["index"]))
            for output in self.outputs
        ]


def _normalize_emotion(value: Any) -> str:
    emotion = str(value or "neutral").strip().lower().replace("-", "_").replace(" ", "_")
    if emotion in {"", "auto", "default", "none"}:
        return "neutral"
    return emotion


def _emotion_guidance_null_weight(terms: list[_EmotionGuidanceTerm]) -> float:
    if not terms:
        return 0.0
    return float(1.0 - sum(float(term.scale) for term in terms))


def _emotion_guidance_metadata(terms: list[_EmotionGuidanceTerm]) -> list[dict[str, Any]] | None:
    if not terms:
        return None
    return [
        {"emotion": term.emotion, "emotion_id": int(term.emotion_id), "scale": float(term.scale)}
        for term in terms
    ]


def _blend_optional_numpy_logits(
    null_logits: list[float] | None,
    term_logits: list[float] | None,
    *,
    null_weight: float,
    term_weight: float,
) -> list[float] | None:
    if null_logits is None and term_logits is None:
        return None
    if null_logits is None or term_logits is None:
        raise RuntimeError(
            "Duration guidance branches disagree on pause-presence output"
        )
    return (
        float(null_weight) * np.asarray(null_logits, dtype=np.float32)
        + float(term_weight) * np.asarray(term_logits, dtype=np.float32)
    ).tolist()


def _blend_optional_numpy_quantiles(
    null_quantiles: list[list[float]] | None,
    term_quantiles: list[list[float]] | None,
    *,
    null_weight: float,
    term_weight: float,
) -> list[list[float]] | None:
    if null_quantiles is None and term_quantiles is None:
        return None
    if null_quantiles is None or term_quantiles is None:
        raise RuntimeError(
            "Duration guidance branches disagree on quantile output"
        )
    blended = (
        float(null_weight) * np.asarray(null_quantiles, dtype=np.float32)
        + float(term_weight) * np.asarray(term_quantiles, dtype=np.float32)
    )
    return np.sort(np.maximum(blended, 0.0), axis=-1).tolist()


def _repeat_batch(value: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if int(array.shape[0]) == int(count):
        return np.ascontiguousarray(array)
    if int(array.shape[0]) != 1:
        raise ValueError(
            f"Cannot expand input with batch {array.shape[0]} to batch {int(count)}"
        )
    return np.repeat(array, int(count), axis=0)


def _array_or_zeros(value: np.ndarray | None, dim: int) -> np.ndarray:
    if value is None:
        return np.zeros((int(dim),), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    output = np.zeros((int(dim),), dtype=np.float32)
    count = min(int(dim), int(array.size))
    if count > 0:
        output[:count] = array[:count]
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _mask_value(value: np.ndarray | None) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return 0.0
    return 1.0 if float(array[0]) > 0.0 else 0.0


def _effective_reference_mask(
    reference_mask: float,
    *,
    native_reference_mask: float,
    fallback_reference_mask: float,
    fallback_weight: float,
) -> float:
    if reference_mask <= 0.0:
        return 0.0
    if native_reference_mask > 0.0:
        return 1.0
    if fallback_reference_mask > 0.0:
        return max(0.0, float(fallback_weight))
    return 1.0


def _standardize_rows(rows: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((int(dim),), dtype=np.float32), np.ones((int(dim),), dtype=np.float32)
    matrix = np.stack([_array_or_zeros(row, dim) for row in rows]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std).astype(np.float32)
    return mean, std


def _normalize_style(style: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    output = (np.asarray(style, dtype=np.float32) - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-4)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm = max(float(np.linalg.norm(output)), 1e-6)
    return (output / norm).astype(np.float32)


def _normalize_prosody(prosody: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = _transform_prosody(prosody)
    output = (transformed - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-4)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for index in PROSODY_DROP_INDICES:
        if index < int(output.size):
            output[index] = 0.0
    return output.astype(np.float32)


def _transform_prosody(prosody: np.ndarray) -> np.ndarray:
    output = np.nan_to_num(np.asarray(prosody, dtype=np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    for index in PROSODY_LOG1P_INDICES:
        if index < int(output.size):
            output[index] = np.log1p(max(0.0, float(output[index])))
    for index in PROSODY_DROP_INDICES:
        if index < int(output.size):
            output[index] = 0.0
    return output.astype(np.float32)


def _pronunciation_overrides_for_language(data: Mapping[str, Any], *, language: str) -> dict[str, Any]:
    languages = data.get("languages", data) if isinstance(data, Mapping) else {}
    if not isinstance(languages, Mapping):
        return {}
    raw = languages.get(str(language))
    if not isinstance(raw, Mapping):
        return {}
    return {str(word).strip().lower(): spec for word, spec in raw.items() if str(word).strip()}


def _words_for_pronunciation_overrides(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")}


def _terminal_g2p_word(text: str) -> str:
    matches = list(
        re.finditer(
            r"[^\W\d_]+(?:['’\-][^\W\d_]+)*",
            str(text or "").lower(),
            flags=re.UNICODE,
        )
    )
    return matches[-1].group(0) if matches else ""


def _trim_g2p_terminal_artifacts(
    phrase_phones: list[str],
    isolated_word_phones: list[str],
    *,
    max_removed_phones: int = 2,
) -> tuple[list[str], list[str]]:
    """Remove only a short suffix following an exact isolated-word match."""
    if not phrase_phones or not isolated_word_phones or max_removed_phones <= 0:
        return phrase_phones, []
    isolated_count = len(isolated_word_phones)
    earliest = max(0, len(phrase_phones) - isolated_count - int(max_removed_phones))
    latest = len(phrase_phones) - isolated_count
    for start in range(latest, earliest - 1, -1):
        end = start + isolated_count
        if phrase_phones[start:end] != isolated_word_phones:
            continue
        removed = phrase_phones[end:]
        if 0 < len(removed) <= int(max_removed_phones):
            return phrase_phones[:end], removed
    return phrase_phones, []


def _override_phone_list(spec: Any, *, key: str) -> list[str]:
    if isinstance(spec, Mapping):
        value = spec.get(key)
    elif key == "phones":
        value = spec
    else:
        value = None
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return []


def _replace_phone_subsequence(phones: list[str], source: list[str], target: list[str]) -> tuple[list[str], int]:
    if not phones or not source:
        return phones, 0
    output: list[str] = []
    count = 0
    index = 0
    source_len = len(source)
    while index < len(phones):
        if phones[index : index + source_len] == source:
            output.extend(target)
            index += source_len
            count += 1
        else:
            output.append(phones[index])
            index += 1
    return output, count


def _balanced_span_context_ids(
    before_ids: list[int],
    target_ids: list[int],
    after_ids: list[int],
    *,
    max_phones: int,
) -> tuple[list[int], list[int]]:
    max_phones = max(1, int(max_phones))
    target = [int(item) for item in target_ids[:max_phones]]
    if len(target) >= max_phones:
        return target, [1] * len(target)
    remaining = max_phones - len(target)
    before_keep = min(len(before_ids), (remaining + 1) // 2)
    after_keep = min(len(after_ids), remaining - before_keep)
    spare = remaining - before_keep - after_keep
    if spare > 0 and before_keep < len(before_ids):
        extra = min(spare, len(before_ids) - before_keep)
        before_keep += extra
        spare -= extra
    if spare > 0 and after_keep < len(after_ids):
        after_keep += min(spare, len(after_ids) - after_keep)
    before = [int(item) for item in before_ids[-before_keep:]] if before_keep else []
    after = [int(item) for item in after_ids[:after_keep]] if after_keep else []
    ids = before + target + after
    segments = ([0] * len(before)) + ([1] * len(target)) + ([2] * len(after))
    return ids, segments


def _target_bucket_frames(controls: Mapping[str, Any], default_latent_frames: int) -> tuple[int, ...]:
    target_buckets = controls.get("target_buckets") if isinstance(controls, Mapping) else None
    frames: list[int] = []
    if isinstance(target_buckets, Mapping):
        raw_buckets = target_buckets.get("buckets")
        if isinstance(raw_buckets, list):
            for item in raw_buckets:
                if isinstance(item, Mapping):
                    value = item.get("latent_frames")
                else:
                    value = item
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0 and parsed not in frames:
                    frames.append(parsed)
    default_frames = int(default_latent_frames)
    if default_frames > 0 and default_frames not in frames:
        frames.append(default_frames)
    return tuple(sorted(frames)) or (max(1, default_frames),)


def _load_interpreter_class() -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import]
        return Interpreter
    except Exception:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import]
        return Interpreter
    except Exception:
        pass
    try:
        from tensorflow.lite import Interpreter  # type: ignore[import]
        return Interpreter
    except Exception as exc:
        raise RuntimeError(
            "LiteRT graph execution requires ai-edge-litert, tflite-runtime, or TensorFlow. "
            "Run Scylla's Band with the deployment Python environment that contains a LiteRT interpreter."
        ) from exc


def _input_arg_index(detail: Mapping[str, Any]) -> int:
    name = str(detail.get("name", ""))
    match = re.search(r"args_(\d+)", name)
    if match:
        return int(match.group(1))
    if len(name) == 0 and "index" in detail:
        return int(detail["index"])
    raise RuntimeError(f"Cannot infer LiteRT argument index from tensor name {name!r}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = _load_json(path)
    return dict(data) if isinstance(data, dict) else {}


def _load_token_to_id(path: Path) -> dict[str, int]:
    data = _load_json(path)
    return {str(key): int(value) for key, value in data["token_to_id"].items()}


def _load_index(path: Path) -> dict[str, int]:
    rows = _load_json(path)
    return {str(row["id"]): int(row["index"]) for row in rows}


def _expanded_vector_timing_features(
    *,
    phones: list[str],
    durations: list[int],
    word_boundary_candidate_indices: set[int],
    punctuation_events: list[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any],
) -> _VectorTimingFeatures | None:
    """Derive the export timing quartet from final, post-floor durations."""

    if not bool(config.get("enabled", False)):
        return None
    if len(phones) != len(durations):
        raise ValueError("Vector timing phones and durations must align")
    values = [int(value) for value in durations]
    if any(value < 0 for value in values):
        raise ValueError("Vector timing durations must be non-negative")
    latent_length = sum(values)
    features = dict(config.get("features") or {})
    boundary_config = dict(config.get("boundary_events") or {})
    modifier_config = dict(config.get("modifier_events") or {})
    phone_config = dict(config.get("phone_vocab") or {})
    phone_vocab_size = int(phone_config.get("size") or 0)
    if phone_vocab_size <= 0:
        raise ValueError("Vector timing phone vocabulary size must be positive")

    spans: list[tuple[int, int] | None] = []
    cursor = 0
    phase = np.zeros((1, latent_length), dtype=np.float32)
    log_duration = np.zeros((1, latent_length), dtype=np.float32)
    for duration in values:
        if duration <= 0:
            spans.append(None)
            continue
        end = cursor + duration
        spans.append((cursor, end))
        if bool(features.get("local_timing", False)):
            phase[0, cursor:end] = (
                np.arange(duration, dtype=np.float32) + np.float32(0.5)
            ) / np.float32(duration)
            log_duration[0, cursor:end] = np.float32(math.log1p(duration))
        cursor = end

    boundary = np.zeros((1, latent_length), dtype=np.int64)
    if bool(features.get("boundary_events", False)):
        punctuation_symbols = [
            str(item) for item in boundary_config.get("punctuation_symbols", ())
        ]
        punctuation_ids = [
            int(item) for item in boundary_config.get("punctuation_phone_ids", ())
        ]
        punctuation_to_id = dict(
            zip(punctuation_symbols, punctuation_ids, strict=True)
        )
        handled_indices: set[int] = set()
        for event in punctuation_events or []:
            try:
                index = int(event.get("phone_index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(phones):
                raise ValueError("Vector timing punctuation event index is out of range")
            source_phone = str(event.get("source_phone") or "")
            event_id = punctuation_to_id.get(source_phone)
            if event_id is None:
                raise ValueError(
                    f"Punctuation event {source_phone!r} is absent from timing controls"
                )
            emitted_phone = str(event.get("emitted_phone") or phones[index])
            owner_index = index if emitted_phone == "<sil>" else index + 1
            if owner_index >= len(phones) or phones[owner_index] != "<sil>":
                raise ValueError(
                    f"Punctuation event {source_phone!r} lacks its <sil> owner"
                )
            owner = spans[owner_index]
            if owner is not None:
                boundary[0, owner[0] : owner[1]] = event_id
            else:
                owner = _next_vector_timing_span(
                    phones,
                    spans,
                    set(punctuation_symbols),
                    owner_index + 1,
                    1,
                )
                if owner is not None:
                    boundary[0, owner[0]] = event_id
            handled_indices.add(index)
        for index, phone in enumerate(phones):
            if index in handled_indices:
                continue
            event_id = punctuation_to_id.get(phone)
            if event_id is None:
                continue
            if values[index] != 0:
                raise ValueError(f"Punctuation {phone!r} must have zero duration")
            if index + 1 >= len(phones) or phones[index + 1] != "<sil>":
                raise ValueError(
                    f"Punctuation {phone!r} lacks its following <sil> owner"
                )
            owner = spans[index + 1]
            if owner is not None:
                boundary[0, owner[0] : owner[1]] = event_id
            else:
                owner = _next_vector_timing_span(
                    phones,
                    spans,
                    set(punctuation_symbols),
                    index + 2,
                    1,
                )
                if owner is not None:
                    boundary[0, owner[0]] = event_id
        synthetic_id = int(boundary_config.get("synthetic_word_boundary_id", -1))
        for index in sorted(int(item) for item in word_boundary_candidate_indices):
            if index < 0 or index >= len(spans):
                raise ValueError("Vector timing word-boundary index is out of range")
            owner = spans[index]
            if owner is not None:
                if not np.any(boundary[0, owner[0] : owner[1]]):
                    boundary[0, owner[0] : owner[1]] = synthetic_id
                continue
            owner = _next_vector_timing_span(
                phones,
                spans,
                set(punctuation_symbols),
                index + 1,
                1,
            )
            if owner is not None and boundary[0, owner[0]] == 0:
                boundary[0, owner[0]] = synthetic_id

    modifier = np.zeros((1, latent_length), dtype=np.int64)
    unrepresented_modifier_events = 0
    if bool(features.get("modifier_events", False)):
        modifier_symbols = [
            str(item) for item in modifier_config.get("symbols", ())
        ]
        modifier_masks = [
            int(item) for item in modifier_config.get("bit_masks", ())
        ]
        modifier_to_mask = dict(
            zip(modifier_symbols, modifier_masks, strict=True)
        )
        punctuation = set(
            str(item) for item in boundary_config.get("punctuation_symbols", ())
        )
        for index, phone in enumerate(phones):
            if not is_non_acoustic_phone_modifier(phone):
                continue
            if values[index] != 0:
                raise ValueError(f"Modifier {phone!r} must have zero duration")
            event_mask = modifier_to_mask.get(phone)
            if event_mask is None:
                raise ValueError(f"Modifier {phone!r} is absent from timing controls")
            is_stress = phone in {"ˈ", "ˌ"}
            if is_stress:
                owner = _next_vector_acoustic_span(
                    phones, spans, punctuation, index + 1, 1
                )
            else:
                owner = _postfix_vector_modifier_owner_span(
                    phones, spans, punctuation, index
                )
            if is_stress and owner is None:
                owner = _next_vector_acoustic_span(
                    phones, spans, punctuation, index - 1, -1
                )
            if owner is not None:
                modifier[0, owner[0] : owner[1]] |= event_mask
                continue
            unrepresented_modifier_events += 1

    return _VectorTimingFeatures(
        boundary_event_ids=boundary,
        modifier_event_ids=modifier,
        phone_phase=phase,
        phone_log_duration=log_duration,
        unrepresented_zero_frame_modifier_events=unrepresented_modifier_events,
    )


def _next_vector_timing_span(
    phones: list[str],
    spans: list[tuple[int, int] | None],
    punctuation: set[str],
    start: int,
    step: int,
) -> tuple[int, int] | None:
    index = start
    while 0 <= index < len(spans):
        if phones[index] == "<sil>" or phones[index] in punctuation:
            return None
        if spans[index] is not None:
            return spans[index]
        index += step
    return None


def _next_vector_acoustic_span(
    phones: list[str],
    spans: list[tuple[int, int] | None],
    punctuation: set[str],
    start: int,
    step: int,
) -> tuple[int, int] | None:
    index = start
    while 0 <= index < len(phones):
        phone = phones[index]
        if phone == "<sil>" or phone in punctuation:
            return None
        if spans[index] is not None and not is_non_acoustic_phone_modifier(phone):
            return spans[index]
        index += step
    return None


def _postfix_vector_modifier_owner_span(
    phones: list[str],
    spans: list[tuple[int, int] | None],
    punctuation: set[str],
    modifier_index: int,
) -> tuple[int, int] | None:
    index = modifier_index - 1
    while index >= 0 and is_non_acoustic_phone_modifier(phones[index]):
        index -= 1
    if index < 0 or phones[index] == "<sil>" or phones[index] in punctuation:
        return None
    return spans[index]


def _pad_vector_timing_features(
    features: _VectorTimingFeatures | None,
    *,
    fixed_latent_frames: int,
) -> _VectorTimingFeatures | None:
    if features is None:
        return None

    def pad(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
        source = np.asarray(value, dtype=dtype)
        if source.ndim != 2 or source.shape[0] != 1:
            raise ValueError("Vector timing arrays must have shape [1, frames]")
        if source.shape[1] > fixed_latent_frames:
            raise ValueError("Vector timing arrays exceed selected latent bucket")
        output = np.zeros((1, fixed_latent_frames), dtype=dtype)
        output[:, : source.shape[1]] = source
        return output

    return _VectorTimingFeatures(
        boundary_event_ids=pad(features.boundary_event_ids, np.dtype(np.int64)),
        modifier_event_ids=pad(features.modifier_event_ids, np.dtype(np.int64)),
        phone_phase=pad(features.phone_phase, np.dtype(np.float32)),
        phone_log_duration=pad(
            features.phone_log_duration, np.dtype(np.float32)
        ),
        unrepresented_zero_frame_modifier_events=(
            features.unrepresented_zero_frame_modifier_events
        ),
    )


def _vector_timing_items(
    features: _VectorTimingFeatures,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("expanded_boundary_event_ids", features.boundary_event_ids),
        ("expanded_modifier_event_ids", features.modifier_event_ids),
        ("expanded_phone_phase", features.phone_phase),
        ("expanded_phone_log_duration", features.phone_log_duration),
    )


def _frames_to_durations(
    values: list[float],
    *,
    phones: list[str],
    scale: float,
    min_phone_frames: int,
    zero_duration_punctuation: bool = False,
    word_boundary_candidate_indices: set[int] | None = None,
    word_boundary_presence_threshold_frames: float = 0.5,
) -> list[int]:
    candidates = {int(item) for item in (word_boundary_candidate_indices or set())}
    durations: list[int] = []
    for index, (value, phone) in enumerate(zip(values, phones)):
        scaled_frames = max(0.0, float(value)) * float(scale)
        frame_count = int(round(scaled_frames))
        if is_non_acoustic_phone_modifier(phone):
            frame_count = 0
        elif zero_duration_punctuation and phone in _ZERO_DURATION_PUNCTUATION_PHONES:
            frame_count = 0
        elif index in candidates:
            # Presence is a model decision in unscaled frame space. Pacing may
            # resize an observed pause, but must not create or erase one.
            frame_count = (
                0
                if max(0.0, float(value)) < float(
                    word_boundary_presence_threshold_frames
                )
                else max(1, frame_count)
            )
        elif phone and min_phone_frames > 0:
            frame_count = max(int(min_phone_frames), frame_count)
        durations.append(frame_count)
    return durations


def _apply_duration_pause_presence(
    durations: list[int],
    *,
    pause_presence_logits: list[float],
    phones: list[str],
    punctuation_events: list[Mapping[str, Any]],
    word_boundary_candidate_indices: set[int],
    threshold_probability: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    if len(durations) != len(phones) or len(pause_presence_logits) != len(phones):
        raise RuntimeError(
            "Duration pause-presence outputs do not align with phones"
        )
    if not 0.0 < float(threshold_probability) <= 1.0:
        raise ValueError("Duration pause-presence threshold must be in (0, 1]")
    punctuation_owned, word_owned = _duration_pause_owner_indices(
        phones=phones,
        punctuation_events=punctuation_events,
        word_boundary_candidate_indices=word_boundary_candidate_indices,
    )
    eligible = punctuation_owned | word_owned
    gated = list(durations)
    audit: list[dict[str, Any]] = []
    for index in sorted(eligible):
        logit = float(pause_presence_logits[index])
        probability = _stable_sigmoid(logit)
        present = probability >= float(threshold_probability)
        before = int(gated[index])
        gated[index] = max(1, before) if present else 0
        audit.append(
            {
                "phone_index": index,
                "probability": probability,
                "present": present,
                "duration_before_gate": before,
                "duration_after_gate": int(gated[index]),
                "punctuation_owned": index in punctuation_owned,
                "explicit_word_boundary": index in word_owned,
            }
        )
    return gated, audit


def _duration_pause_owner_indices(
    *,
    phones: list[str],
    punctuation_events: list[Mapping[str, Any]],
    word_boundary_candidate_indices: set[int],
) -> tuple[set[int], set[int]]:
    punctuation_owned: set[int] = set()
    for event in punctuation_events:
        try:
            phone_index = int(event.get("phone_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= phone_index < len(phones):
            continue
        emitted_phone = str(
            event.get("emitted_phone") or phones[phone_index]
        )
        owner = (
            phone_index
            if emitted_phone == "<sil>"
            else phone_index + 1
        )
        if 0 <= owner < len(phones) and phones[owner] == "<sil>":
            punctuation_owned.add(owner)
    for index, phone in enumerate(phones):
        if (
            phone == "<sil>"
            and index > 0
            and phones[index - 1] in _ZERO_DURATION_PUNCTUATION_PHONES
        ):
            punctuation_owned.add(index)
    word_owned = {
        int(index)
        for index in word_boundary_candidate_indices
        if 0 <= int(index) < len(phones) and phones[int(index)] == "<sil>"
    }
    return punctuation_owned, word_owned


def _duration_hierarchy_sampling_key(
    *,
    language: str,
    voice: str,
    phones: list[str],
) -> str:
    values = [str(language), str(voice), str(len(phones)), *phones]
    parts: list[str] = []
    for value in values:
        encoded = str(value).encode("utf-8")
        parts.append(f"{len(encoded)}:{encoded.decode('utf-8')}")
    return "|".join(parts)


def _duration_hierarchy_hash_normal(
    seed: int,
    scope: str,
    *,
    max_abs_z: float,
) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{scope}".encode("utf-8")).digest()
    first = (int.from_bytes(digest[:8], "big") + 0.5) / float(1 << 64)
    second = (int.from_bytes(digest[8:16], "big") + 0.5) / float(1 << 64)
    value = math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)
    return max(-float(max_abs_z), min(float(max_abs_z), value))


def _interpolate_duration_quantiles(
    quantiles: list[float],
    *,
    probability: float,
) -> float:
    if len(quantiles) != 3:
        raise RuntimeError("Duration quantiles must contain p10,p50,p90")
    lower, median, upper = (float(value) for value in quantiles)
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in quantiles)
        or not lower <= median <= upper
    ):
        raise RuntimeError(
            "Duration quantiles must be finite, nonnegative, and monotonic"
        )
    bounded = max(0.10, min(0.90, float(probability)))
    if bounded <= 0.50:
        alpha = (bounded - 0.10) / 0.40
        return lower + alpha * (median - lower)
    alpha = (bounded - 0.50) / 0.40
    return median + alpha * (upper - median)


def _apply_duration_hierarchy_sampling(
    p50_durations: list[int],
    *,
    duration_quantiles: list[list[float]],
    pause_presence_logits: list[float] | None,
    phones: list[str],
    punctuation_events: list[Mapping[str, Any]],
    word_boundary_candidate_indices: set[int],
    language: str,
    voice: str,
    seed: int,
    duration_scale: float,
    config: Mapping[str, Any],
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    if len(p50_durations) != len(phones) or len(duration_quantiles) != len(phones):
        raise RuntimeError("Duration hierarchy outputs do not align with phones")
    if pause_presence_logits is not None and len(pause_presence_logits) != len(phones):
        raise RuntimeError("Duration hierarchy presence logits do not align with phones")
    defaults = dict(config.get("defaults") or {})
    pause_strength = float(defaults.get("pause_strength"))
    speech_strength = float(defaults.get("speech_strength"))
    sample_presence = bool(defaults.get("sample_presence"))
    max_abs_z = float(defaults.get("max_abs_z"))
    punctuation_owned, word_owned = _duration_pause_owner_indices(
        phones=phones,
        punctuation_events=punctuation_events,
        word_boundary_candidate_indices=word_boundary_candidate_indices,
    )
    pause_owned = punctuation_owned | word_owned
    phrase_indices: list[int] = []
    phrase_index = 0
    for index in range(len(phones)):
        phrase_indices.append(phrase_index)
        if index in punctuation_owned:
            phrase_index += 1
    key = _duration_hierarchy_sampling_key(
        language=language,
        voice=voice,
        phones=phones,
    )
    key_sha256 = hashlib.sha256(key.encode("utf-8")).hexdigest()
    scope = f"duration-hierarchy:{key_sha256}"
    utterance_z = _duration_hierarchy_hash_normal(
        seed,
        f"{scope}:utterance",
        max_abs_z=max_abs_z,
    )
    phrase_units: list[float] = []
    phrase_quantiles: list[float] = []
    for current in range(phrase_index + 1):
        phrase_z = _duration_hierarchy_hash_normal(
            seed,
            f"{scope}:phrase:{current}",
            max_abs_z=max_abs_z,
        )
        combined_z = max(
            -max_abs_z,
            min(max_abs_z, (utterance_z + phrase_z) / math.sqrt(2.0)),
        )
        unit = 0.5 * (1.0 + math.erf(combined_z / math.sqrt(2.0)))
        phrase_units.append(unit)
        phrase_quantiles.append(
            max(0.10, min(0.90, 0.5 + (unit - 0.5) * pause_strength))
        )
    output = list(p50_durations)
    audit: list[dict[str, Any]] = []
    for index in sorted(pause_owned):
        phrase = phrase_indices[index]
        continuous = _interpolate_duration_quantiles(
            duration_quantiles[index],
            probability=phrase_quantiles[phrase],
        ) * float(duration_scale)
        probability = (
            _stable_sigmoid(float(pause_presence_logits[index]))
            if pause_presence_logits is not None
            else 1.0
        )
        present = (
            phrase_units[phrase] >= 1.0 - probability
            if sample_presence
            else probability >= 0.5
        )
        before = int(output[index])
        output[index] = max(1, int(round(continuous))) if present else 0
        audit.append(
            {
                "phone_index": index,
                "probability": probability,
                "sample_probability": phrase_units[phrase],
                "present": present,
                "duration_before_gate": before,
                "duration_after_gate": int(output[index]),
                "punctuation_owned": index in punctuation_owned,
                "explicit_word_boundary": index in word_owned,
            }
        )
    return output, audit, {
        "schema_version": "scyllasband_duration_hierarchy_sampling_v1",
        "enabled": True,
        "mode": "sampled",
        "policy": str(config.get("policy") or ""),
        "seed": int(seed),
        "sampling_key_sha256": key_sha256,
        "pause_strength": pause_strength,
        "speech_strength": speech_strength,
        "sample_presence": sample_presence,
        "max_abs_z": max_abs_z,
        "utterance_z": utterance_z,
        "before_total_frames": int(sum(p50_durations)),
        "after_total_frames": int(sum(output)),
    }


def _stable_sigmoid(value: float) -> float:
    numeric = float(value)
    if numeric >= 0.0:
        return 1.0 / (1.0 + math.exp(-numeric))
    exponential = math.exp(numeric)
    return exponential / (1.0 + exponential)


_ZERO_DURATION_PUNCTUATION_PHONES = frozenset({
    "<pause_comma>",
    "<pause_semicolon>",
    "<pause_colon>",
    "<pause_dash>",
    "<end_stmt>",
    "<end_question>",
    "<end_exclaim>",
    "<ellipsis>",
})
_SENTENCE_PUNCTUATION_PHONES = frozenset({"<end_stmt>", "<end_question>", "<end_exclaim>", "<ellipsis>", "<ctx_sentence_end>"})
_CLAUSE_PUNCTUATION_PHONES = frozenset({"<pause_comma>", "<pause_semicolon>", "<pause_colon>", "<pause_dash>", "<ctx_continuation>"})


def _request_has_following_chunk(request: Any) -> bool:
    try:
        chunk_index = int(getattr(request, "chunk_index", None))
        chunk_count = int(getattr(request, "chunk_count", None))
    except (TypeError, ValueError):
        return False
    return chunk_count > 0 and 0 <= chunk_index < chunk_count - 1


def _ellipsis_dot_counts_from_request(request: Any) -> tuple[int, ...]:
    explicit = getattr(request, "ellipsis_dot_counts", ())
    if explicit:
        counts = tuple(int(value) for value in explicit)
        if any(value < 2 for value in counts):
            raise ValueError("ellipsis_dot_counts values must be at least two")
        return counts
    text = str(getattr(request, "text", "") or "")
    return tuple(
        3 if match.group(0) == "…" else len(match.group(0))
        for match in re.finditer(r"\.{2,}|…", text)
    )


def _pause_ms_to_latent_frames(value: object, *, sample_rate: int, latent_hop_length: int) -> int:
    try:
        pause_ms = max(0.0, float(value or 0.0))
    except Exception:
        pause_ms = 0.0
    if pause_ms <= 0.0 or sample_rate <= 0 or latent_hop_length <= 0:
        return 0
    return max(1, int(round(pause_ms * float(sample_rate) / (1000.0 * float(latent_hop_length)))))


def _apply_punctuation_duration_floors(
    durations: list[int],
    *,
    phones: list[str],
    sentence_frames: int,
    clause_frames: int,
    calibrated_frames: Mapping[str, int] | None = None,
    semantic_phones: Mapping[int, str] | None = None,
    punctuation_repeat_counts: Mapping[int, int] | None = None,
    floor_terminal: bool = False,
    audit: list[dict[str, Any]] | None = None,
) -> list[int]:
    calibrated = {
        str(phone): max(0, int(frames))
        for phone, frames in dict(calibrated_frames or {}).items()
    }
    semantic = {
        int(index): str(phone)
        for index, phone in dict(semantic_phones or {}).items()
        if str(phone)
    }
    repeat_counts = {
        int(index): max(2, int(count))
        for index, count in dict(punctuation_repeat_counts or {}).items()
    }
    if (
        sentence_frames <= 0
        and clause_frames <= 0
        and not any(calibrated.values())
        and not any(count > 3 for count in repeat_counts.values())
    ):
        return durations

    def floor_for(phone: str) -> tuple[int, int, int]:
        calibrated_floor = int(calibrated.get(phone, 0))
        generic_floor = 0
        if phone in _SENTENCE_PUNCTUATION_PHONES:
            generic_floor = max(0, int(sentence_frames))
        elif phone in _CLAUSE_PUNCTUATION_PHONES:
            generic_floor = max(0, int(clause_frames))
        return max(calibrated_floor, generic_floor), calibrated_floor, generic_floor

    out = list(durations)
    for index, emitted_phone in enumerate(phones):
        if index >= len(out):
            break
        source_phone = semantic.get(index, emitted_phone)
        punctuation_floor, calibrated_floor, generic_floor = floor_for(source_phone)
        repeat_count = repeat_counts.get(index, 3)
        repeat_scale = min(4.0, max(1.0, float(repeat_count) / 3.0))
        if (
            source_phone in _SENTENCE_PUNCTUATION_PHONES
            or source_phone in _CLAUSE_PUNCTUATION_PHONES
        ) and (punctuation_floor > 0 or repeat_scale > 1.0):
            if emitted_phone in _SILENCE_PHONES:
                target_index = index
            else:
                target_index = (
                    index + 1
                    if index + 1 < len(phones) and phones[index + 1] in _SILENCE_PHONES
                    else index
                )
            if target_index >= len(out):
                continue
            terminal_target = target_index + 1 >= len(phones)
            learned_frames = int(out[target_index])
            floor_allowed = bool(floor_terminal or not terminal_target)
            if floor_allowed:
                out[target_index] = max(learned_frames, punctuation_floor)
            repeat_applied = False
            if source_phone == "<ellipsis>" and repeat_scale > 1.0:
                scaled_frames = int(round(float(out[target_index]) * repeat_scale))
                if scaled_frames > out[target_index]:
                    out[target_index] = scaled_frames
                    repeat_applied = True
            if audit is not None:
                audit.append(
                    {
                        "phone_index": index,
                        "target_phone_index": target_index,
                        "source_phone": source_phone,
                        "emitted_phone": emitted_phone,
                        "remapped": source_phone != emitted_phone,
                        "learned_frames": learned_frames,
                        "calibrated_floor_frames": calibrated_floor,
                        "generic_floor_frames": generic_floor,
                        "requested_floor_frames": punctuation_floor,
                        "applied_frames": int(out[target_index]),
                        "floor_applied": int(out[target_index]) > learned_frames,
                        "terminal_target": terminal_target,
                        "terminal_floor_allowed": floor_allowed,
                        "repeat_count": repeat_count,
                        "repeat_scale": repeat_scale,
                        "repeat_applied": repeat_applied,
                    }
                )
    return out


def _load_punctuation_pause_floor_table(
    punctuation_silence: Mapping[str, Any],
) -> tuple[dict[str, float], str | None]:
    raw_table = punctuation_silence.get("floor_table_ms")
    if not isinstance(raw_table, Mapping) or not raw_table:
        return {}, None
    if punctuation_silence.get("calibrated_floors") is not True:
        raise ValueError("Bundle punctuation floor table is not marked calibrated")
    table: dict[str, float] = {}
    for key, raw_floor in raw_table.items():
        floor = float(raw_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError(f"Bundle has invalid punctuation pause floor {key}={raw_floor!r}")
        table[str(key)] = floor
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    actual_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected_sha = str(punctuation_silence.get("floor_table_sha256") or "")
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(
            "Bundle punctuation pause floor-table hash mismatch: "
            f"{actual_sha} != {expected_sha or '<missing>'}"
        )
    return table, actual_sha


def _expand_phone_lists_to_length(phone_ids: list[int], durations: list[int], target_length: int) -> list[int]:
    if target_length <= 0:
        return []
    if not phone_ids:
        return [0 for _ in range(target_length)]
    adjusted = _adjust_durations_to_length(durations, target_length)
    expanded: list[int] = []
    for phone_id, duration in zip(phone_ids, adjusted):
        if duration <= 0:
            continue
        expanded.extend([int(phone_id)] * int(duration))
        if len(expanded) >= target_length:
            return expanded[:target_length]
    fallback = int(phone_ids[-1])
    if len(expanded) < target_length:
        expanded.extend([fallback] * (target_length - len(expanded)))
    return expanded[:target_length]


def _adjust_durations_to_length(durations: list[int], target_length: int) -> list[int]:
    if target_length <= 0:
        return [0 for _ in durations]
    if not durations:
        return []
    total = sum(max(0, int(item)) for item in durations)
    if total <= 0:
        adjusted = [0 for _ in durations]
        adjusted[0] = target_length
        return adjusted
    if total == target_length:
        return [max(0, int(item)) for item in durations]
    raw = [max(0.0, float(item)) * float(target_length) / float(total) for item in durations]
    floors = [int(math.floor(item)) for item in raw]
    positive = [idx for idx, item in enumerate(durations) if item > 0]
    if target_length >= len(positive):
        for idx in positive:
            floors[idx] = max(1, floors[idx])
    diff = target_length - sum(floors)
    if diff > 0:
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - math.floor(raw[idx]), reverse=True)
        for idx in order:
            if diff <= 0:
                break
            if durations[idx] <= 0:
                continue
            floors[idx] += 1
            diff -= 1
    elif diff < 0:
        removable = -diff
        order = sorted(range(len(floors)), key=lambda idx: floors[idx], reverse=True)
        for idx in order:
            if removable <= 0:
                break
            minimum = 1 if durations[idx] > 0 and target_length >= len(positive) else 0
            take = min(removable, max(0, floors[idx] - minimum))
            floors[idx] -= take
            removable -= take
    return floors


def _g2p_segment_config_for_fixed_text(
    config: Mapping[str, Any] | None,
    *,
    tokenizer: Mapping[str, Any] | None,
    fixed_text_tokens: int,
) -> dict[str, Any]:
    config_map = dict(config or {})
    configured_max = _positive_int_config(
        config_map.get("chunk_max_chars", config_map.get("phrase_chunk_max_chars")),
        DEFAULT_G2P_PHRASE_MAX_CHARS,
    )
    try:
        char_repeats = max(1, int(dict(tokenizer or {}).get("char_repeats", 1)))
    except Exception:
        char_repeats = 1
    fixed_tokens = max(1, int(fixed_text_tokens or 0))
    safe_chars = max(1, (fixed_tokens - 2) // char_repeats)
    config_map["chunk_max_chars"] = min(configured_max, safe_chars)
    config_map["fixed_text_tokens"] = fixed_tokens
    config_map["tokenizer_char_repeats"] = char_repeats
    return config_map


def _g2p_text_segments(text: str, *, config: Mapping[str, Any] | None = None) -> list[str]:
    config_map = dict(config or {})
    input_granularity = str(config_map.get("input_granularity") or "phrase").strip().lower()
    if input_granularity == "phrase":
        max_chars = _positive_int_config(
            config_map.get("chunk_max_chars", config_map.get("phrase_chunk_max_chars")),
            DEFAULT_G2P_PHRASE_MAX_CHARS,
        )
        drop_bracketed_notes = _bool_config(config_map.get("drop_bracketed_notes"), True)
        segments = g2p_phrase_segments(
            text,
            max_chars=max_chars,
            drop_bracketed_notes=drop_bracketed_notes,
        )
        return segments or [str(text or "").strip()]

    segments = [segment.strip() for segment in re.split(r"[.!?,;:]+", text) if segment.strip()]
    return segments or [str(text or "").strip()]


def _positive_int_config(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _skip_g2p_output_symbol(symbol: str | None) -> bool:
    if not symbol or symbol.startswith("<") or symbol == "_":
        return True
    if not symbol.strip():
        return True
    return all(unicodedata.category(char)[0] in {"P", "Z"} for char in symbol)


def _is_internal_word_boundary_marker(
    phones: list[str], index: int
) -> bool:
    if not (0 <= int(index) < len(phones)):
        return False
    if phones[index] != _WORD_BOUNDARY_MARKER:
        return False
    if index > 0 and phones[index - 1] == _WORD_BOUNDARY_MARKER:
        return False
    has_left = any(
        phone not in {_WORD_BOUNDARY_MARKER, *_SILENCE_PHONES}
        for phone in phones[:index]
    )
    has_right = any(
        phone not in {_WORD_BOUNDARY_MARKER, *_SILENCE_PHONES}
        for phone in phones[index + 1 :]
    )
    return has_left and has_right


def _prob_product(values: list[float]) -> float:
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += math.log(max(float(value), 1e-12))
    return float(math.exp(total))


_SILENCE_PHONES = frozenset({"<sil>", "sil", "sp", "<sp>"})


def _boundary_phone_for_segment(segment: str, phone_to_id: Mapping[str, int]) -> str | None:
    token = punctuation_phone_token(segment)
    if token and token in phone_to_id:
        return token
    return None


_CONTEXT_BOUNDARY_PHONE = {
    "paragraph_start": "<ctx_sentence_start>",
    "sentence_start": "<ctx_sentence_start>",
    "clause_continue": "<ctx_continuation>",
    "chunk_continue": "<ctx_chunk_continue>",
    "sentence_end": "<ctx_sentence_end>",
    "paragraph_end": "<ctx_sentence_end>",
}

_DEFAULT_BOUNDARY_BEFORE_VALUES = ["sentence_start", "paragraph_start", "clause_continue", "chunk_continue"]
_DEFAULT_BOUNDARY_AFTER_VALUES = ["sentence_end", "paragraph_end", "clause_continue", "chunk_continue"]


def _boundary_context_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, Mapping):
        chunk_context = config.get("chunk_context")
        if isinstance(chunk_context, Mapping):
            return dict(chunk_context)
    return {
        "schema_version": 2,
        "conditioning": "boundary_embeddings",
        "boundary_before_values": list(_DEFAULT_BOUNDARY_BEFORE_VALUES),
        "boundary_after_values": list(_DEFAULT_BOUNDARY_AFTER_VALUES),
        "default_boundary_before": "sentence_start",
        "default_boundary_after": "sentence_end",
        "emit_context_phone_tokens": False,
    }


def _boundary_before_id(boundary: str, config: Mapping[str, Any] | None) -> int:
    chunk_context = _boundary_context_config(config)
    values = [str(item) for item in chunk_context.get("boundary_before_values", _DEFAULT_BOUNDARY_BEFORE_VALUES)]
    if boundary not in values:
        boundary = str(chunk_context.get("default_boundary_before") or _DEFAULT_BOUNDARY_BEFORE_VALUES[0])
    return values.index(boundary) if boundary in values else 0


def _boundary_after_id(boundary: str, config: Mapping[str, Any] | None) -> int:
    chunk_context = _boundary_context_config(config)
    values = [str(item) for item in chunk_context.get("boundary_after_values", _DEFAULT_BOUNDARY_AFTER_VALUES)]
    if boundary not in values:
        boundary = str(chunk_context.get("default_boundary_after") or _DEFAULT_BOUNDARY_AFTER_VALUES[0])
    return values.index(boundary) if boundary in values else 0


def _normalize_boundary_before(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto", "none", "start", "utterance_start"}:
        return "paragraph_start"
    if normalized in {"paragraph", "paragraph_start", "document_start", "doc_start"}:
        return "paragraph_start"
    if normalized in {"sentence", "sentence_start"}:
        return "sentence_start"
    if normalized in {"clause", "clause_continue", "continuation", "continue"}:
        return "clause_continue"
    if normalized in {"chunk", "chunk_continue", "artificial_continue", "budget_continue"}:
        return "chunk_continue"
    return normalized


def _normalize_boundary_after(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto", "none", "end", "utterance_end"}:
        return "sentence_end"
    if normalized in {"paragraph", "paragraph_end", "document_end", "doc_end"}:
        return "paragraph_end"
    if normalized in {"sentence", "sentence_end"}:
        return "sentence_end"
    if normalized in {"clause", "clause_continue", "continuation", "continue"}:
        return "clause_continue"
    if normalized in {"chunk", "chunk_continue", "artificial_continue", "budget_continue"}:
        return "chunk_continue"
    return normalized


def _insert_leading_silence(boundary_before: str) -> bool:
    return boundary_before in {"paragraph_start", "sentence_start"}


def _emit_context_phone_tokens(config: Mapping[str, Any] | None) -> bool:
    chunk_context = dict(_boundary_context_config(config))
    return bool(chunk_context.get("emit_context_phone_tokens", False))


def _context_phone_for_boundary(
    boundary: str,
    phone_to_id: Mapping[str, int],
    config: Mapping[str, Any] | None,
) -> str | None:
    if not _emit_context_phone_tokens(config):
        return None
    token = _CONTEXT_BOUNDARY_PHONE.get(boundary)
    if token and token in phone_to_id:
        return token
    return None


def _continuation_phone(
    boundary: str,
    phone_to_id: Mapping[str, int],
    config: Mapping[str, Any] | None,
    *,
    fallback_pause: str | None,
) -> str | None:
    context_phone = _context_phone_for_boundary(boundary, phone_to_id, config)
    if context_phone is not None:
        return context_phone
    if "<pause_comma>" in phone_to_id:
        return "<pause_comma>"
    return fallback_pause


def _trailing_boundary_phone(
    boundary: str,
    phone_to_id: Mapping[str, int],
    config: Mapping[str, Any] | None,
) -> str | None:
    if boundary == "clause_continue":
        return _continuation_phone(boundary, phone_to_id, config, fallback_pause=None)
    if boundary == "chunk_continue":
        return None
    return _context_phone_for_boundary(boundary, phone_to_id, config)


def _ends_in_boundary_phone(phones: list[str]) -> bool:
    for phone in reversed(phones):
        if phone in _SILENCE_PHONES:
            continue
        return phone in BOUNDARY_PHONE_TOKENS
    return False
