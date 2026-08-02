"""Capability-driven validation job expansion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scyllasband.contract import ScyllasBandBundleManifest, validate_bundle_layout

from .corpus import canonical_json_bytes, load_suite, sha256_bytes, sha256_file


RUN_SCHEMA_VERSION = 1
JOB_SCHEMA_VERSION = 1
ENGLISH_LANGUAGES = {"en", "en_us", "en_gb"}


class ValidationJobError(ValueError):
    """Raised when a suite cannot be expanded against a bundle contract."""


def affect_axes(manifest: ScyllasBandBundleManifest) -> tuple[str, ...]:
    controls = manifest.controls if isinstance(manifest.controls, dict) else {}
    affect = controls.get("affect")
    if not isinstance(affect, dict) or not bool(affect.get("enabled", False)):
        return ()
    return tuple(str(item) for item in affect.get("axes", ()) if str(item))


def model_contract_fingerprint(manifest: ScyllasBandBundleManifest) -> str:
    payload = {
        "contract_version": manifest.contract_version,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "architecture": manifest.architecture,
        "audio": {
            "sample_rate": manifest.audio.sample_rate,
            "hop_length": manifest.audio.hop_length,
            "n_mels": manifest.audio.n_mels,
            "latent_dim": manifest.audio.latent_dim,
            "latent_hop_length": manifest.audio.latent_hop_length,
        },
        "languages": list(manifest.languages),
        "default_language": manifest.default_language,
        "voices": [
            {
                "id": voice.id,
                "languages": list(voice.languages),
                "default_language": voice.default_language,
                "styles": list(voice.styles),
            }
            for voice in manifest.voices
        ],
        "controls": manifest.controls,
        "components": {
            name: {
                "inputs": list(component.inputs),
                "outputs": list(component.outputs),
            }
            for name, component in sorted(manifest.components.items())
        },
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _matches(values: set[str] | None, value: str) -> bool:
    return not values or value in values


def _document_languages_for_voice(
    document: Mapping[str, Any],
    *,
    voice_languages: Iterable[str],
    default_language: str,
) -> list[str]:
    supported = set(str(item) for item in voice_languages)
    output: list[str] = []
    for language in document.get("languages", ()):
        language = str(language)
        if language not in supported:
            continue
        if language in {"en_us", "en_gb"} and default_language in {"en_us", "en_gb"}:
            if language != default_language:
                continue
        output.append(language)
    return output


def _selected_voices(
    manifest: ScyllasBandBundleManifest,
    *,
    tier: str,
    voice_filter: set[str] | None,
) -> list[Any]:
    voices = [voice for voice in manifest.voices if _matches(voice_filter, voice.id)]
    if tier != "smoke" or voice_filter:
        return voices
    selected: list[Any] = []
    seen_english: set[str] = set()
    for voice in voices:
        dialect = voice.default_language if voice.default_language in {"en_us", "en_gb"} else ""
        if dialect and dialect not in seen_english:
            selected.append(voice)
            seen_english.add(dialect)
        if seen_english == {"en_us", "en_gb"}:
            break
    for voice in voices:
        if len(selected) >= 2:
            break
        if voice not in selected:
            selected.append(voice)
    return selected


def _condition_affect(
    condition: Mapping[str, Any],
    axes: tuple[str, ...],
) -> tuple[dict[str, float] | None, list[str]]:
    ratings = {str(key): float(value) for key, value in condition.get("ratings", {}).items()}
    missing = sorted(axis for axis in ratings if axis not in axes)
    if missing:
        return None, missing
    if not axes:
        return (None, [] if not ratings else sorted(ratings))
    values = {axis: 0.0 for axis in axes}
    for axis, rating in ratings.items():
        values[axis] = rating / 4.0
    return values, []


def _tier_document_kinds(suite: Mapping[str, Any], tier: str) -> set[str]:
    tiers = suite.get("tiers", {})
    if isinstance(tiers, Mapping) and isinstance(tiers.get(tier), Mapping):
        values = tiers[tier].get("document_kinds", ())
        selected = {str(item) for item in values if str(item)}
        if selected:
            return selected
    if tier == "smoke":
        return {"challenge"}
    if tier == "core":
        return {"narrative"}
    if tier == "full":
        return {"narrative", "dialogue_punctuation", "challenge"}
    raise ValidationJobError(f"Unsupported validation tier: {tier}")


def _job_payload(
    *,
    suite: Mapping[str, Any],
    document: Mapping[str, Any],
    condition: Mapping[str, Any],
    affect: dict[str, float] | None,
    manifest: ScyllasBandBundleManifest,
    model_fingerprint: str,
    voice: Any,
    language: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": JOB_SCHEMA_VERSION,
        "suite_id": suite["id"],
        "suite_sha256": suite["suite_sha256"],
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "model_contract_sha256": model_fingerprint,
        "document_id": document["id"],
        "document_kind": document["kind"],
        "document_path": document["input_path"],
        "document_sha256": document["input_sha256"],
        "training_overlap": document["training_overlap"],
        "text": document["text"],
        "asr_reference_text": document["asr_reference_text"],
        "spoken_replacements": document.get("spoken_replacements", []),
        "challenge_spans": document.get("challenge_spans", []),
        "voice": voice.id,
        "language": language,
        "condition": condition["id"],
        "ratings": dict(condition.get("ratings", {})),
        "affect_axes": list(affect_axes(manifest)),
        "affect": affect,
        "generation": dict(generation),
    }
    contract = sha256_bytes(canonical_json_bytes(payload))
    payload["job_contract_sha256"] = contract
    payload["job_id"] = (
        f"{document['id']}__{voice.id}__{language}__{condition['id']}__{contract[:12]}"
    )
    return payload


def build_run(
    *,
    suite_path: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    tier: str = "core",
    voices: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    documents: Iterable[str] | None = None,
    steps: int = 8,
    sampler: str = "heun",
    seed: int = 2027,
    affect_guidance_scale: float = 1.0,
    max_chunk_chars: int = 220,
    min_chunk_chars: int = 48,
    min_sentence_pause_ms: float = 320.0,
    min_clause_pause_ms: float = 160.0,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    bundle_path = Path(bundle_dir).expanduser().resolve()
    # Planning is a release gate, so reject incomplete bundles and invalid affect
    # contracts before freezing thousands of jobs. In particular this prevents
    # the sixth v1 `questioning` coordinate from ever being treated as v2
    # `whisper` (or vice versa).
    manifest = validate_bundle_layout(bundle_path)
    model_fingerprint = model_contract_fingerprint(manifest)
    manifest_path = bundle_path / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    axes = affect_axes(manifest)

    voice_filter = {str(item) for item in voices or () if str(item)} or None
    language_filter = {str(item) for item in languages or () if str(item)} or None
    condition_filter = {str(item) for item in conditions or () if str(item)} or None
    document_filter = {str(item) for item in documents or () if str(item)} or None
    document_kinds = _tier_document_kinds(suite, tier)
    selected_voices = _selected_voices(manifest, tier=tier, voice_filter=voice_filter)

    generation = {
        "steps": int(steps),
        "sampler": str(sampler),
        "seed": int(seed),
        "affect_guidance_scale": float(affect_guidance_scale),
        "max_chunk_chars": int(max_chunk_chars),
        "min_chunk_chars": int(min_chunk_chars),
        "min_sentence_pause_ms": float(min_sentence_pause_ms),
        "min_clause_pause_ms": float(min_clause_pause_ms),
    }

    supported_conditions: list[tuple[Mapping[str, Any], dict[str, float] | None]] = []
    skipped_conditions: list[dict[str, Any]] = []
    for condition in suite["conditions"]:
        condition_id = str(condition["id"])
        if not _matches(condition_filter, condition_id):
            continue
        resolved_affect, missing_axes = _condition_affect(condition, axes)
        if missing_axes:
            skipped_conditions.append(
                {"condition": condition_id, "reason": "unsupported_affect_axes", "axes": missing_axes}
            )
            continue
        supported_conditions.append((condition, resolved_affect))

    jobs: list[dict[str, Any]] = []
    for document in suite["documents"]:
        if str(document["kind"]) not in document_kinds:
            continue
        if not _matches(document_filter, str(document["id"])):
            continue
        for voice in selected_voices:
            resolved_languages = _document_languages_for_voice(
                document,
                voice_languages=voice.languages,
                default_language=voice.default_language,
            )
            for language in resolved_languages:
                if not _matches(language_filter, language):
                    continue
                for condition, resolved_affect in supported_conditions:
                    jobs.append(
                        _job_payload(
                            suite=suite,
                            document=document,
                            condition=condition,
                            affect=resolved_affect,
                            manifest=manifest,
                            model_fingerprint=model_fingerprint,
                            voice=voice,
                            language=language,
                            generation=generation,
                        )
                    )
    jobs.sort(key=lambda item: str(item["job_id"]))
    if not jobs:
        raise ValidationJobError("Validation filters resolved to zero jobs")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    run = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_kind": "scyllasband_long_form_validation_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite["id"],
        "suite_path": str(Path(suite["suite_path"])),
        "suite_sha256": suite["suite_sha256"],
        "tier": tier,
        "bundle_dir": str(bundle_path),
        "bundle_manifest_sha256": manifest_sha,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "model_contract_sha256": model_fingerprint,
        "capabilities": {
            "languages": list(manifest.languages),
            "voices": [
                {
                    "id": voice.id,
                    "languages": list(voice.languages),
                    "default_language": voice.default_language,
                }
                for voice in manifest.voices
            ],
            "affect_axes": list(axes),
            "preferred_backends": list(manifest.preferred_backends),
        },
        "generation": generation,
        "job_count": len(jobs),
        "skipped_conditions": skipped_conditions,
    }
    # Absolute paths and wall-clock time are provenance, not render semantics.
    run_contract = {
        key: value
        for key, value in run.items()
        if key not in {"created_at", "suite_path", "bundle_dir"}
    }
    run["run_contract_sha256"] = sha256_bytes(canonical_json_bytes(run_contract))

    run_path = output_path / "run.json"
    jobs_path = output_path / "jobs.jsonl"
    if run_path.exists() or jobs_path.exists():
        if run_path.is_file() and jobs_path.is_file():
            existing_run, existing_jobs = load_run(output_path)
            same_jobs = [item.get("job_contract_sha256") for item in existing_jobs] == [
                item.get("job_contract_sha256") for item in jobs
            ]
            if existing_run.get("run_contract_sha256") == run["run_contract_sha256"] and same_jobs:
                return {**existing_run, "output_dir": str(output_path), "reused": True}
        raise ValidationJobError(
            f"Validation output already contains a different frozen run: {output_path}"
        )

    _write_json(run_path, run)
    with jobs_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    return {**run, "output_dir": str(output_path)}


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(run_dir).expanduser().resolve()
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    jobs = [
        json.loads(line)
        for line in (root / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if int(run.get("job_count") or -1) != len(jobs):
        raise ValidationJobError("run.json job_count does not match jobs.jsonl")
    for job in jobs:
        claimed = str(job.get("job_contract_sha256") or "")
        contract_payload = {
            key: value for key, value in job.items()
            if key not in {"job_id", "job_contract_sha256"}
        }
        if not claimed or claimed != sha256_bytes(canonical_json_bytes(contract_payload)):
            raise ValidationJobError(f"Frozen job contract is invalid: {job.get('job_id')}")
    claimed_run = str(run.get("run_contract_sha256") or "")
    run_payload = {
        key: value for key, value in run.items()
        if key not in {"created_at", "suite_path", "bundle_dir", "run_contract_sha256"}
    }
    if not claimed_run or claimed_run != sha256_bytes(canonical_json_bytes(run_payload)):
        raise ValidationJobError("Frozen run contract is invalid")
    return run, jobs


def selected_jobs(
    jobs: Iterable[dict[str, Any]],
    *,
    voices: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    documents: Iterable[str] | None = None,
    job_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    filters = {
        "voice": {str(item) for item in voices or () if str(item)},
        "language": {str(item) for item in languages or () if str(item)},
        "condition": {str(item) for item in conditions or () if str(item)},
        "document_id": {str(item) for item in documents or () if str(item)},
        "job_id": {str(item) for item in job_ids or () if str(item)},
    }
    output = []
    for job in jobs:
        if all(not values or str(job.get(key)) in values for key, values in filters.items()):
            output.append(job)
    return output


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
