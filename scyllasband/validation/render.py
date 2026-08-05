"""Resumable long-form rendering through public Scylla's Band backends."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any, Iterable, Mapping

from scyllasband.contract import load_bundle_manifest, validate_bundle_layout
from scyllasband.planner import PlannerOptions
from scyllasband.runtime import ScyllasBandRuntime
from scyllasband.streaming import render_text_records

from .audio import audio_metrics, file_sha256, read_wav, write_wav
from .corpus import canonical_json_bytes, sha256_bytes, sha256_file
from .jobs import load_run, model_contract_fingerprint, selected_jobs


RENDER_SCHEMA_VERSION = 1
SUPPORTED_PUBLIC_BACKENDS = {"onnx", "litert", "coreml", "coreai"}


class ValidationRenderError(RuntimeError):
    """Raised when a backend cannot render a frozen validation job."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    return str(value)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def plan_signature(plan: Mapping[str, Any]) -> str:
    payload = {
        "version": plan.get("version"),
        "source_kind": plan.get("source_kind"),
        "normalized_text": plan.get("normalized_text"),
        "records": [
            {
                key: record.get(key)
                for key in (
                    "record_id",
                    "voice",
                    "language",
                    "text",
                    "normalized_text",
                    "affect",
                    "affect_guidance_scale",
                )
            }
            for record in plan.get("records", ())
        ],
        "chunks": [
            {
                key: chunk.get(key)
                for key in (
                    "chunk_id",
                    "record_id",
                    "order_index",
                    "text",
                    "voice",
                    "language",
                    "context_before",
                    "context_after",
                    "boundary_before",
                    "boundary_after",
                    "prefix_policy",
                )
            }
            for chunk in plan.get("chunks", ())
        ],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _component_inventory(bundle_dir: Path, backend: str, *, hash_components: bool) -> dict[str, Any]:
    manifest = load_bundle_manifest(bundle_dir)
    output: dict[str, Any] = {}
    for name, component in manifest.components.items():
        declared = None
        if component.artifacts and backend in component.artifacts:
            declared = component.artifacts[backend].path
        elif component.path:
            declared = component.path
        if not declared:
            continue
        path = bundle_dir / declared
        exists = path.is_file() or (backend == "coreai" and path.is_dir())
        record: dict[str, Any] = {
            "path": declared,
            "exists": exists,
            "size_bytes": _artifact_size(path) if exists else None,
        }
        if hash_components and exists:
            record["sha256"] = (
                sha256_file(path) if path.is_file() else _artifact_tree_sha256(path)
            )
        output[name] = record
    return output


def _artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _artifact_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _backend_metadata(
    *,
    bundle_dir: Path,
    backend: str,
    hash_components: bool,
) -> dict[str, Any]:
    manifest = validate_bundle_layout(bundle_dir)
    return {
        "schema_version": RENDER_SCHEMA_VERSION,
        "artifact_kind": "scyllasband_validation_backend",
        "backend": backend,
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "model_contract_sha256": model_contract_fingerprint(manifest),
        "preferred_backends": list(manifest.preferred_backends),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "component_inventory": _component_inventory(
            bundle_dir,
            backend,
            hash_components=hash_components,
        ),
    }


def _planner_options(job: Mapping[str, Any], backend: str) -> PlannerOptions:
    generation = dict(job.get("generation", {}))
    return PlannerOptions(
        max_chunk_chars=int(generation.get("max_chunk_chars", 220)),
        min_chunk_chars=int(generation.get("min_chunk_chars", 48)),
        seed=int(generation.get("seed", 2027)),
        steps=int(generation.get("steps", 8)),
        sampler=str(generation.get("sampler", "heun")),
        backend=backend,
        affect=job.get("affect"),
        affect_guidance_scale=float(generation.get("affect_guidance_scale", 1.0)),
        min_sentence_pause_ms=float(generation.get("min_sentence_pause_ms", 320.0)),
        min_clause_pause_ms=float(generation.get("min_clause_pause_ms", 160.0)),
    )


def _render_complete(path: Path, job: Mapping[str, Any]) -> bool:
    metadata_path = path / "metadata.json"
    audio_path = path / "audio.wav"
    if not metadata_path.is_file() or not audio_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("state") == "complete"
        and metadata.get("job_contract_sha256") == job.get("job_contract_sha256")
        and metadata.get("audio_sha256") == file_sha256(audio_path)
    )


def _write_chunk_wavs(
    output_dir: Path,
    audio: list[float],
    sample_rate: int,
    chunks: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fallback_index, chunk in enumerate(chunks):
        index = int(chunk.get("index", fallback_index))
        start = max(0, int(chunk.get("start_sample", 0)))
        end = min(len(audio), max(start, int(chunk.get("end_sample", start))))
        path = output_dir / "chunks" / f"{index:04d}.wav"
        write_wav(path, audio[start:end], sample_rate)
        records.append(
            {
                "index": index,
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(path),
                "start_sample": start,
                "end_sample": end,
                "text": chunk.get("text"),
                "source_text": chunk.get("source_text"),
                "metrics": audio_metrics(audio[start:end], sample_rate),
            }
        )
    return records


def render_backend(
    *,
    run_dir: str | Path,
    backend: str,
    bundle_dir: str | Path,
    voices: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    documents: Iterable[str] | None = None,
    job_ids: Iterable[str] | None = None,
    overwrite_failed: bool = False,
    fail_fast: bool = False,
    hash_components: bool = True,
    litert_accelerator: str = "cpu",
    onnx_providers: list[str] | None = None,
    onnx_intra_op_num_threads: int = 0,
    onnx_inter_op_num_threads: int = 0,
) -> dict[str, Any]:
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED_PUBLIC_BACKENDS:
        raise ValidationRenderError(f"Unsupported public validation backend: {backend}")
    if backend == "coreml":
        raise ValidationRenderError("Core ML validation is reserved by the schema but not implemented yet")
    run_root = Path(run_dir).expanduser().resolve()
    run, all_jobs = load_run(run_root)
    jobs = selected_jobs(
        all_jobs,
        voices=voices,
        languages=languages,
        conditions=conditions,
        documents=documents,
        job_ids=job_ids,
    )
    bundle_path = Path(bundle_dir).expanduser().resolve()
    backend_record = _backend_metadata(
        bundle_dir=bundle_path,
        backend=backend,
        hash_components=hash_components,
    )
    if backend_record["model_contract_sha256"] != run.get("model_contract_sha256"):
        raise ValidationRenderError(
            "Backend bundle model contract does not match the frozen validation run"
        )
    backend_root = run_root / "renders" / backend
    backend_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(backend_root / "backend.json", backend_record)

    runtime = ScyllasBandRuntime.from_bundle(
        bundle_path,
        backends=[backend],
        litert_accelerator=litert_accelerator,
        onnx_providers=onnx_providers,
        onnx_intra_op_num_threads=onnx_intra_op_num_threads,
        onnx_inter_op_num_threads=onnx_inter_op_num_threads,
    )
    completed = skipped = failed = 0
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        job_root = backend_root / str(job["job_id"])
        if _render_complete(job_root, job):
            skipped += 1
            continue
        if job_root.exists() and overwrite_failed:
            shutil.rmtree(job_root)
        job_root.mkdir(parents=True, exist_ok=True)
        job_started = time.perf_counter()
        try:
            records = [
                {
                    "voice": job["voice"],
                    "language": job["language"],
                    "affect": job.get("affect"),
                    "affect_guidance_scale": job["generation"].get("affect_guidance_scale", 1.0),
                    "text": job["text"],
                }
            ]
            audio, sample_rate, synthesis = render_text_records(
                runtime,
                records,
                _planner_options(job, backend),
                source_kind="validation",
                progress_stream=None,
            )
            elapsed = time.perf_counter() - job_started
            audio_path = write_wav(job_root / "audio.wav", audio, sample_rate)
            chunks = _write_chunk_wavs(
                job_root,
                audio,
                sample_rate,
                synthesis.get("chunks", ()),
            )
            duration = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
            metadata = {
                "schema_version": RENDER_SCHEMA_VERSION,
                "artifact_kind": "scyllasband_validation_render",
                "state": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "backend": backend,
                "job_id": job["job_id"],
                "job_contract_sha256": job["job_contract_sha256"],
                "model_contract_sha256": backend_record["model_contract_sha256"],
                "bundle_manifest_sha256": backend_record["bundle_manifest_sha256"],
                "audio_path": "audio.wav",
                "audio_sha256": file_sha256(audio_path),
                "audio_metrics": audio_metrics(audio, sample_rate),
                "sample_rate": sample_rate,
                "duration_seconds": duration,
                "elapsed_seconds": elapsed,
                "realtime_factor": elapsed / duration if duration > 0 else None,
                "plan_signature": plan_signature(synthesis.get("plan", {})),
                "synthesis": synthesis,
                "chunks": chunks,
            }
            _write_json_atomic(job_root / "metadata.json", metadata)
            error_path = job_root / "error.json"
            if error_path.exists():
                error_path.unlink()
            completed += 1
            print(
                f"[{backend}] {index}/{len(jobs)} complete {job['job_id']} "
                f"({duration:.1f}s audio, RTF {metadata['realtime_factor']:.2f})",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            error = {
                "state": "failed",
                "backend": backend,
                "job_id": str(job["job_id"]),
                "job_contract_sha256": str(job["job_contract_sha256"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json_atomic(job_root / "error.json", error)
            errors.append({"job_id": str(job["job_id"]), "error": str(exc)})
            print(f"[{backend}] FAILED {job['job_id']}: {exc}", file=sys.stderr, flush=True)
            if fail_fast:
                raise
    return {
        "backend": backend,
        "selected_jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.perf_counter() - started,
        "errors": errors,
    }


def import_backend_results(
    *,
    run_dir: str | Path,
    backend: str,
    source_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    _, jobs = load_run(run_root)
    by_id = {str(job["job_id"]): job for job in jobs}
    source = Path(source_dir).expanduser().resolve()
    destination = run_root / "renders" / str(backend)
    destination.mkdir(parents=True, exist_ok=True)
    imported = skipped = rejected = 0
    for metadata_path in sorted(source.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        job_id = str(metadata.get("job_id") or metadata_path.parent.name)
        job = by_id.get(job_id)
        audio_path = metadata_path.parent / "audio.wav"
        contract_matches = (
            job is not None
            and metadata.get("job_contract_sha256") == job.get("job_contract_sha256")
            and metadata.get("state") == "complete"
            and audio_path.is_file()
            and metadata.get("audio_sha256") == file_sha256(audio_path)
        )
        model_contract = metadata.get("model_contract_sha256")
        if job is not None and model_contract and model_contract != job.get("model_contract_sha256"):
            contract_matches = False
        if not contract_matches:
            rejected += 1
            continue
        target = destination / job_id
        if target.exists() and not overwrite:
            skipped += 1
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(metadata_path.parent, target)
        copied_metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        copied_metadata["backend"] = str(backend)
        copied_metadata["imported_from"] = str(metadata_path.parent)
        _write_json_atomic(target / "metadata.json", copied_metadata)
        imported += 1
    return {"backend": backend, "imported": imported, "skipped": skipped, "rejected": rejected}
