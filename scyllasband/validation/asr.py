"""Optional faster-whisper scoring for completed validation renders."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

from .alignment import align_words, character_error_rate
from .jobs import load_run, selected_jobs


ASR_SCHEMA_VERSION = 1
_LANGUAGE_HINTS = {
    "en": "en",
    "en_us": "en",
    "en_gb": "en",
    "es": "es",
    "it": "it",
    "de": "de",
    "fr": "fr",
    "vi": "vi",
}


class ValidationASRError(RuntimeError):
    """Raised when the optional ASR evaluation cannot run."""


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ValidationASRError(
                "Install the public validation extra with `pip install -e '.[validation]'`"
            ) from exc
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, path: Path, *, language: str) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            str(path),
            language=_LANGUAGE_HINTS.get(language, language.split("_", 1)[0]),
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        text_parts: list[str] = []
        segment_records: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                text_parts.append(text)
            segment_words: list[dict[str, Any]] = []
            for word in getattr(segment, "words", ()) or ():
                record = {
                    "word": str(getattr(word, "word", "") or "").strip(),
                    "start": float(getattr(word, "start", 0.0) or 0.0),
                    "end": float(getattr(word, "end", 0.0) or 0.0),
                    "probability": float(getattr(word, "probability", 0.0) or 0.0),
                }
                words.append(record)
                segment_words.append(record)
            segment_records.append(
                {
                    "start": float(getattr(segment, "start", 0.0) or 0.0),
                    "end": float(getattr(segment, "end", 0.0) or 0.0),
                    "text": text,
                    "words": segment_words,
                }
            )
        return {
            "text": " ".join(text_parts).strip(),
            "detected_language": str(getattr(info, "language", "") or ""),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "segments": segment_records,
            "words": words,
        }


def _score(reference: str, transcription: Mapping[str, Any], language: str) -> dict[str, Any]:
    hypothesis = str(transcription.get("text") or "")
    alignment = align_words(reference, hypothesis, language=language)
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "cer": character_error_rate(reference, hypothesis, language=language),
        "alignment": alignment,
        "transcription": dict(transcription),
    }


def _chunk_reference(job: Mapping[str, Any], chunk: Mapping[str, Any]) -> str:
    text = str(chunk.get("text") or chunk.get("source_text") or "")
    for replacement in job.get("spoken_replacements", ()):
        surface = str(replacement.get("surface") or "")
        spoken = str(replacement.get("spoken") or "")
        if surface and spoken:
            text = text.replace(surface, spoken)
    return text


def _score_job(
    *,
    transcriber: FasterWhisperTranscriber,
    run_root: Path,
    backend: str,
    job: Mapping[str, Any],
    scope: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    render_root = run_root / "renders" / backend / str(job["job_id"])
    metadata_path = render_root / "metadata.json"
    if not metadata_path.is_file():
        raise ValidationASRError(f"Completed render metadata is missing: {metadata_path}")
    render_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if render_metadata.get("state") != "complete":
        raise ValidationASRError(f"Render is not complete: {metadata_path}")
    result: dict[str, Any] = {
        "schema_version": ASR_SCHEMA_VERSION,
        "artifact_kind": "scyllasband_validation_asr",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "job_id": job["job_id"],
        "job_contract_sha256": job["job_contract_sha256"],
        "language": job["language"],
        "config": dict(config),
    }
    if scope in {"final", "both"}:
        transcription = transcriber.transcribe(render_root / "audio.wav", language=str(job["language"]))
        result["final"] = _score(str(job["asr_reference_text"]), transcription, str(job["language"]))
    if scope in {"chunks", "both"}:
        chunk_results = []
        for chunk in render_metadata.get("chunks", ()):
            chunk_path = render_root / str(chunk["path"])
            transcription = transcriber.transcribe(chunk_path, language=str(job["language"]))
            chunk_results.append(
                {
                    "index": int(chunk["index"]),
                    "path": str(chunk["path"]),
                    **_score(_chunk_reference(job, chunk), transcription, str(job["language"])),
                }
            )
        result["chunks"] = chunk_results
        reference_words = sum(item["alignment"]["reference_words"] for item in chunk_results)
        errors = sum(item["alignment"]["errors"] for item in chunk_results)
        tolerant_errors = sum(item["alignment"]["format_tolerant_errors"] for item in chunk_results)
        result["chunk_aggregate"] = {
            "reference_words": reference_words,
            "errors": errors,
            "strict_wer": errors / float(reference_words) if reference_words else 0.0,
            "format_tolerant_errors": tolerant_errors,
            "format_tolerant_wer": (
                tolerant_errors / float(reference_words) if reference_words else 0.0
            ),
        }
    return result


def run_asr(
    *,
    run_dir: str | Path,
    backends: Iterable[str] | None = None,
    model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 1,
    vad_filter: bool = True,
    workers: int = 1,
    scope: str = "both",
    voices: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    conditions: Iterable[str] | None = None,
    documents: Iterable[str] | None = None,
    job_ids: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if scope not in {"final", "chunks", "both"}:
        raise ValidationASRError(f"Unsupported ASR scope: {scope}")
    run_root = Path(run_dir).expanduser().resolve()
    _, all_jobs = load_run(run_root)
    jobs = selected_jobs(
        all_jobs,
        voices=voices,
        languages=languages,
        conditions=conditions,
        documents=documents,
        job_ids=job_ids,
    )
    selected_backends = [str(item) for item in backends or () if str(item)]
    if not selected_backends:
        renders_root = run_root / "renders"
        if not renders_root.is_dir():
            raise ValidationASRError(f"Validation run has no renders: {renders_root}")
        selected_backends = sorted(path.name for path in renders_root.iterdir() if path.is_dir())
    config = {
        "backend": "faster-whisper",
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "beam_size": int(beam_size),
        "vad_filter": bool(vad_filter),
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "scope": scope,
    }
    pending: list[tuple[str, dict[str, Any], Path]] = []
    skipped = 0
    for backend in selected_backends:
        for job in jobs:
            render_metadata = run_root / "renders" / backend / str(job["job_id"]) / "metadata.json"
            if not render_metadata.is_file():
                continue
            target = run_root / "asr" / backend / f"{job['job_id']}.json"
            if target.is_file() and not overwrite:
                try:
                    existing = json.loads(target.read_text(encoding="utf-8"))
                    if (
                        existing.get("job_contract_sha256") == job.get("job_contract_sha256")
                        and existing.get("config") == config
                    ):
                        skipped += 1
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            pending.append((backend, job, target))

    thread_state = threading.local()

    def transcriber() -> FasterWhisperTranscriber:
        value = getattr(thread_state, "transcriber", None)
        if value is None:
            value = FasterWhisperTranscriber(
                model=model,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            thread_state.transcriber = value
        return value

    def work(item: tuple[str, dict[str, Any], Path]) -> tuple[Path, dict[str, Any]]:
        backend, job, target = item
        return target, _score_job(
            transcriber=transcriber(),
            run_root=run_root,
            backend=backend,
            job=job,
            scope=scope,
            config=config,
        )

    completed = failed = 0
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(work, item): item for item in pending}
        for future in as_completed(futures):
            backend, job, target = futures[future]
            try:
                output_path, result = future.result()
                _write_json_atomic(output_path, result)
                completed += 1
                print(f"[asr] {completed}/{len(pending)} {backend}/{job['job_id']}", flush=True)
            except Exception as exc:
                failed += 1
                errors.append({"backend": backend, "job_id": str(job["job_id"]), "error": str(exc)})
    return {
        "backends": selected_backends,
        "selected_jobs": len(jobs),
        "pending": len(pending),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "config": config,
    }
