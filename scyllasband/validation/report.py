"""Aggregate validation artifacts and build a relocatable HTML review."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
from typing import Any, Iterable, Mapping

from .jobs import load_run


REPORT_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, percentile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _asr_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "strict_wer": None,
            "format_tolerant_wer": None,
            "cer": None,
            "reference_words": 0,
            "errors": 0,
            "hypothesis": "",
            "operations": [],
            "chunk_strict_wer": None,
        }
    final = payload.get("final") if isinstance(payload.get("final"), Mapping) else {}
    alignment = final.get("alignment") if isinstance(final.get("alignment"), Mapping) else {}
    chunks = payload.get("chunk_aggregate") if isinstance(payload.get("chunk_aggregate"), Mapping) else {}
    return {
        "strict_wer": alignment.get("strict_wer"),
        "format_tolerant_wer": alignment.get("format_tolerant_wer"),
        "cer": final.get("cer"),
        "reference_words": int(alignment.get("reference_words") or 0),
        "errors": int(alignment.get("errors") or 0),
        "hypothesis": str(final.get("hypothesis") or ""),
        "operations": list(alignment.get("operations", ()) or ()),
        "chunk_strict_wer": chunks.get("strict_wer"),
    }


def collect_rows(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(run_dir).expanduser().resolve()
    run, jobs = load_run(root)
    job_by_id = {str(job["job_id"]): job for job in jobs}
    rows: list[dict[str, Any]] = []
    renders_root = root / "renders"
    if not renders_root.is_dir():
        return run, rows
    for backend_dir in sorted(path for path in renders_root.iterdir() if path.is_dir()):
        backend = backend_dir.name
        for metadata_path in sorted(backend_dir.glob("*/metadata.json")):
            render = _read_json(metadata_path)
            if not render or render.get("state") != "complete":
                continue
            job_id = str(render.get("job_id") or metadata_path.parent.name)
            job = job_by_id.get(job_id)
            if job is None:
                continue
            asr = _read_json(root / "asr" / backend / f"{job_id}.json")
            metrics = _asr_metrics(asr)
            audio = render.get("audio_metrics") if isinstance(render.get("audio_metrics"), Mapping) else {}
            rows.append(
                {
                    "backend": backend,
                    "job_id": job_id,
                    "job_contract_sha256": job["job_contract_sha256"],
                    "voice": job["voice"],
                    "language": job["language"],
                    "condition": job["condition"],
                    "ratings": job.get("ratings", {}),
                    "affect": job.get("affect"),
                    "document_id": job["document_id"],
                    "document_kind": job["document_kind"],
                    "training_overlap": job["training_overlap"],
                    "text": job["text"],
                    "asr_reference_text": job["asr_reference_text"],
                    "challenge_spans": job.get("challenge_spans", []),
                    "audio_path": (
                        Path("..")
                        / "renders"
                        / backend
                        / job_id
                        / "audio.wav"
                    ).as_posix(),
                    "metadata_path": (
                        Path("..")
                        / "renders"
                        / backend
                        / job_id
                        / "metadata.json"
                    ).as_posix(),
                    "audio_sha256": render.get("audio_sha256"),
                    "sample_rate": render.get("sample_rate"),
                    "duration_seconds": float(render.get("duration_seconds") or 0.0),
                    "elapsed_seconds": float(render.get("elapsed_seconds") or 0.0),
                    "realtime_factor": render.get("realtime_factor"),
                    "chunk_count": len(render.get("chunks", ()) or ()),
                    "plan_signature": render.get("plan_signature"),
                    "audio_metrics": dict(audio),
                    **metrics,
                }
            )
    return run, rows


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    output = []
    for values, members in sorted(groups.items()):
        wer_values = [float(row["strict_wer"]) for row in members if row.get("strict_wer") is not None]
        total_words = sum(int(row.get("reference_words") or 0) for row in members)
        total_errors = sum(int(row.get("errors") or 0) for row in members)
        durations = [float(row.get("duration_seconds") or 0.0) for row in members]
        record = {key: value for key, value in zip(keys, values)}
        record.update(
            {
                "renders": len(members),
                "asr_renders": len(wer_values),
                "reference_words": total_words,
                "errors": total_errors,
                "micro_wer": total_errors / float(total_words) if total_words else None,
                "macro_wer": mean(wer_values) if wer_values else None,
                "median_wer": median(wer_values) if wer_values else None,
                "p90_wer": _percentile(wer_values, 0.90),
                "p95_wer": _percentile(wer_values, 0.95),
                "max_wer": max(wer_values) if wer_values else None,
                "audio_seconds": sum(durations),
                "invalid_audio": sum(
                    not bool(row.get("audio_metrics", {}).get("finite", True)) for row in members
                ),
                "clipped_renders": sum(
                    int(row.get("audio_metrics", {}).get("hard_clipped_sample_count") or 0) > 0
                    for row in members
                ),
            }
        )
        output.append(record)
    return output


def build_summary(run: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = (
        ("backend",),
        ("backend", "voice"),
        ("backend", "language"),
        ("backend", "condition"),
        ("backend", "document_kind"),
        ("backend", "voice", "language", "condition"),
    )
    aggregates = {"__".join(keys): _aggregate(rows, keys) for keys in dimensions}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": "scyllasband_long_form_validation_summary",
        "run_contract_sha256": run.get("run_contract_sha256"),
        "model_name": run.get("model_name"),
        "model_version": run.get("model_version"),
        "expected_jobs": run.get("job_count"),
        "render_rows": len(rows),
        "backends": sorted({str(row["backend"]) for row in rows}),
        "aggregates": aggregates,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "backend",
        "job_id",
        "voice",
        "language",
        "condition",
        "document_id",
        "document_kind",
        "strict_wer",
        "format_tolerant_wer",
        "cer",
        "reference_words",
        "errors",
        "chunk_strict_wer",
        "duration_seconds",
        "realtime_factor",
        "chunk_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _html(data: Mapping[str, Any]) -> str:
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="description" content="Scylla's Band voice, language, affect, and long-form intelligibility benchmark.">
<title>Scylla's Band Voice Benchmark</title>
<style>
:root{color-scheme:light;--ink:#113452;--muted:#526271;--paper:#fff;--wash:#fff7e8;--panel:#fffdf8;--line:#173b59;--line-soft:rgba(17,52,82,.18);--red:#c83b2d;--coral:#ed6241;--orange:#f29a59;--apricot:#ffc375;--cream:#f2dfb2;--teal:#0c7d86;--seafoam:#9bcfc3;--shadow:6px 7px 0 rgba(17,52,82,.09);--bad:#a4231a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:linear-gradient(180deg,var(--wash) 0,var(--paper) 31rem);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--red);text-decoration-thickness:1.5px;text-underline-offset:.18em}a:hover{color:var(--teal)}.shell{width:min(1460px,calc(100% - 32px));margin-inline:auto}.hero{padding:54px 0 34px}.hero:before{content:"";display:block;width:min(460px,78vw);height:14px;margin-bottom:28px;border:1px solid var(--line);border-radius:999px;background:linear-gradient(90deg,var(--red) 0 20%,var(--coral) 20% 40%,var(--orange) 40% 60%,var(--apricot) 60% 76%,var(--seafoam) 76% 90%,var(--teal) 90% 100%);box-shadow:4px 4px 0 var(--cream)}.eyebrow{margin:0 0 8px;color:var(--red);font-weight:850;letter-spacing:.14em;text-transform:uppercase}.hero h1{max-width:980px;margin:0;color:var(--ink);font-size:clamp(3rem,7vw,6.2rem);font-weight:850;line-height:.9;letter-spacing:-.065em;text-shadow:3px 3px 0 var(--cream)}.lede{max-width:850px;margin:24px 0 0;color:var(--muted);font-size:clamp(1.03rem,2vw,1.25rem)}.hero-links{display:flex;flex-wrap:wrap;gap:16px;margin-top:20px}.hero-links a{font-weight:800;text-decoration:none}.hero-links a:hover{text-decoration:underline}main{padding:0 0 80px}.status{display:flex;align-items:center;gap:12px;margin:4px 0 16px;padding:13px 16px;border:1.5px solid var(--line);border-left:8px solid var(--teal);border-radius:12px;background:var(--panel);box-shadow:var(--shadow)}.status.partial{border-left-color:var(--orange)}.status strong{font-size:1.05rem}.status span{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}.metric,.job,.summary-card{border:1.5px solid var(--line);border-radius:16px;background:var(--panel);box-shadow:var(--shadow)}.metric{padding:15px;border-top:6px solid var(--red)}.metric:nth-child(2){border-top-color:var(--orange)}.metric:nth-child(3){border-top-color:var(--teal)}.metric:nth-child(4){border-top-color:var(--seafoam)}.metric span{color:var(--muted);font-size:.82rem;font-weight:750;text-transform:uppercase;letter-spacing:.06em}.metric strong{display:block;margin-top:2px;color:var(--ink);font-size:1.65rem}.section-title{display:inline-block;margin:42px 0 16px;border-bottom:6px solid var(--apricot);font-size:clamp(1.8rem,3vw,2.7rem);letter-spacing:-.04em}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}.summary-card{overflow:auto;padding:14px;border-top:6px solid var(--teal)}.summary-card:nth-child(2){border-top-color:var(--orange)}.summary-card h3{margin:0 0 10px;font-size:1.15rem}.summary-note{max-width:900px;margin:0 0 16px;color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid var(--line-soft);text-align:left;white-space:nowrap}th{color:var(--red);font-size:.75rem;letter-spacing:.06em;text-transform:uppercase}.job{margin:16px 0;padding:17px;border-top:6px solid var(--orange)}.job:nth-child(3n){border-top-color:var(--teal)}.job:nth-child(3n+1){border-top-color:var(--red)}.job-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.job-head strong{font-size:1.08rem}.grow{flex:1}.badge{padding:3px 9px;border:1px solid var(--line-soft);border-radius:999px;background:var(--wash);font-size:.8rem;font-weight:750}.field-label{margin:16px 0 5px;color:var(--red);font-size:.76rem;font-weight:850;letter-spacing:.09em;text-transform:uppercase}.script,.transcript{margin:0;color:var(--ink);font-size:1rem}.audio-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:14px}.render{padding:12px;border:1px solid var(--line-soft);border-radius:12px;background:var(--paper)}.render-meta{margin-bottom:8px}.wer-bad{color:var(--bad);font-weight:850}.wer-ok{color:var(--teal);font-weight:850}.asr-pending{color:var(--orange);font-weight:850}audio{display:block;width:100%;height:42px;accent-color:var(--red)}audio::-webkit-media-controls-panel{background-color:var(--cream)}.diff{margin-top:5px;padding:10px;border-radius:9px;background:var(--wash);white-space:pre-wrap}.op-substitute{background:#ffe0a9}.op-delete{background:#ffd0c9;text-decoration:line-through}.op-insert{background:#d7eee7}.challenges{color:var(--teal);font-weight:700}footer{padding:24px 0 42px;border-top:5px solid var(--cream);color:var(--muted)}@media(max-width:700px){.shell{width:min(100% - 20px,1460px)}.hero{padding-top:34px}.summary-grid{grid-template-columns:1fr}.audio-grid{grid-template-columns:1fr}.status{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<header class="hero shell"><p class="eyebrow">Measured release listening</p><h1>Scylla's Band<br>Voice Benchmark</h1><p class="lede">Long-form intelligibility, voice, language, and affect results. WER is an ASR signal—not a substitute for listening—so each result presents the source script, generated audio, ASR transcript, and word-level diff together.</p><div class="hero-links"><a href="https://lowkeytea.github.io/scyllasband/index.html">Voice gallery &rarr;</a><a href="https://github.com/lowkeytea/scyllasband/tree/main/benchmark">Benchmark data &rarr;</a></div></header>
<main class="shell"><div id="status"></div><div id="metrics" class="cards"></div><h2 class="section-title">Release findings</h2><p class="summary-note">Micro WER weights each spoken word equally. P90 shows the difficult tail. Listen alongside the displayed script and ASR comparison when interpreting these measurements.</p><div id="summary" class="summary-grid"></div><h2 class="section-title">Audio and transcripts</h2><div id="jobs"></div></main>
<footer><div class="shell">Scylla's Band by <strong>Spybyscript</strong>. Benchmark audio and results use the frozen public validation corpus.</div></footer>
<script>const DATA=""" + embedded + r""";
const rows=DATA.rows;
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(v,n=3){return v==null?'—':Number(v).toFixed(n)}
function pct(v){return v==null?'—':`${(Number(v)*100).toFixed(2)}%`}
function seconds(v){if(v==null)return'—';const m=Math.floor(Number(v)/60),s=Math.round(Number(v)%60);return m?`${m}m ${s}s`:`${s}s`}
function renderOps(ops){return (ops||[]).map(o=>`<span class="op-${o.operation}" title="${esc(o.operation)}">${esc(o.reference??'')} ${o.operation==='substitute'?'→ '+esc(o.hypothesis):o.operation==='insert'?'+ '+esc(o.hypothesis):''}</span>`).join(' ')}
function aggregateTable(key,label){const values=DATA.summary.aggregates[key]||[],dimension=key.split('__').at(-1);return `<section class="summary-card"><h3>${esc(label)}</h3><table><thead><tr><th>${esc(dimension)}</th><th>clips</th><th>words</th><th>micro WER</th><th>P90</th></tr></thead><tbody>${values.map(x=>`<tr><td>${esc(x[dimension])}</td><td>${x.renders}</td><td>${x.reference_words}</td><td>${pct(x.micro_wer)}</td><td>${pct(x.p90_wer)}</td></tr>`).join('')}</tbody></table></section>`}
function render(){const byJob=new Map();for(const r of rows){if(!byJob.has(r.job_id))byJob.set(r.job_id,[]);byJob.get(r.job_id).push(r)}const asr=rows.filter(r=>r.strict_wer!=null),words=asr.reduce((a,r)=>a+r.reference_words,0),errors=asr.reduce((a,r)=>a+r.errors,0),expected=Number(DATA.summary.expected_jobs||0),allJobs=byJob.size,complete=expected>0&&allJobs>=expected,modelVersion=esc(DATA.run.model_version||'unknown'),overall=(DATA.summary.aggregates.backend||[])[0]||{};$('status').innerHTML=`<div class="status ${complete?'':'partial'}"><strong>${complete?'Complete release matrix':'Benchmark in progress'}</strong><span>Model v${modelVersion} · ${allJobs} of ${expected||'—'} planned jobs published · ${esc((DATA.summary.backends||[]).join(', ')||'no backend')}</span></div>`;$('metrics').innerHTML=`<div class="metric"><span>Renders</span><strong>${rows.length}</strong></div><div class="metric"><span>Spoken words</span><strong>${words.toLocaleString()}</strong></div><div class="metric"><span>Micro WER</span><strong>${words?pct(errors/words):'—'}</strong></div><div class="metric"><span>P95 WER</span><strong>${pct(overall.p95_wer)}</strong></div>`;$('summary').innerHTML=aggregateTable('backend__voice','WER by voice')+aggregateTable('backend__language','WER by language');$('jobs').innerHTML=[...byJob.entries()].map(([id,items])=>{const first=items[0],challenge=(first.challenge_spans||[]).map(x=>`${x.surface} [${x.tags.join(', ')}]`).join(' · ');return `<article class="job" data-id="${esc(id)}"><div class="job-head"><strong>${esc(first.voice)} / ${esc(first.language)} / ${esc(first.condition)}</strong><span class="badge">${esc(first.document_id)}</span></div><p class="field-label">Script</p><p class="script">${esc(first.text)}</p>${challenge?`<p class="challenges">Challenges: ${esc(challenge)}</p>`:''}<div class="audio-grid">${items.map(r=>{const werClass=r.strict_wer==null?'asr-pending':r.strict_wer<=.03?'wer-ok':'wer-bad',wer=r.strict_wer==null?'ASR pending':`WER ${pct(r.strict_wer)}`;return `<div class="render"><div class="render-meta"><strong>${esc(r.backend)}</strong> · <span class="${werClass}">${wer}</span> · chunk ${pct(r.chunk_strict_wer)} · ${seconds(r.duration_seconds)}</div><audio controls preload="none" src="${esc(r.audio_path)}"></audio><p class="field-label">ASR transcript</p><p class="transcript">${esc(r.hypothesis||'ASR has not been run for this clip.')}</p><p class="field-label">ASR diff</p><div class="diff">${renderOps(r.operations)||'No word edits.'}</div></div>`}).join('')}</div></article>`}).join('')}
render();
</script></body></html>"""


def build_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    run, rows = collect_rows(root)
    summary = build_summary(run, rows)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(root / "summary.csv", rows)
    payload = {"run": run, "summary": summary, "rows": rows}
    (report_dir / "index.html").write_text(_html(payload), encoding="utf-8")
    return {
        "rows": len(rows),
        "backends": summary["backends"],
        "summary": str(root / "summary.json"),
        "csv": str(root / "summary.csv"),
        "html": str(report_dir / "index.html"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_run(run: Mapping[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(run, ensure_ascii=False))
    public.pop("bundle_dir", None)
    public["suite_path"] = "data/testing/long_form/suite.json"
    return public


def _encode_public_audio(source: Path, target: Path, audio_format: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists():
        temporary.unlink()
    if audio_format == "wav":
        shutil.copy2(source, temporary)
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to publish MP3 benchmark audio")
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map_metadata",
                "-1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(temporary),
            ],
            check=True,
        )
    os.replace(temporary, target)


def _sync_publication(source: Path, destination: Path, *, overwrite: bool) -> None:
    if source.resolve() == destination.resolve():
        return
    marker = destination / "benchmark.json"
    if destination.exists() and any(destination.iterdir()) and not marker.is_file():
        raise RuntimeError(f"Refusing to replace non-benchmark Pages directory: {destination}")
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def publish_benchmark(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    pages_dir: str | Path | None = None,
    audio_format: str = "mp3",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish a compact, relocatable benchmark from a validation work directory."""

    if audio_format not in {"mp3", "wav"}:
        raise ValueError(f"Unsupported benchmark audio format: {audio_format}")
    root = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    run, rows = collect_rows(root)
    if not rows:
        raise RuntimeError(f"No completed renders are available to publish from {root}")

    existing = _read_json(output / "benchmark.json")
    if existing and existing.get("run_contract_sha256") != run.get("run_contract_sha256"):
        if not overwrite:
            raise RuntimeError(
                f"Benchmark output belongs to a different frozen run: {output}; use --overwrite"
            )
        shutil.rmtree(output)
        existing = None
    output.mkdir(parents=True, exist_ok=True)

    previous_results = _read_json(output / "results.json") or {}
    previous_rows = {
        (str(item.get("backend")), str(item.get("job_id"))): item
        for item in previous_results.get("rows", ())
        if isinstance(item, Mapping)
    }
    public_rows: list[dict[str, Any]] = []
    encoded = reused = 0
    suffix = ".mp3" if audio_format == "mp3" else ".wav"
    for row in rows:
        source = root / "renders" / str(row["backend"]) / str(row["job_id"]) / "audio.wav"
        source_sha = _file_sha256(source)
        relative = Path("audio") / str(row["backend"]) / f"{row['job_id']}{suffix}"
        target = output / relative
        previous = previous_rows.get((str(row["backend"]), str(row["job_id"])), {})
        can_reuse = (
            not overwrite
            and target.is_file()
            and previous.get("source_audio_sha256") == source_sha
            and previous.get("audio_format") == audio_format
            and previous.get("published_audio_sha256") == _file_sha256(target)
        )
        if can_reuse:
            reused += 1
        else:
            _encode_public_audio(source, target, audio_format)
            encoded += 1
        public_row = dict(row)
        public_row.pop("metadata_path", None)
        public_row["audio_path"] = relative.as_posix()
        public_row["audio_format"] = audio_format
        public_row["source_audio_sha256"] = source_sha
        public_row["published_audio_sha256"] = _file_sha256(target)
        public_rows.append(public_row)

    public_run = _public_run(run)
    summary = build_summary(public_run, public_rows)
    published_jobs = len({str(row["job_id"]) for row in public_rows})
    expected_jobs = int(summary.get("expected_jobs") or 0)
    publication = {
        "schema_version": 1,
        "artifact_kind": "scyllasband_public_voice_benchmark",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "run_contract_sha256": run.get("run_contract_sha256"),
        "model_name": run.get("model_name"),
        "model_version": run.get("model_version"),
        "suite_id": run.get("suite_id"),
        "suite_sha256": run.get("suite_sha256"),
        "tier": run.get("tier"),
        "expected_jobs": expected_jobs,
        "published_jobs": published_jobs,
        "complete": expected_jobs > 0 and published_jobs >= expected_jobs,
        "backends": summary["backends"],
        "audio_format": audio_format,
    }
    payload = {
        "publication": publication,
        "run": public_run,
        "summary": summary,
        "rows": public_rows,
    }
    (output / "benchmark.json").write_text(
        json.dumps(publication, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "summary.csv", public_rows)
    (output / "index.html").write_text(_html(payload), encoding="utf-8")

    pages_output = None
    if pages_dir is not None:
        pages_output = Path(pages_dir).expanduser().resolve()
        _sync_publication(output, pages_output, overwrite=overwrite)
    return {
        "output_dir": str(output),
        "pages_dir": str(pages_output) if pages_output else None,
        "published_jobs": published_jobs,
        "expected_jobs": expected_jobs,
        "complete": publication["complete"],
        "encoded_audio": encoded,
        "reused_audio": reused,
        "audio_format": audio_format,
        "html": str(output / "index.html"),
    }
