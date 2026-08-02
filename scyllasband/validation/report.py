"""Aggregate validation artifacts and build a relocatable HTML review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
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
                    "metadata_path": str(metadata_path),
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scylla's Band long-form validation</title>
<style>
:root{color-scheme:dark;--bg:#11151b;--panel:#1b222c;--line:#364353;--ink:#eef4fb;--muted:#a9b4c2;--accent:#64d8cb;--bad:#ff7474;--warn:#ffbf69;--ins:#c59cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:18px}.toolbar{position:sticky;top:0;z-index:5;background:rgba(17,21,27,.97);border-bottom:1px solid var(--line);padding:12px 0}.filters{display:flex;gap:8px;flex-wrap:wrap}select,button,textarea,input{font:inherit;color:var(--ink);background:#222c38;border:1px solid var(--line);border-radius:7px;padding:7px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}.metric,.job{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}.metric strong{display:block;font-size:1.35rem}.job{margin:12px 0}.job-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.grow{flex:1}.badge{background:#283545;border-radius:999px;padding:3px 8px}.audio-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}.render{background:#151b23;border:1px solid #2e3947;border-radius:8px;padding:10px}audio{width:100%}.diff{white-space:pre-wrap;padding:10px;background:#111820;border-radius:7px}.op-substitute{background:#735515}.op-delete{background:#6b252b;text-decoration:line-through}.op-insert{background:#4c3670}.challenges{color:var(--warn)}.muted{color:var(--muted)}.bad{color:var(--bad)}table{width:100%;border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid var(--line);text-align:left}th{position:sticky;top:74px;background:#18202a}.review{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:10px}.hidden{display:none}@media(max-width:700px){main{padding:9px}th{position:static}}
</style>
</head>
<body><main>
<div class="toolbar"><h1>Scylla's Band long-form validation</h1><div class="filters">
<select id="backend"><option value="">All backends</option></select><select id="voice"><option value="">All voices</option></select><select id="language"><option value="">All languages</option></select><select id="condition"><option value="">All conditions</option></select><select id="document"><option value="">All documents</option></select><select id="reviewState"><option value="">All review states</option><option>unreviewed</option><option>reviewed</option></select><button id="next">Next unreviewed</button><button id="export">Export labels</button></div></div>
<div id="metrics" class="cards"></div><div id="summary"></div><div id="jobs"></div>
</main><script>const DATA=""" + embedded + r""";
const rows=DATA.rows,stateKey=`scyllasband_validation_${DATA.run.run_contract_sha256}`,state=JSON.parse(localStorage.getItem(stateKey)||'{}');
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(id,key){const s=$(id),values=[...new Set(rows.map(r=>r[key]))].sort();for(const v of values){const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o)}}
for(const [id,key] of [['backend','backend'],['voice','voice'],['language','language'],['condition','condition'],['document','document_id']])options(id,key);
function selected(){return rows.filter(r=>(!$('backend').value||r.backend===$('backend').value)&&(!$('voice').value||r.voice===$('voice').value)&&(!$('language').value||r.language===$('language').value)&&(!$('condition').value||r.condition===$('condition').value)&&(!$('document').value||r.document_id===$('document').value)&&(!$('reviewState').value||($('reviewState').value==='reviewed')===!!state[r.job_id]))}
function fmt(v,n=3){return v==null?'—':Number(v).toFixed(n)}
function renderOps(ops){return (ops||[]).map(o=>`<span class="op-${o.operation}" title="${esc(o.operation)}">${esc(o.reference??'')} ${o.operation==='substitute'?'→ '+esc(o.hypothesis):o.operation==='insert'?'+ '+esc(o.hypothesis):''}</span>`).join(' ')}
function reviewControls(job){const v=state[job.job_id]||{};return `<div class="review"><label>Intelligibility<select data-field="intelligibility"><option></option><option>pass</option><option>minor</option><option>fail</option></select></label><label>Affect 0–4<select data-field="affect"><option></option>${[0,1,2,3,4].map(x=>`<option>${x}</option>`).join('')}</select></label><label>Identity<select data-field="identity"><option></option><option>stable</option><option>slight drift</option><option>wrong voice</option></select></label><label>Continuity<select data-field="continuity"><option></option><option>pass</option><option>awkward</option><option>fail</option></select></label><label>Parity<select data-field="parity"><option></option><option>equivalent</option><option>left better</option><option>right better</option><option>both fail</option></select></label><label>ASR<select data-field="asr"><option></option><option>real TTS error</option><option>ASR false positive</option><option>normalization problem</option><option>uncertain</option></select></label><label>Artifacts<input data-field="artifacts" value="${esc(v.artifacts||'')}" placeholder="stutter, chirp, cutoff..."></label><label>Notes<input data-field="notes" value="${esc(v.notes||'')}"></label></div>`}
function render(){const filtered=selected(),byJob=new Map();for(const r of filtered){if(!byJob.has(r.job_id))byJob.set(r.job_id,[]);byJob.get(r.job_id).push(r)}const asr=filtered.filter(r=>r.strict_wer!=null),words=asr.reduce((a,r)=>a+r.reference_words,0),errors=asr.reduce((a,r)=>a+r.errors,0);$('metrics').innerHTML=`<div class="metric"><span>Visible renders</span><strong>${filtered.length}</strong></div><div class="metric"><span>Visible jobs</span><strong>${byJob.size}</strong></div><div class="metric"><span>Micro WER</span><strong>${words?fmt(errors/words):'—'}</strong></div><div class="metric"><span>Reviewed</span><strong>${[...byJob.keys()].filter(id=>state[id]).length}</strong></div>`;$('jobs').innerHTML=[...byJob.entries()].map(([id,items])=>{const first=items[0],challenge=(first.challenge_spans||[]).map(x=>`${x.surface} [${x.tags.join(', ')}]`).join(' · ');return `<article class="job" data-id="${esc(id)}"><div class="job-head"><strong>${esc(first.voice)} / ${esc(first.language)} / ${esc(first.condition)}</strong><span class="badge">${esc(first.document_id)}</span><span class="grow"></span><span>${state[id]?'reviewed':'unreviewed'}</span></div><p>${esc(first.text)}</p>${challenge?`<p class="challenges">Challenges: ${esc(challenge)}</p>`:''}<div class="audio-grid">${items.map(r=>`<div class="render"><strong>${esc(r.backend)}</strong> · WER ${fmt(r.strict_wer)} · chunk WER ${fmt(r.chunk_strict_wer)} · ${fmt(r.duration_seconds,1)}s<audio controls preload="none" src="${esc(r.audio_path)}"></audio><details><summary>ASR diff</summary><div class="diff">${renderOps(r.operations)}</div><p class="muted">${esc(r.hypothesis)}</p></details></div>`).join('')}</div>${reviewControls(first)}</article>`}).join('');for(const article of document.querySelectorAll('.job')){const id=article.dataset.id,v=state[id]||{};for(const input of article.querySelectorAll('[data-field]')){input.value=v[input.dataset.field]??'';input.onchange=input.oninput=()=>{const next={};for(const x of article.querySelectorAll('[data-field]'))if(x.value)next[x.dataset.field]=x.value;if(Object.keys(next).length)state[id]=next;else delete state[id];localStorage.setItem(stateKey,JSON.stringify(state));}}}}
for(const id of ['backend','voice','language','condition','document','reviewState'])$(id).onchange=render;$('next').onclick=()=>{const item=[...document.querySelectorAll('.job')].find(x=>!state[x.dataset.id]);if(item)item.scrollIntoView({behavior:'smooth',block:'start'})};$('export').onclick=()=>{const payload={schema_version:1,artifact_kind:'scyllasband_long_form_validation_review',run_contract_sha256:DATA.run.run_contract_sha256,exported_at:new Date().toISOString(),decisions:state},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='scyllasband_validation_review_labels.json';a.click();URL.revokeObjectURL(a.href)};render();
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
