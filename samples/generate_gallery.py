#!/usr/bin/env python3
"""Render the voice gallery (docs/index.html and docs/audio/) and samples/README.md from samples/gallery_script.json.

    python samples/generate_gallery.py --bundle scyllasband/models/litert

Every clip goes through the public runtime with the script's seed, sampler and steps. MP3 encoding uses ffmpeg.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scyllasband import ScyllasBandRuntime, SynthesisRequest  # noqa: E402
from scyllasband.cli import write_wav  # noqa: E402
from scyllasband.contract import validate_bundle_layout  # noqa: E402
from scyllasband.planner import parse_group_lines, records_from_text  # noqa: E402

GALLERY_URL = "https://lowkeytea.github.io/scyllasband/"
REPO_URL = "https://github.com/lowkeytea/scyllasband"
MODEL_URL = "https://huggingface.co/spybyscript/scyllasband"
LOCALE_NAMES = {"en_us": "English (US)", "en_gb": "English (UK)", "es": "Spanish", "it": "Italian", "fr": "French",
                "de": "German", "vi": "Vietnamese"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "scyllasband" / "models" / "litert")
    parser.add_argument("--script", type=Path, default=REPO_ROOT / "samples" / "gallery_script.json")
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bitrate", default="96k")
    args = parser.parse_args()

    script = json.loads(args.script.read_text(encoding="utf-8"))
    runtime = ScyllasBandRuntime.from_bundle(args.bundle, threads=args.threads)
    audio_dir = args.docs / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    options = dict(seed=script["seed"], steps=script["steps"], sampler=script["sampler"])

    def save(name: str, audio, sample_rate: int) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            write_wav(wav, audio, sample_rate)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", args.bitrate,
                            str(audio_dir / f"{name}.mp3")], check=True)
        print(name, flush=True)
        return dict(src=f"audio/{name}.mp3", seconds=len(audio) / sample_rate)

    voices = []
    for voice in runtime.available_voices():
        spec = runtime.voice_for_id(voice)
        english = spec.default_language
        languages = []
        for language in script["languages"]:
            locale = english if language["id"] == "en" else language["id"]
            result = runtime.synthesize(SynthesisRequest(text=language["line"], voice_id=voice, language=locale, **options))
            languages.append(dict(language, locale=locale, **save(f"{voice}_{language['id']}", result.audio, result.sample_rate)))
        deliveries = []
        for demo in script["deliveries"]:
            result = runtime.synthesize(SynthesisRequest(text=demo["line"], voice_id=voice, language=english,
                                                         delivery=demo["delivery"], **options))
            deliveries.append(dict(demo, **save(f"{voice}_{demo['id']}", result.audio, result.sample_rate)))
        voices.append(dict(id=voice, english=english, languages=languages, deliveries=deliveries))

    long_form = []
    for item in script["long_form"]:
        text = (REPO_ROOT / item["file"]).read_text(encoding="utf-8")
        if item.get("group"):
            records = parse_group_lines(text, default_voice=None, default_language=None)
        else:
            records = records_from_text(text, voice=item["voice"], language=item["language"], delivery=item["delivery"])
        result = runtime.render_records(records, pause_ms=item.get("pause_ms", 0.0), **options)
        voices_used = list(dict.fromkeys(r["voice"] for r in records))
        locales_used = list(dict.fromkeys(r["language"] for r in records))
        long_form.append(dict(item, words=len(re.findall(r"\w+", " ".join(r["text"] for r in records))), voices=voices_used,
                              locales=locales_used, **save(item["id"], result.audio, result.sample_rate)))

    produced = {Path(entry["src"]).name for v in voices for entry in v["languages"] + v["deliveries"]}
    produced |= {Path(item["src"]).name for item in long_form}
    for stale in audio_dir.glob("*.mp3"):   # clips dropped or renamed in the script
        if stale.name not in produced:
            stale.unlink()
    release = validate_bundle_layout(args.bundle)["release"]
    context = dict(script=script, voices=voices, long_form=long_form, backend=runtime.backend, release=release)
    (args.docs / "index.html").write_text(render_page(context), encoding="utf-8")
    (REPO_ROOT / "samples" / "README.md").write_text(render_readme(context), encoding="utf-8")
    (args.docs / ".nojekyll").touch()
    return 0


def _duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(round(seconds % 60)):02d}"


def _audio(src: str, label: str) -> str:
    return (f'<audio controls preload="none" src="{escape(src)}" aria-label="{escape(label)}"></audio>'
            f'<a class="audio-link" href="{escape(src)}">Open audio file</a>')


def render_page(ctx: dict) -> str:
    script, voices = ctx["script"], ctx["voices"]
    n_languages = sum(len(v["languages"]) for v in voices)
    n_deliveries = sum(len(v["deliveries"]) for v in voices)
    nav = "".join(f'<a href="#{v["id"]}">{escape(v["id"].title())}</a>' for v in voices)

    long_cards = []
    for item in ctx["long_form"]:
        who = (escape(item["voices"][0].title()) if len(item["voices"]) == 1 else f'{len(item["voices"])} voices')
        where = " · ".join(LOCALE_NAMES.get(l, l) for l in item["locales"])
        long_cards.append(f'''
        <article class="long-form-card" id="{item["id"]}">
          <div class="long-form-heading"><h3>{escape(item["title"])}</h3><span class="duration">{_duration(item["seconds"])}</span></div>
          <p class="long-form-meta">{who} · {escape(where)} · {item["words"]} words</p>
          <p class="line">{escape(item["summary"])}</p>
          {_audio(item["src"], item["title"])}
          <a class="source-link" href="{REPO_URL}/blob/main/{escape(item["file"])}">Read the source text &rarr;</a>
        </article>''')

    language_cards = []
    for v in voices:
        samples = "".join(f'''
            <div class="sample" id="{v["id"]}-{l["id"]}">
              <span class="sample-label">{escape(LOCALE_NAMES.get(l["locale"], l["name"]))}</span>
              {_audio(l["src"], f'{v["id"].title()} in {l["name"]}')}
            </div>''' for l in v["languages"])
        language_cards.append(f'''
        <article class="voice-card" id="{v["id"]}">
          <h3>{escape(v["id"].title())}</h3>{samples}
        </article>''')

    delivery_sections = []
    for v in voices:
        cards = "".join(f'''
            <article class="demo-card" id="{v["id"]}-{d["id"]}">
              <h4>{escape(d["title"])}</h4>
              <p class="delivery">{escape(d["delivery"].replace(",", ", "))}</p>
              <p class="line">{escape(d["line"])}</p>
              {_audio(d["src"], f'{v["id"].title()}, {d["title"].lower()} delivery')}
            </article>''' for d in v["deliveries"])
        delivery_sections.append(f'''
        <section class="voice-section">
          <h3>{escape(v["id"].title())} <span>{escape(LOCALE_NAMES[v["english"]])}</span></h3>
          <div class="demo-grid">{cards}
          </div>
        </section>''')

    language_names = ", ".join(l["name"] for l in script["languages"][:-1]) + " and " + script["languages"][-1]["name"]
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Voice, language and delivery samples for Scylla's Band text-to-speech.">
  <meta name="color-scheme" content="light">
  <title>Scylla's Band Voice Gallery</title>
  <style>{STYLE}</style>
</head>
<body>
  <header class="hero shell">
    <p class="eyebrow">Spybyscript presents</p>
    <h1>Scylla's Band<br>Voice Gallery</h1>
    <p class="lede">Ten voices speaking {escape(language_names)}, with delivery set by energy, tension, valence and
      assertiveness on 0–4 scales.</p>
    <div class="metrics" aria-label="Gallery summary">
      <span class="metric"><strong>{len(voices)}</strong> voices</span>
      <span class="metric"><strong>{len(script["languages"])}</strong> languages</span>
      <span class="metric"><strong>{n_languages}</strong> language clips</span>
      <span class="metric"><strong>{n_deliveries}</strong> delivery clips</span>
      <span class="metric"><strong>{len(ctx["long_form"])}</strong> long-form clips</span>
      <span class="metric"><strong>24 kHz</strong> output</span>
    </div>
    <div class="links">
      <a href="{REPO_URL}">GitHub repository &rarr;</a>
      <a href="{MODEL_URL}">Model on Hugging Face &rarr;</a>
    </div>
  </header>
  <nav class="shell" aria-label="Sample navigation">
    <a href="#long-form">Long form</a><a href="#languages">Languages</a><a href="#delivery">Delivery</a>{nav}
  </nav>
  <main class="shell">
    <p class="render-note">Every clip was generated on CPU with the default LiteRT bundle (release {escape(ctx["release"])}),
      {script["sampler"].title()} sampling with {script["steps"]} steps and seed {script["seed"]}. Audio loads when you press play.</p>
    <section id="long-form">
      <div class="section-heading">
        <h2>Long form</h2>
        <p>Whole texts, spoken as passages of up to about nine seconds. Each passage sees the text around it and continues from
          the sound of the one before it.</p>
      </div>
      <div class="long-form-grid">{"".join(long_cards)}
      </div>
    </section>
    <section id="languages">
      <div class="section-heading">
        <h2>Languages</h2>
        <p>Each voice reads the same line in its own English and in five more languages, with neutral delivery.</p>
      </div>
      <div class="voice-grid">{"".join(language_cards)}
      </div>
    </section>
    <section id="delivery">
      <div class="section-heading">
        <h2>Delivery</h2>
        <p>Each line was written for its delivery and uses the settings shown. Values run from 0 to 4 with 2 as neutral; the axes
          interact, and their useful range varies by voice.</p>
      </div>{"".join(delivery_sections)}
    </section>
  </main>
  <footer><div class="shell">Scylla's Band by <strong>Spybyscript</strong>. Released under Apache 2.0.</div></footer>
</body>
</html>
'''


def render_readme(ctx: dict) -> str:
    script, voices = ctx["script"], ctx["voices"]
    header = "| Voice | " + " | ".join(l["name"] for l in script["languages"]) + " |"
    rule = "| --- |" + " --- |" * len(script["languages"])
    rows = [f'| {v["id"].title()} | ' + " | ".join(f'[Listen]({GALLERY_URL}#{v["id"]}-{l["id"]})' for l in v["languages"]) + " |"
            for v in voices]
    deliveries = "\n".join(f'| {d["title"]} | `{d["delivery"]}` | {d["line"]} |' for d in script["deliveries"])
    long_form = "\n".join(f'- [{item["title"]}]({GALLERY_URL}#{item["id"]}) ({_duration(item["seconds"])}), from `{item["file"]}`'
                          for item in ctx["long_form"])
    return f"""# Scylla's Band Voice Samples

[Open the voice gallery]({GALLERY_URL}) to play every clip in the browser.

The clips use the default LiteRT bundle (release `{ctx["release"]}`), {script["sampler"]} sampling with {script["steps"]} steps and
seed `{script["seed"]}`. To regenerate them after downloading the model:

```bash
python samples/generate_gallery.py --bundle scyllasband/models/litert
```

The script reads `samples/gallery_script.json` and writes `docs/index.html`, `docs/audio/` and this file.

## Long form

{long_form}

## Languages

Each voice reads the same line in its own English and in five more languages.

{header}
{rule}
{chr(10).join(rows)}

## Delivery

Every voice speaks each line below in its own English, with the settings shown.

| Delivery | Settings | Line |
| --- | --- | --- |
{deliveries}
"""


STYLE = """
    :root {
      color-scheme: light;
      --ink: #113452; --muted: #526271; --paper: #ffffff; --wash: #fff7e8; --panel: #fffdf8;
      --line: #173b59; --line-soft: rgba(17, 52, 82, .18);
      --red: #c83b2d; --coral: #ed6241; --orange: #f29a59; --apricot: #ffc375; --cream: #f2dfb2;
      --teal: #0c7d86; --seafoam: #9bcfc3; --shadow: 6px 7px 0 rgba(17, 52, 82, .09);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0; color: var(--ink);
      background: linear-gradient(180deg, var(--wash) 0, var(--paper) 30rem) var(--paper);
      font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: var(--red); text-decoration-thickness: 1.5px; text-underline-offset: .18em; }
    a:hover { color: var(--teal); }
    .shell { width: min(1180px, calc(100% - 32px)); margin-inline: auto; }
    .hero { padding: 64px 0 42px; }
    .hero::before {
      content: ""; display: block; width: min(460px, 78vw); height: 14px; margin-bottom: 30px;
      border: 1px solid var(--line); border-radius: 999px;
      background: linear-gradient(90deg, var(--red) 0 20%, var(--coral) 20% 40%, var(--orange) 40% 60%,
        var(--apricot) 60% 76%, var(--seafoam) 76% 90%, var(--teal) 90% 100%);
      box-shadow: 4px 4px 0 var(--cream);
    }
    .eyebrow { margin: 0 0 10px; color: var(--red); font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    h1 {
      margin: 0; font-size: clamp(3rem, 8vw, 6.8rem); line-height: .9; letter-spacing: -.065em;
      font-weight: 850; text-shadow: 3px 3px 0 var(--cream); text-wrap: balance;
    }
    .lede { max-width: 780px; margin: 28px 0 0; color: var(--muted); font-size: clamp(1.05rem, 2vw, 1.28rem); }
    .metrics, .links, nav { display: flex; flex-wrap: wrap; gap: 10px; }
    .metrics, .links { margin-top: 24px; }
    .metric, nav a { border: 1.5px solid var(--line); border-radius: 999px; padding: 7px 12px; background: var(--panel); }
    .metric strong { color: var(--red); font-variant-numeric: tabular-nums; }
    .links a { font-weight: 800; text-decoration: none; }
    .links a:hover { text-decoration: underline; }
    nav {
      position: sticky; top: 0; z-index: 10; padding-block: 12px;
      border-top: 1px solid var(--line-soft); border-bottom: 3px solid var(--cream); border-radius: 0 0 14px 14px;
      background: rgba(255, 255, 255, .94); box-shadow: 0 4px 0 rgba(17, 52, 82, .05); backdrop-filter: blur(14px);
    }
    nav a { color: var(--ink); border-color: var(--line-soft); background: var(--paper); font-size: .86rem; font-weight: 750;
      text-decoration: none; }
    nav a:hover { color: var(--red); background: var(--wash); }
    main { padding-bottom: 80px; }
    .render-note { max-width: 780px; margin: 28px 0 0; color: var(--muted); font-size: .92rem; }
    section, article { scroll-margin-top: 84px; }
    .section-heading { margin: 62px 0 22px; }
    h2 { display: inline-block; margin: 0; border-bottom: 6px solid var(--apricot); font-size: clamp(2rem, 4vw, 3.4rem);
      letter-spacing: -.04em; }
    .section-heading p { max-width: 780px; margin: 8px 0 0; color: var(--muted); }
    .voice-grid, .demo-grid, .long-form-grid { display: grid; gap: 16px; }
    .voice-grid { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    .long-form-grid { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .demo-grid { grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
    .voice-card, .demo-card, .long-form-card { border: 1.5px solid var(--line); border-radius: 18px; background: var(--panel);
      box-shadow: var(--shadow); }
    .voice-card { padding: 20px; border-top: 6px solid var(--red); }
    .voice-card:nth-child(3n + 2) { border-top-color: var(--orange); }
    .voice-card:nth-child(3n) { border-top-color: var(--teal); }
    .voice-card h3 { margin: 0 0 6px; font-size: 1.45rem; }
    .demo-card { display: flex; flex-direction: column; padding: 18px; border-top: 5px solid var(--orange); }
    .demo-card h4 { margin: 0; font-size: 1.15rem; }
    .demo-card .line { flex: 1; }
    .long-form-card { display: flex; flex-direction: column; padding: 22px; border-top: 6px solid var(--teal); }
    .long-form-card .line { flex: 1; }
    .sample { margin-top: 14px; }
    .sample-label { display: block; margin-bottom: 5px; color: var(--muted); font-size: .85rem; font-weight: 700; }
    audio { display: block; width: 100%; height: 42px; accent-color: var(--red); }
    .audio-link { display: inline-block; margin-top: 5px; font-size: .78rem; }
    .voice-section { margin-top: 42px; }
    .voice-section h3 { margin: 0 0 14px; font-size: 1.45rem; }
    .voice-section h3 span { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .12em;
      text-transform: uppercase; }
    .long-form-heading { display: flex; align-items: start; justify-content: space-between; gap: 16px; }
    .long-form-heading h3 { margin: 0; font-size: 1.45rem; line-height: 1.2; text-wrap: balance; }
    .duration { flex: 0 0 auto; border: 1px solid var(--line); border-radius: 999px; padding: 3px 9px; color: var(--red);
      background: var(--wash); font-size: .78rem; font-weight: 800; font-variant-numeric: tabular-nums; }
    .long-form-meta { margin: 9px 0 0; color: var(--teal); font-size: .86rem; font-weight: 800; }
    .source-link { display: inline-block; margin: 10px 0 0; font-size: .82rem; font-weight: 700; }
    .delivery { margin: 5px 0 0; color: var(--teal); font: 600 .8rem/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .line { margin: 14px 0; color: var(--muted); }
    footer { padding: 24px 0 42px; border-top: 5px solid var(--cream); color: var(--muted); }
    @media (max-width: 560px) {
      .shell { width: min(100% - 32px, 1180px); }
      .hero { padding-top: 36px; }
      nav { overflow-x: auto; flex-wrap: nowrap; }
      nav a { white-space: nowrap; }
    }
"""


if __name__ == "__main__":
    raise SystemExit(main())
