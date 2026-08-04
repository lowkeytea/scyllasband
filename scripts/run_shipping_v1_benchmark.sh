#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
ASR_PYTHON="${ASR_PYTHON:-${PYTHON}}"
RUN_DIR="${RUN_DIR:-validation_runs/shipping-v1-core}"
BUNDLE_DIR="${BUNDLE_DIR:-scyllasband/models/onnx}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark/v1}"
PAGES_DIR="${PAGES_DIR:-docs/benchmark/v1}"
RENDER_THREADS="${RENDER_THREADS:-8}"
ASR_WORKERS="${ASR_WORKERS:-1}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

voices=(ariadne felix gwen ink max orpheus rex scylla stone tuesday)
plan_voices=()
for voice in "${voices[@]}"; do
    plan_voices+=(--voice "${voice}")
done

"${PYTHON}" -m scyllasband.validation plan \
    --bundle "${BUNDLE_DIR}" \
    --tier core \
    "${plan_voices[@]}" \
    --output "${RUN_DIR}"

render_group() {
    local voice_args=()
    local voice
    for voice in "$@"; do
        voice_args+=(--voice "${voice}")
    done
    "${PYTHON}" -m scyllasband.validation render \
        --run "${RUN_DIR}" \
        --backend onnx \
        --bundle "${BUNDLE_DIR}" \
        --onnx-intra-op-threads "${RENDER_THREADS}" \
        --onnx-inter-op-threads 1 \
        "${voice_args[@]}"
}

render_group ariadne felix gwen ink max &
render_a=$!
render_group orpheus rex scylla stone tuesday &
render_b=$!
wait "${render_a}"
wait "${render_b}"

"${ASR_PYTHON}" -m scyllasband.validation asr \
    --run "${RUN_DIR}" \
    --profile release \
    --workers "${ASR_WORKERS}" \
    --scope both

"${PYTHON}" -m scyllasband.validation report --run "${RUN_DIR}"
"${PYTHON}" -m scyllasband.validation publish \
    --run "${RUN_DIR}" \
    --output "${BENCHMARK_DIR}" \
    --pages-output "${PAGES_DIR}" \
    --audio-format mp3 \
    --overwrite
