#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="$(cd "${script_dir}/.." && pwd)"
workspace_root="$(cd "${runtime_root}/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_dir="${1:-${workspace_root}/scyllasband_v2_test_samples/onnx_release_candidate_20260819}"
steps="${STEPS:-8}"
sampler="${SAMPLER:-heun}"
seed="${SEED:-20260819}"
data_dir="${runtime_root}/data/scyllasbandv2"
test_document="${runtime_root}/data/test_document.txt"
export PYTHONPATH="${runtime_root}${PYTHONPATH:+:${PYTHONPATH}}"

render_variant() {
    local variant="$1"
    local bundle="$2"
    local variant_dir="${output_dir}/${variant}"
    mkdir -p "${variant_dir}"
    "${python_bin}" -m scyllasband validate-bundle "${bundle}"

    "${python_bin}" -m scyllasband group-speak "${bundle}" --backend onnx \
        --file "${data_dir}/walkthrough_demo.txt" \
        --steps "${steps}" --sampler "${sampler}" --seed "${seed}" \
        --metadata "${variant_dir}/walkthrough_all_voices_7lang.json" \
        -o "${variant_dir}/walkthrough_all_voices_7lang.wav"

    "${python_bin}" -m scyllasband group-speak "${bundle}" --backend onnx \
        --file "${data_dir}/ink_gwen_rex_affect_axis_sweep_v1.txt" \
        --steps "${steps}" --sampler "${sampler}" --seed "${seed}" \
        --metadata "${variant_dir}/affect_ink_gwen_rex.json" \
        -o "${variant_dir}/affect_ink_gwen_rex.wav"

    "${python_bin}" -m scyllasband group-speak "${bundle}" --backend onnx \
        --file "${data_dir}/max_scylla_felix_multilingual_sweep_v1.txt" \
        --steps "${steps}" --sampler "${sampler}" --seed "${seed}" \
        --metadata "${variant_dir}/multilingual_max_scylla_felix.json" \
        -o "${variant_dir}/multilingual_max_scylla_felix.wav"

    "${python_bin}" -m scyllasband speak "${bundle}" --backend onnx \
        --file "${test_document}" --voice ariadne --language en_us \
        --steps "${steps}" --sampler "${sampler}" --seed "${seed}" \
        --metadata "${variant_dir}/test_document_ariadne_en_us.json" \
        -o "${variant_dir}/test_document_ariadne_en_us.wav"

    "${python_bin}" -m scyllasband speak "${bundle}" --backend onnx \
        --file "${test_document}" --voice tuesday --language en_gb \
        --steps "${steps}" --sampler "${sampler}" --seed "${seed}" \
        --metadata "${variant_dir}/test_document_tuesday_en_gb.json" \
        -o "${variant_dir}/test_document_tuesday_en_gb.wav"
}

mkdir -p "${output_dir}"
render_variant fp32 "${runtime_root}/models/v2/onnx"
render_variant int8 "${runtime_root}/models/v2/onnx-int8"
find "${output_dir}" -type f \( -name '*.wav' -o -name '*.json' \) -print0 \
    | sort -z | xargs -0 sha256sum > "${output_dir}/SHA256SUMS.txt"
echo "release samples complete: ${output_dir}"
