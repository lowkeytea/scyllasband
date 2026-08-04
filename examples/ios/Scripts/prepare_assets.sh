#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: prepare_assets.sh APP_RESOURCES_DIRECTORY" >&2
    exit 64
fi

script_dir="${0:A:h}"
scyllasband_root="${script_dir}/../../.."
destination_root="$1/scyllasband"
requested_bundle="${SCYLLASBAND_IOS_BUNDLE_DIR:-${scyllasband_root}/scyllasband/models/onnx-int8}"

if [[ -f "${requested_bundle}/manifest.json" ]]; then
    model_bundle="${requested_bundle}"
elif [[ -z "${SCYLLASBAND_IOS_BUNDLE_DIR:-}" && -f "${scyllasband_root}/scyllasband/models/onnx/manifest.json" ]]; then
    model_bundle="${scyllasband_root}/scyllasband/models/onnx"
    echo "warning: onnx-int8 is unavailable; embedding the larger ONNX bundle from ${model_bundle}"
else
    echo "error: Scylla's Band ONNX bundle not found at ${requested_bundle}" >&2
    echo "error: run 'python -m scyllasband download --runtime-bundles onnx-int8' or set SCYLLASBAND_IOS_BUNDLE_DIR" >&2
    exit 1
fi

required_files=(
    "manifest.json"
    "onnx/components/shared_weights.bin"
    "onnx/g2p/model.onnx"
)
for relative_path in "${required_files[@]}"; do
    if [[ ! -f "${model_bundle}/${relative_path}" ]]; then
        echo "error: model bundle is missing ${relative_path}" >&2
        exit 1
    fi
done

/bin/mkdir -p "${destination_root}/examples"
/usr/bin/ditto "${model_bundle}" "${destination_root}/onnx-int8"

for example_name in emotional_text.txt groupSpeak.txt test_document.txt; do
    source_example="${scyllasband_root}/data/${example_name}"
    if [[ ! -f "${source_example}" ]]; then
        echo "error: example script is missing at ${source_example}" >&2
        exit 1
    fi
    /usr/bin/ditto "${source_example}" "${destination_root}/examples/${example_name}"
done
