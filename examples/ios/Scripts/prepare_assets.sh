#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: prepare_assets.sh APP_RESOURCES_DIRECTORY" >&2
    exit 64
fi

script_dir="${0:A:h}"
scyllasband_root="${script_dir}/../../.."
models_dir="${scyllasband_root}/scyllasband/models"
destination_root="$1/scyllasband"

# Embed whatever the download script fetched into scyllasband/models:
#   coreai    -> used on iOS 27+
#   onnx-int8 -> ONNX fallback for earlier iOS (preferred: smaller)
#   onnx      -> ONNX fallback when int8 was not downloaded
# SCYLLASBAND_IOS_BUNDLE_DIR overrides discovery with a single bundle.

typeset -a embed_sources embed_names

verify_coreai_bundle() {
    local bundle="$1"
    local required=(
        "manifest.json"
        "coreai/g2p.aimodel"
        "coreai/duration_predictor.aimodel"
        "coreai/vector_context_encoder.aimodel"
        "coreai/vector_estimator.aimodel"
        "coreai/vocoder.aimodel"
    )
    for relative_path in "${required[@]}"; do
        if [[ ! -e "${bundle}/${relative_path}" ]]; then
            echo "error: Core AI bundle is missing ${relative_path}" >&2
            exit 1
        fi
    done
}

verify_onnx_bundle() {
    local bundle="$1"
    local required=(
        "manifest.json"
        "onnx/components/shared_weights.bin"
        "onnx/g2p/model.onnx"
    )
    for relative_path in "${required[@]}"; do
        if [[ ! -f "${bundle}/${relative_path}" ]]; then
            echo "error: ONNX bundle is missing ${relative_path}" >&2
            exit 1
        fi
    done
}

if [[ -n "${SCYLLASBAND_IOS_BUNDLE_DIR:-}" ]]; then
    override="${SCYLLASBAND_IOS_BUNDLE_DIR}"
    if [[ ! -f "${override}/manifest.json" ]]; then
        echo "error: Scylla's Band bundle not found at ${override}" >&2
        exit 1
    fi
    if [[ -d "${override}/coreai" ]]; then
        verify_coreai_bundle "${override}"
        embed_sources+=("${override}")
        embed_names+=("coreai")
    else
        verify_onnx_bundle "${override}"
        embed_sources+=("${override}")
        embed_names+=("onnx-int8")
    fi
else
    if [[ -f "${models_dir}/coreai/manifest.json" ]]; then
        verify_coreai_bundle "${models_dir}/coreai"
        embed_sources+=("${models_dir}/coreai")
        embed_names+=("coreai")
    fi
    for onnx_name in onnx-int8 onnx; do
        if [[ -f "${models_dir}/${onnx_name}/manifest.json" ]]; then
            verify_onnx_bundle "${models_dir}/${onnx_name}"
            embed_sources+=("${models_dir}/${onnx_name}")
            embed_names+=("${onnx_name}")
            break
        fi
    done
fi

if (( ${#embed_sources[@]} == 0 )); then
    echo "error: no Scylla's Band bundle found under ${models_dir}" >&2
    echo "error: run 'python -m scyllasband download' (Core AI plus ONNX on macOS 27, ONNX elsewhere)" >&2
    echo "error: or set SCYLLASBAND_IOS_BUNDLE_DIR to a specific runtime bundle" >&2
    exit 1
fi

# The destination persists across builds; clear it so bundles removed from
# scyllasband/models (or files removed within a bundle) don't linger in the app.
/bin/rm -rf "${destination_root}"
/bin/mkdir -p "${destination_root}/examples"
for (( index = 1; index <= ${#embed_sources[@]}; index++ )); do
    echo "note: embedding ${embed_names[index]} bundle from ${embed_sources[index]}"
    /usr/bin/ditto "${embed_sources[index]}" "${destination_root}/${embed_names[index]}"
done

for example_name in emotional_text.txt groupSpeak.txt test_document.txt; do
    source_example="${scyllasband_root}/data/${example_name}"
    if [[ ! -f "${source_example}" ]]; then
        echo "error: example script is missing at ${source_example}" >&2
        exit 1
    fi
    /usr/bin/ditto "${source_example}" "${destination_root}/examples/${example_name}"
done
