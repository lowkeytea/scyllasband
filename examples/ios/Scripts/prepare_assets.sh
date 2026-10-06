#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: prepare_assets.sh APP_RESOURCES_DIRECTORY" >&2
    exit 64
fi

script_dir="${0:A:h}"
scyllasband_root="${script_dir:h:h:h}"
models_dir="${scyllasband_root}/scyllasband/models"
destination_root="$1/scyllasband"

# Embeds the Scylla's Band bundles found in scyllasband/models (symlinks are followed):
#   coreai -> Core AI, used on iOS and visionOS 27 and later (devices only)
#   coreml -> Core ML, used on iOS 18-26 and visionOS 2-26, in the Simulators, and wherever Core AI is unavailable
# SCYLLASBAND_IOS_BUNDLE_DIR embeds one specific bundle instead.

typeset -a embed_sources embed_names

bundle_flavor() {
    local bundle="$1"
    if [[ -d "${bundle}/coreai" ]]; then
        echo "coreai"
    elif [[ -d "${bundle}/coreml" ]]; then
        echo "coreml"
    else
        echo ""
    fi
}

verify_bundle() {
    local bundle="$1" flavor="$2" extension
    case "${flavor}" in
        coreai) extension="aimodel" ;;
        coreml) extension="mlmodelc" ;;
        *) echo "error: ${bundle} is neither a Core AI nor a Core ML bundle" >&2; exit 1 ;;
    esac
    if [[ ! -f "${bundle}/manifest.json" ]]; then
        echo "error: the ${flavor} bundle at ${bundle} is missing manifest.json" >&2
        exit 1
    fi
    if [[ ! -d "${bundle}/assets" ]]; then
        echo "error: the ${flavor} bundle at ${bundle} is missing assets/" >&2
        exit 1
    fi
    for asset in g2p duration_predictor vector_context_encoder vector_estimator vocoder; do
        if [[ ! -d "${bundle}/${flavor}/${asset}.${extension}" ]]; then
            echo "error: the ${flavor} bundle at ${bundle} is missing ${flavor}/${asset}.${extension}" >&2
            exit 1
        fi
    done
}

# Core AI has no Simulator runtime, so Simulator builds embed only Core ML.
simulator_build=0
[[ "${PLATFORM_NAME:-}" == *simulator ]] && simulator_build=1

if [[ -n "${SCYLLASBAND_IOS_BUNDLE_DIR:-}" ]]; then
    override="${SCYLLASBAND_IOS_BUNDLE_DIR:A}"
    flavor="$(bundle_flavor "${override}")"
    verify_bundle "${override}" "${flavor}"
    if [[ "${flavor}" == "coreai" && ${simulator_build} -eq 1 ]]; then
        echo "error: SCYLLASBAND_IOS_BUNDLE_DIR is a Core AI bundle; the Simulator needs a Core ML bundle" >&2
        exit 1
    fi
    embed_sources+=("${override}")
    embed_names+=("${flavor}")
else
    for flavor in coreai coreml; do
        candidate="${models_dir}/${flavor}"
        [[ -e "${candidate}" ]] || continue
        if [[ "${flavor}" == "coreai" && ${simulator_build} -eq 1 ]]; then
            echo "note: skipping the Core AI bundle; the Simulator runs the Core ML bundle"
            continue
        fi
        verify_bundle "${candidate}" "${flavor}"
        embed_sources+=("${candidate:A}")
        embed_names+=("${flavor}")
    done
fi

if (( ${#embed_sources[@]} == 0 )); then
    echo "error: no usable Scylla's Band bundle found in ${models_dir} (expected coreml/ and/or coreai/)" >&2
    echo "error: run 'python -m scyllasband download --flavor coreml --yes' (iOS 18+, visionOS 2+ and the Simulators)" >&2
    echo "error: and/or 'python -m scyllasband download --flavor coreai --yes' (iOS and visionOS 27+ devices) at the repository root," >&2
    echo "error: or set SCYLLASBAND_IOS_BUNDLE_DIR to one bundle directory" >&2
    exit 1
fi

# The destination persists across builds. rsync copies only what changed and --delete removes files that left a
# bundle; bundles, examples and older layouts no longer embedded are removed first. The .mlpackage sources are not needed at
# run time: the manifest points at the compiled .mlmodelc directories.
for entry in "${destination_root}"/*(N); do
    if (( ${embed_names[(Ie)${entry:t}]} == 0 )); then
        /bin/rm -rf "${entry}"
    fi
done
/bin/mkdir -p "${destination_root}/examples"
for (( index = 1; index <= ${#embed_sources[@]}; index++ )); do
    echo "note: embedding the ${embed_names[index]} bundle from ${embed_sources[index]}"
    /usr/bin/rsync -aL --delete --exclude '*.mlpackage' --exclude '.DS_Store' \
        "${embed_sources[index]}/" "${destination_root}/${embed_names[index]}/"
done

for example_name in walkthrough_demo.txt test_document.txt emotional_text.txt; do
    source_example="${scyllasband_root}/data/${example_name}"
    if [[ ! -f "${source_example}" ]]; then
        echo "error: example document is missing at ${source_example}" >&2
        exit 1
    fi
    /bin/cp -f "${source_example}" "${destination_root}/examples/${example_name}"
done
