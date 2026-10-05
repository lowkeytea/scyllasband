"""Public Scylla's Band inference API.

Names load on first use, so ``python -m scyllasband`` can set up a missing environment before numpy is imported.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "CONTRACT_VERSION": "contract",
    "GRAPH_CONTRACT": "contract",
    "BundleError": "contract",
    "load_manifest": "contract",
    "validate_bundle_layout": "contract",
    "DEFAULT_MODELS_DIR": "download",
    "FLAVORS": "download",
    "RELEASE": "download",
    "REPO_ID": "download",
    "download_bundle": "download",
    "Engine": "engine",
    "OverlongError": "engine",
    "Sentence": "engine",
    "PlanChunk": "planner",
    "SynthesisPlan": "planner",
    "parse_group_lines": "planner",
    "SUPPORTED_BACKENDS": "runtime",
    "SUPPORTED_SAMPLERS": "runtime",
    "ScyllasBandRuntime": "runtime",
    "SynthesisRequest": "runtime",
    "SynthesisResult": "runtime",
    "VoiceSpec": "runtime",
    "StreamingEvent": "streaming",
    "StreamOptions": "streaming",
    "SpokenTextNormalizer": "text_normalizer",
    "SpokenTextNormalizerConfig": "text_normalizer",
    "normalize_spoken_text": "text_normalizer",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'scyllasband' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
