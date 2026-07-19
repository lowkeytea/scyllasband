from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from scyllasband.planner import PlannerOptions, plan_chunks_from_prepared, split_text_chunk_records


class _PlanRequest(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("max_chunk_chars", ctypes.c_int32),
        ("min_chunk_chars", ctypes.c_int32),
    ]


class _PlanResult(ctypes.Structure):
    _fields_ = [("metadata_json", ctypes.c_char_p)]


def _native_plan(text: str, *, max_chars: int, min_chars: int) -> dict[str, object]:
    library_path = Path(os.environ["SCYLLASBAND_NATIVE_LIBRARY"])
    library = ctypes.CDLL(str(library_path))
    library.scyllasband_plan_long_form_chunks.argtypes = [
        ctypes.POINTER(_PlanRequest),
        ctypes.POINTER(_PlanResult),
    ]
    library.scyllasband_plan_long_form_chunks.restype = ctypes.c_int
    library.scyllasband_chunk_plan_result_free.argtypes = [ctypes.POINTER(_PlanResult)]
    request = _PlanRequest(text.encode("utf-8"), max_chars, min_chars)
    result = _PlanResult()
    status = library.scyllasband_plan_long_form_chunks(ctypes.byref(request), ctypes.byref(result))
    if status != 0 or not result.metadata_json:
        raise RuntimeError(f"native plan failed with status {status}")
    try:
        return json.loads(result.metadata_json.decode("utf-8"))
    finally:
        library.scyllasband_chunk_plan_result_free(ctypes.byref(result))


def main() -> None:
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu; "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )
    max_chars = 48
    min_chars = 16
    native = _native_plan(text, max_chars=max_chars, min_chars=min_chars)
    prepared = split_text_chunk_records(text, max_chars=max_chars, min_chars=min_chars)
    for row in prepared:
        row.update({"record_id": "record-0000", "voice": "", "language": ""})
    python_chunks = [
        chunk.to_dict()
        for chunk in plan_chunks_from_prepared(
            None,
            prepared,
            PlannerOptions(
                max_chunk_chars=max_chars,
                min_chunk_chars=min_chars,
                lookahead_chunks=1,
            ),
        )
    ]
    native_chunks = list(native["chunks"])
    if len(python_chunks) != len(native_chunks):
        raise AssertionError(f"chunk count mismatch: {len(python_chunks)} != {len(native_chunks)}")
    projection = (
        "chunk_id",
        "record_id",
        "chain_id",
        "order_index",
        "text",
        "context_before",
        "context_after",
        "boundary_before",
        "boundary_after",
        "starts_sentence",
        "ends_sentence",
        "split_reason",
        "prefix_policy",
    )
    for index, (python_chunk, native_chunk) in enumerate(zip(python_chunks, native_chunks)):
        python_projection = {key: python_chunk.get(key) for key in projection}
        native_projection = {key: native_chunk.get(key) for key in projection}
        if python_projection != native_projection:
            raise AssertionError(
                f"chunk {index} plan projection mismatch:\n"
                f"python={json.dumps(python_projection, sort_keys=True)}\n"
                f"native={json.dumps(native_projection, sort_keys=True)}"
            )
    after_values = {str(chunk["boundary_after"]) for chunk in native_chunks}
    if not {"chunk_continue", "clause_continue"}.issubset(after_values):
        raise AssertionError(f"parity fixture missed required boundary types: {sorted(after_values)}")
    expected_chains = sorted({str(chunk["chain_id"]) for chunk in python_chunks})
    native_chains = list(native["scheduler_hints"]["parallelizable_chains"])
    if expected_chains != native_chains:
        raise AssertionError(f"scheduler chain mismatch: {expected_chains} != {native_chains}")


if __name__ == "__main__":
    main()
