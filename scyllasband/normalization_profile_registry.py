"""Lossless, immutable profile references for trusted normalization JSON bytes.

Profiles share dependency-array objects without changing a single original byte.
References authenticate storage, not the semantics of arbitrary producer edits;
call the existing strict producer/source reader after restoration.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import secrets
from typing import Any

REFERENCE_SCHEMA = "scyllasband_normalization_profile_reference_v1"
PROFILE_SCHEMA = "scyllasband_normalization_profile_object_v1"
SOURCE_SCHEMA = "scyllasband_fresh_source_text_trace_v1"
SPOKEN_SCHEMA = "scyllasband_spoken_normalization_trace_v1"
MAX_BYTES = 64 * 1024 * 1024
MAX_DEPTH = 40
MAX_VALUES = 1_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid content SHA256")
    return value


def _size(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= MAX_BYTES:
        raise ValueError("invalid or excessive byte size")
    return value


def _keys(value: Any, keys: set[str]) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid profile registry fields")


def _reference(value: Any) -> dict:
    _keys(value, {"sha256", "size_bytes"})
    _sha(value["sha256"])
    _size(value["size_bytes"])
    return value


def _ref(data: bytes) -> dict:
    return {"sha256": sha256(data), "size_bytes": len(data)}


def _check(data: bytes, reference: dict) -> None:
    _reference(reference)
    if len(data) != reference["size_bytes"] or sha256(data) != reference["sha256"]:
        raise ValueError("content-addressed bytes differ from binding")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode() + b"\n"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: Any) -> bytes:
    if type(value) is not str or len(value) > (MAX_BYTES + 2) // 3 * 4:
        raise ValueError("invalid or excessive base64 field")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid base64") from exc
    if _b64(decoded) != value:
        raise ValueError("noncanonical base64")
    return decoded


def strict_json(data: bytes, *, paths: tuple[tuple[str, ...], ...] = ()) -> tuple[Any, dict]:
    """Parse JSON structurally, returning exact requested value-token byte spans.

    Keys are decoded as JSON strings; duplicate/escaped aliases are rejected.
    Limits apply before materializing arbitrarily deep or huge documents. There
    is no substring search for profile/dependencies keys or inferred byte span.
    """
    if type(data) is not bytes or len(data) > MAX_BYTES:
        raise ValueError("invalid or excessive JSON bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid UTF8 JSON") from exc
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    spans = {}
    count = 0

    def whitespace(index):
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    def scalar(index):
        try:
            value, end = decoder.raw_decode(text, index)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid JSON scalar") from exc
        if type(value) is str:
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("unpaired surrogate in JSON string") from exc
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value, end

    def parse(index, path, depth):
        nonlocal count
        count += 1
        if count > MAX_VALUES or depth > MAX_DEPTH:
            raise ValueError("JSON structure exceeds bounded parser limits")
        index = whitespace(index)
        start = index
        if index >= len(text):
            raise ValueError("incomplete JSON value")
        if text[index] == "{":
            value = {}
            index = whitespace(index + 1)
            if index < len(text) and text[index] == "}":
                index += 1
            else:
                while True:
                    if index >= len(text) or text[index] != '"':
                        raise ValueError("JSON object requires a string key")
                    key, index = scalar(index)
                    if key in value:
                        raise ValueError("duplicate JSON key")
                    index = whitespace(index)
                    if index >= len(text) or text[index] != ":":
                        raise ValueError("missing JSON colon")
                    item, index = parse(index + 1, path + (key,), depth + 1)
                    value[key] = item
                    index = whitespace(index)
                    if index < len(text) and text[index] == "}":
                        index += 1
                        break
                    if index >= len(text) or text[index] != ",":
                        raise ValueError("missing JSON object delimiter")
                    index = whitespace(index + 1)
        elif text[index] == "[":
            value = []
            index = whitespace(index + 1)
            if index < len(text) and text[index] == "]":
                index += 1
            else:
                while True:
                    item, index = parse(index, path + (len(value),), depth + 1)
                    value.append(item)
                    index = whitespace(index)
                    if index < len(text) and text[index] == "]":
                        index += 1
                        break
                    if index >= len(text) or text[index] != ",":
                        raise ValueError("missing JSON array delimiter")
                    index = whitespace(index + 1)
        else:
            value, index = scalar(index)
        if path in paths:
            spans[path] = (len(text[:start].encode("utf-8")), len(text[:index].encode("utf-8")))
        return value, index

    value, end = parse(0, (), 0)
    if whitespace(end) != len(text):
        raise ValueError("trailing JSON content")
    return value, spans


def _profile_span(payload: bytes) -> tuple[dict, tuple[int, int]]:
    paths = (("profile",), ("normalization", "profile"))
    value, spans = strict_json(payload, paths=paths)
    if type(value) is not dict:
        raise ValueError("trace envelope must be a JSON object")
    schema = value.get("schema")
    if schema == SPOKEN_SCHEMA:
        path, profile = paths[0], value.get("profile")
    elif schema == SOURCE_SCHEMA:
        normalization = value.get("normalization")
        if type(normalization) is not dict or normalization.get("schema") != SPOKEN_SCHEMA:
            raise ValueError("missing production normalization envelope")
        path, profile = paths[1], normalization.get("profile")
    else:
        raise ValueError("unsupported original trace schema")
    if set(spans) != {path} or type(profile) is not dict:
        raise ValueError("expected exactly one structurally addressed producer profile")
    return value, spans[path]


def _registry_name(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None:
        raise ValueError("registry must be one safe sibling directory name")
    return value


class ProfileRegistry:
    """Digest-derived objects beneath an explicit caller-owned registry root.

    Atomic hard-link publication is exclusive and never replaces an object.
    Existing objects must match exactly. Dir-fd/O_NOFOLLOW operations reject
    symlink components beneath the root and avoid traversal or publication races.
    """
    def __init__(self, root: str | Path, *, create: bool = False) -> None:
        self.root = Path(root).absolute()
        if create:
            self.root.mkdir(exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("registry root must be an existing real directory")

    @contextmanager
    def _directory(self, kind: str, digest: str, *, create: bool):
        if kind not in {"profiles", "dependencies"}:
            raise ValueError("invalid object kind")
        _sha(digest)
        handles = [os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)]
        try:
            for part in (kind, digest[:2]):
                if create:
                    try:
                        os.mkdir(part, dir_fd=handles[-1])
                    except FileExistsError:
                        pass
                handles.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handles[-1]))
            yield handles[-1], digest + ".json"
        finally:
            for handle in reversed(handles):
                os.close(handle)

    @staticmethod
    def _read_at(directory: int, name: str) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
                raise ValueError("registry object is not a bounded regular file")
            data = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        geometry_changed = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        # Atomic publication temporarily has final+pending hard links. Removing
        # only the pending link changes ctime/nlink, not the already sealed bytes.
        publication_cleanup = before.st_nlink == 2 and after.st_nlink == 1
        if geometry_changed or (before.st_ctime_ns != after.st_ctime_ns and not publication_cleanup):
            raise ValueError("registry object changed while reading")
        if len(data) > MAX_BYTES:
            raise ValueError("registry object exceeds byte limit")
        return data

    def _read(self, kind: str, digest: str) -> bytes:
        with self._directory(kind, digest, create=False) as (directory, name):
            return self._read_at(directory, name)

    def _publish(self, kind: str, digest: str, data: bytes) -> None:
        if len(data) > MAX_BYTES:
            raise ValueError("registry object exceeds byte limit")
        with self._directory(kind, digest, create=True) as (directory, name):
            try:
                existing = self._read_at(directory, name)
            except FileNotFoundError:
                pass
            else:
                if existing != data:
                    raise ValueError("competing or stale content-addressed object")
                return
            # Temporary objects and final names are in the same opened directory.
            temporary = "." + secrets.token_hex(12) + ".pending"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                    os.fsync(directory)
                except FileExistsError:
                    if self._read_at(directory, name) != data:
                        raise ValueError("competing or stale content-addressed object")
            finally:
                os.unlink(temporary, dir_fd=directory)

    def store_profile(self, data: bytes) -> dict:
        profile, spans = strict_json(data, paths=(("dependencies",),))
        if type(profile) is not dict or type(profile.get("dependencies")) is not list or set(spans) != {("dependencies",)}:
            raise ValueError("profile requires one explicit dependency inventory")
        start, end = spans[("dependencies",)]
        dependencies = data[start:end]
        dependency_ref = _ref(dependencies)
        reference = _ref(data)
        descriptor = {"schema": PROFILE_SCHEMA, "profile": reference, "dependencies": dependency_ref,
                      "prefix_b64": _b64(data[:start]), "suffix_b64": _b64(data[end:])}
        self._publish("dependencies", dependency_ref["sha256"], dependencies)
        self._publish("profiles", reference["sha256"], _canonical(descriptor))
        return reference

    def read_profile(self, reference: dict) -> bytes:
        _reference(reference)
        descriptor, _ = strict_json(self._read("profiles", reference["sha256"]))
        _keys(descriptor, {"schema", "profile", "dependencies", "prefix_b64", "suffix_b64"})
        _reference(descriptor["profile"])
        if descriptor["schema"] != PROFILE_SCHEMA or descriptor["profile"] != reference:
            raise ValueError("wrong profile object schema/binding")
        dependency_ref = _reference(descriptor["dependencies"])
        dependencies = self._read("dependencies", dependency_ref["sha256"])
        _check(dependencies, dependency_ref)
        if type(strict_json(dependencies)[0]) is not list:
            raise ValueError("dependency object must be an array")
        prefix, suffix = _unb64(descriptor["prefix_b64"]), _unb64(descriptor["suffix_b64"])
        if len(prefix) + len(dependencies) + len(suffix) != reference["size_bytes"]:
            raise ValueError("profile restoration size mismatch")
        data = prefix + dependencies + suffix
        _check(data, reference)
        profile, spans = strict_json(data, paths=(("dependencies",),))
        if (type(profile) is not dict or spans != {("dependencies",): (len(prefix), len(prefix) + len(dependencies))}
                or type(profile.get("dependencies")) is not list):
            raise ValueError("profile dependency splice is not structurally exact")
        return data


def encode_reference(payload: bytes, *, expected_original_sha256: str,
                     registry: ProfileRegistry, registry_name: str) -> bytes:
    """Store exact trusted envelope profile; return a new reference-file payload."""
    _sha(expected_original_sha256)
    if sha256(payload) != expected_original_sha256:
        raise ValueError("original trace differs from trusted digest")
    _registry_name(registry_name)
    if registry.root.name != registry_name:
        raise ValueError("reference directory name must match the selected registry")
    value, (start, end) = _profile_span(payload)
    profile_ref = registry.store_profile(payload[start:end])
    reference = _canonical({"schema": REFERENCE_SCHEMA, "original_schema": value["schema"],
                       "original": _ref(payload), "registry": registry_name, "profile": profile_ref,
                       "prefix_b64": _b64(payload[:start]), "suffix_b64": _b64(payload[end:])})
    if len(reference) > MAX_BYTES:
        raise ValueError("encoded reference exceeds byte limit")
    return reference


def decode_reference(data: bytes, *, expected_reference_sha256: str, base_dir: str | Path) -> bytes:
    """Restore exact original bytes under an externally trusted reference digest."""
    _sha(expected_reference_sha256)
    if sha256(data) != expected_reference_sha256:
        raise ValueError("reference differs from trusted outer digest")
    value, _ = strict_json(data)
    _keys(value, {"schema", "original_schema", "original", "registry", "profile", "prefix_b64", "suffix_b64"})
    if value["schema"] != REFERENCE_SCHEMA or value["original_schema"] not in {SOURCE_SCHEMA, SPOKEN_SCHEMA}:
        raise ValueError("invalid reference/original schema")
    original = _reference(value["original"])
    registry = ProfileRegistry(Path(base_dir) / _registry_name(value["registry"]))
    profile = registry.read_profile(_reference(value["profile"]))
    prefix, suffix = _unb64(value["prefix_b64"]), _unb64(value["suffix_b64"])
    if len(prefix) + len(profile) + len(suffix) != original["size_bytes"]:
        raise ValueError("envelope restoration size mismatch")
    restored = prefix + profile + suffix
    _check(restored, original)
    envelope, span = _profile_span(restored)
    if span != (len(prefix), len(prefix) + len(profile)) or envelope["schema"] != value["original_schema"]:
        raise ValueError("restored profile is not the declared structural value")
    return restored
