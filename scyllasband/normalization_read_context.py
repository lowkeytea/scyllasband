"""Finite, verified worker context for immutable normalization references.

Inventory JSON is parsed once per resident object. Every use still verifies
stored bytes and identities. Current assets are hashed at first use and guarded
at batch boundaries; this lifecycle is explicitly different from historical
replay and from rehashing every dependency for every recording.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any, Callable

from .normalization_profile_registry import (
    MAX_BYTES, PROFILE_SCHEMA, REFERENCE_SCHEMA, SOURCE_SCHEMA, SPOKEN_SCHEMA,
    ProfileRegistry, _check, _keys, _reference, _registry_name, _sha, _unb64,
    sha256, strict_json,
)
from .spoken_text_trace import _validate_profile

_ACTIVE: ContextVar['ProfileReadContext | None'] = ContextVar('normalization_read_batch', default=None)
POLICIES = {'current_at_batch_boundaries', 'historical_replay'}


class ReadContextError(ValueError):
    pass


def active_profile_read_context() -> 'ProfileReadContext | None':
    return _ACTIVE.get()


def _fingerprint(info) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    result = sys.getsizeof(value)
    if isinstance(value, dict):
        result += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        result += sum(_deep_size(item, seen) for item in value)
    elif is_dataclass(value):
        result += sum(_deep_size(getattr(value, field.name), seen) for field in fields(value))
    return result


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _binding(value: Any) -> dict:
    _keys(value, {'path', 'sha256', 'size_bytes'})
    _sha(value['sha256'])
    if (type(value['path']) is not str or not Path(value['path']).is_absolute() or '\x00' in value['path']
            or type(value['size_bytes']) is not int or value['size_bytes'] < 0):
        raise ValueError('invalid bound file')
    return value


def _bound_bytes(binding: dict, *, limit: int = MAX_BYTES) -> bytes:
    _binding(binding)
    if binding['size_bytes'] > limit:
        raise ReadContextError('bound reference exceeds input budget')
    fd = os.open(binding['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ReadContextError('bound reference is not a regular file')
        data = file.read(limit + 1)
        after = os.fstat(file.fileno())
    if _fingerprint(before) != _fingerprint(after):
        raise ReadContextError('bound reference changed while reading')
    _check(data, {'sha256': binding['sha256'], 'size_bytes': binding['size_bytes']})
    return data


@dataclass
class _StoredObject:
    root: str
    kind: str
    digest: str
    physical_sha256: str
    physical_size: int
    fingerprint: tuple


@dataclass
class _Inventory:
    stored: _StoredObject
    raw: bytes
    value: list


@dataclass
class _Profile:
    stored: _StoredObject
    reference: dict
    inventory_key: tuple[str, str]
    prefix: bytes
    suffix: bytes
    value: dict


@dataclass
class _Asset:
    binding: dict
    fingerprint: tuple


class ProfileReadContext:
    """Internal generic batch engine; source/word adapters own batch consumers.

    The retained-cache budget includes unique private containers/bytes and asset
    identity metadata. Returned deep copies and transient parse allocations are
    separate. Explicit record/reconstructed-byte caps bound each returned batch.
    """
    def __init__(self, *, max_cache_bytes: int = 64 * 1024 * 1024,
                 max_profiles: int = 16, max_inventories: int = 4,
                 max_batch_records: int = 16, max_reconstructed_bytes: int = 128 * 1024 * 1024,
                 asset_policy: str = 'current_at_batch_boundaries') -> None:
        self.max_cache_bytes = _positive(max_cache_bytes, 'max_cache_bytes')
        self.max_profiles = _positive(max_profiles, 'max_profiles')
        self.max_inventories = _positive(max_inventories, 'max_inventories')
        self.max_batch_records = _positive(max_batch_records, 'max_batch_records')
        self.max_reconstructed_bytes = _positive(max_reconstructed_bytes, 'max_reconstructed_bytes')
        if asset_policy not in POLICIES:
            raise ValueError('explicit supported asset policy required')
        self.asset_policy = asset_policy
        self._inventories: dict[tuple[str, str], _Inventory] = {}
        self._profiles: OrderedDict[tuple[str, str], _Profile] = OrderedDict()
        self._assets: dict[tuple[str, str, int], _Asset] = {}
        self._pinned: set[tuple[str, str]] = set()
        self._prepared: set[tuple] = set()
        self._used_assets: set[tuple[str, str, int]] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._stats = {name: 0 for name in ('batches', 'records', 'inventory_parses', 'profile_head_parses',
            'profile_loads', 'registry_full_hash_reads', 'asset_full_hash_reads', 'asset_full_hash_bytes',
            'asset_stat_checks', 'evictions', 'peak_retained_cache_bytes')}

    def _assert_open(self) -> None:
        if self._closed or self._poisoned:
            raise ReadContextError('read context is closed or poisoned; create a new context')

    def _read_object(self, root: str, kind: str, digest: str) -> tuple[bytes, _StoredObject]:
        registry = ProfileRegistry(root)
        root_info = os.stat(registry.root, follow_symlinks=False)
        with registry._directory(kind, digest, create=False) as (directory, name):
            directory_info = os.fstat(directory)
            before = os.stat(name, dir_fd=directory, follow_symlinks=False)
            data = registry._read_at(directory, name)
            after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if _fingerprint(before) != _fingerprint(after):
            raise ReadContextError('registry object changed during context read')
        fingerprint = ((root_info.st_dev, root_info.st_ino),
                       (directory_info.st_dev, directory_info.st_ino), _fingerprint(after))
        self._stats['registry_full_hash_reads'] += 1
        return data, _StoredObject(str(registry.root), kind, digest, sha256(data), len(data), fingerprint)

    def _guard_object(self, stored: _StoredObject) -> None:
        _, observed = self._read_object(stored.root, stored.kind, stored.digest)
        if observed != stored:
            raise ReadContextError('resident registry object identity or bytes changed')

    def _asset_keys(self, profile: _Profile) -> list[tuple[str, str, int]]:
        value = profile.value
        bindings = value['code'] + value['wrapper_code'] + [binding
            for dependency in value['dependencies'] if dependency['installed'] for binding in dependency['files']]
        return [(binding['path'], binding['sha256'], binding['size_bytes']) for binding in bindings]

    def _verify_assets(self, profile: _Profile) -> None:
        if self.asset_policy == 'historical_replay':
            return
        for path, digest, size in self._asset_keys(profile):
            key = (path, digest, size)
            self._used_assets.add(key)
            if key in self._assets:
                continue
            binding = {'path': path, 'sha256': digest, 'size_bytes': size}
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as file:
                before = os.fstat(file.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                    raise ReadContextError(f'current profile asset size/type differs: {path}')
                check = hashlib.sha256()
                total = 0
                for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
                    check.update(chunk)
                    total += len(chunk)
                after = os.fstat(file.fileno())
            if _fingerprint(before) != _fingerprint(after) or total != size or check.hexdigest() != digest:
                raise ReadContextError(f'current profile asset bytes/identity differ: {path}')
            self._assets[key] = _Asset(binding, _fingerprint(after))
            self._stats['asset_full_hash_reads'] += 1
            self._stats['asset_full_hash_bytes'] += total

    def _guard_assets(self) -> None:
        for key in self._used_assets:
            asset = self._assets[key]
            observed = os.stat(asset.binding['path'], follow_symlinks=False)
            self._stats['asset_stat_checks'] += 1
            if _fingerprint(observed) != asset.fingerprint:
                raise ReadContextError(f'current profile asset changed across batch: {asset.binding["path"]}')

    def _collect_unreferenced(self) -> None:
        inventory_keys = {profile.inventory_key for profile in self._profiles.values()}
        self._inventories = {key: value for key, value in self._inventories.items() if key in inventory_keys}
        if self._assets:
            used = {key for profile in self._profiles.values() for key in self._asset_keys(profile)}
            self._assets = {key: value for key, value in self._assets.items() if key in used}

    def _retained_bytes(self) -> int:
        if not self._inventories and not self._profiles and not self._assets:
            return 0
        return _deep_size((self._inventories, self._profiles, self._assets, self._pinned, self._used_assets))

    def _enforce_budget(self) -> None:
        while True:
            retained = self._retained_bytes()
            if (retained <= self.max_cache_bytes and len(self._profiles) <= self.max_profiles
                    and len(self._inventories) <= self.max_inventories):
                self._stats['peak_retained_cache_bytes'] = max(self._stats['peak_retained_cache_bytes'], retained)
                return
            victim = next((key for key in self._profiles if key not in self._pinned), None)
            if victim is None:
                raise ReadContextError('active profile/inventory set exceeds finite cache budget')
            del self._profiles[victim]
            self._stats['evictions'] += 1
            self._collect_unreferenced()

    def _profile(self, root: str, reference: dict) -> tuple[tuple[str, str], _Profile]:
        _reference(reference)
        key = (str(Path(root).absolute()), reference['sha256'])
        self._pinned.add(key)
        if key in self._profiles:
            profile = self._profiles[key]
            if profile.reference != reference:
                raise ReadContextError('resident profile has competing binding')
            self._guard_object(profile.stored)
            self._guard_object(self._inventories[profile.inventory_key].stored)
            self._profiles.move_to_end(key)
            self._verify_assets(profile)
            return key, profile
        raw, stored = self._read_object(key[0], 'profiles', key[1])
        descriptor, _ = strict_json(raw)
        _keys(descriptor, {'schema', 'profile', 'dependencies', 'prefix_b64', 'suffix_b64'})
        if descriptor['schema'] != PROFILE_SCHEMA or _reference(descriptor['profile']) != reference:
            raise ReadContextError('wrong profile descriptor')
        dependency_ref = _reference(descriptor['dependencies'])
        inventory_key = (key[0], dependency_ref['sha256'])
        if inventory_key in self._inventories:
            inventory = self._inventories[inventory_key]
            self._guard_object(inventory.stored)
            _check(inventory.raw, dependency_ref)
        else:
            dependency_bytes, dependency_stored = self._read_object(key[0], 'dependencies', dependency_ref['sha256'])
            _check(dependency_bytes, dependency_ref)
            value, _ = strict_json(dependency_bytes)
            self._stats['inventory_parses'] += 1
            if type(value) is not list:
                raise ReadContextError('dependency inventory is not an array')
            inventory = _Inventory(dependency_stored, dependency_bytes, value)
            self._inventories[inventory_key] = inventory
        prefix, suffix = _unb64(descriptor['prefix_b64']), _unb64(descriptor['suffix_b64'])
        check = hashlib.sha256()
        for part in (prefix, inventory.raw, suffix): check.update(part)
        if check.hexdigest() != reference['sha256'] or len(prefix) + len(inventory.raw) + len(suffix) != reference['size_bytes']:
            raise ReadContextError('profile reconstruction differs from binding')
        value, spans = strict_json(prefix + b'null' + suffix, paths=(('dependencies',),))
        self._stats['profile_head_parses'] += 1
        if (type(value) is not dict or spans != {('dependencies',): (len(prefix), len(prefix) + 4)}
                or value['dependencies'] is not None):
            raise ReadContextError('dependency placeholder is not the actual structural value')
        value['dependencies'] = inventory.value
        _validate_profile(value)
        profile = _Profile(stored, dict(reference), inventory_key, prefix, suffix, value)
        self._profiles[key] = profile
        self._stats['profile_loads'] += 1
        self._verify_assets(profile)
        self._enforce_budget()
        return key, profile

    def _reference(self, binding: dict) -> tuple[dict, str]:
        data = _bound_bytes(binding)
        value, _ = strict_json(data)
        _keys(value, {'schema', 'original_schema', 'original', 'registry', 'profile', 'prefix_b64', 'suffix_b64'})
        if value['schema'] != REFERENCE_SCHEMA or value['original_schema'] not in {SOURCE_SCHEMA, SPOKEN_SCHEMA}:
            raise ReadContextError('batch reader requires the explicit packed-reference schema')
        _reference(value['original'])
        _reference(value['profile'])
        root = str(Path(binding['path']).parent / _registry_name(value['registry']))
        return value, root

    def read_envelope(self, binding: dict, *, require_current: bool) -> dict:
        self._assert_open()
        if _ACTIVE.get() is not self or (binding['path'], binding['sha256'], binding['size_bytes']) not in self._prepared:
            raise ReadContextError('reference is not a member of this active verified batch')
        if type(require_current) is not bool or (require_current and self.asset_policy != 'current_at_batch_boundaries'):
            raise ReadContextError('historical batch cannot satisfy current-profile verification')
        reference, root = self._reference(binding)
        _, profile = self._profile(root, reference['profile'])
        inventory = self._inventories[profile.inventory_key]
        prefix, suffix = _unb64(reference['prefix_b64']), _unb64(reference['suffix_b64'])
        check = hashlib.sha256()
        parts = (prefix, profile.prefix, inventory.raw, profile.suffix, suffix)
        for part in parts: check.update(part)
        if check.hexdigest() != reference['original']['sha256'] or sum(map(len, parts)) != reference['original']['size_bytes']:
            raise ReadContextError('original full envelope binding mismatch')
        paths = (('profile',), ('normalization', 'profile'))
        value, spans = strict_json(prefix + b'null' + suffix, paths=paths)
        path = paths[0] if reference['original_schema'] == SPOKEN_SCHEMA else paths[1]
        if type(value) is not dict:
            raise ReadContextError('restored source envelope must be an object')
        parent = value if path == paths[0] else value.get('normalization')
        if (type(value) is not dict or value.get('schema') != reference['original_schema']
                or type(parent) is not dict or spans != {path: (len(prefix), len(prefix) + 4)}
                or parent['profile'] is not None):
            raise ReadContextError('producer profile placeholder is not structurally exact')
        parent['profile'] = deepcopy(profile.value)
        self._stats['records'] += 1
        return value

    def _run_batch(self, bindings: list[dict], consumer: Callable[[], Any]) -> Any:
        self._assert_open()
        if not self._lock.acquire(blocking=False):
            raise ReadContextError('one worker context cannot run concurrent batches')
        token = None
        try:
            # close() or a failing batch may have won the lock after the
            # optimistic check. Recheck before any registry or consumer work.
            self._assert_open()
            if _ACTIVE.get() is not None:
                raise ReadContextError('nested read contexts are not supported')
            if not 1 <= len(bindings) <= self.max_batch_records:
                raise ReadContextError('batch record budget exceeded or empty batch')
            self._pinned = set()
            self._prepared = set()
            self._used_assets = set()
            total = 0
            for binding in bindings:
                _binding(binding)
                reference, root = self._reference(binding)
                total += reference['original']['size_bytes']
                if total > self.max_reconstructed_bytes:
                    raise ReadContextError('batch reconstructed-byte budget exceeded')
                self._profile(root, reference['profile'])
                self._prepared.add((binding['path'], binding['sha256'], binding['size_bytes']))
            self._enforce_budget()
            self._guard_assets()
            token = _ACTIVE.set(self)
            result = consumer()
            # Recheck all membership objects, not only the last referenced profile.
            for key in self._pinned:
                profile = self._profiles[key]
                self._guard_object(profile.stored)
                self._guard_object(self._inventories[profile.inventory_key].stored)
            for binding in bindings: _bound_bytes(binding)
            self._guard_assets()
            self._stats['batches'] += 1
            return result
        except BaseException:
            self._poisoned = True
            # A rejected oversized admission or stale batch must not leave an
            # over-budget/private cache resident until the caller closes it.
            self._profiles.clear()
            self._inventories.clear()
            self._assets.clear()
            raise
        finally:
            if token is not None: _ACTIVE.reset(token)
            self._pinned.clear()
            self._prepared.clear()
            self._used_assets.clear()
            self._lock.release()

    def snapshot_stats(self) -> dict:
        return {**self._stats, 'retained_cache_bytes': self._retained_bytes(),
                'resident_profiles': len(self._profiles), 'resident_inventories': len(self._inventories),
                'asset_policy': self.asset_policy, 'closed': self._closed, 'poisoned': self._poisoned,
                'max_batch_records': self.max_batch_records,
                'max_reconstructed_bytes': self.max_reconstructed_bytes, 'max_cache_bytes': self.max_cache_bytes}

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise ReadContextError('cannot close an active batch')
        try:
            self._profiles.clear()
            self._inventories.clear()
            self._assets.clear()
            self._closed = True
        finally:
            self._lock.release()
