"""Reviewed unit-level eSpeak 1.51 roles, separate from source prosody.

This narrow profile admits only the seven exactly audited provider bindings.
Native names/roles come from the bound compiled table, and Vietnamese lexical
category meanings from pinned declarations. No acoustic vocabulary is selected.
A trusted raw emission is required; hashes supplied by an arbitrary producer do
not authenticate its native-call semantics. Word/rime/source-time ownership is
not established by a native output unit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .pronunciation_contract import (
    PronunciationError, PronunciationUnit, ProviderBinding, ProviderEmission,
    TextSpan, text_sha256,
)

SCHEMA = "scyllasband_espeak_native_unit_roles_v2"
PROFILE_SCHEMA = "scyllasband_reviewed_espeak_native_roles_v2"
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Exact current native producer, library, all 488 data files and options.
_AUDITED_PROVIDERS = MappingProxyType({'0e0a529ee8bf20406f3eb03ffb1d9fd6b5586831947fec9980941d5f22701bde': 'fr',
 '1115b8d64b3ebb8c338c7f35b4820c1c1e937b81188dd091a717588a4b19fca9': 'en_gb',
 '17bede8306da9678dd7316874e9cff9759515fcee6be12f7de392760d748a732': 'it',
 'e2002f7c810ae6295b470154eef136b829c5a36353cd62d18432341ee03adbc3': 'es',
 'e706ee6c45ea20da9cbf7ed08c5d85c97a9bd6a307f0a5ece5cf945e4358f979': 'de',
 'ed4c8a1f3f3475c62556f71bc57f659f526175457c7b57e525fb9e92dbe1312f': 'en_us',
 'fbec2a41ba329d9838e029c911c264ece05934d5bff1f06dd60360d085a0af94': 'vi'})

# Factual type/name lookup extracted with include-by-code override semantics.
_COMPILED_ROLES = {'de': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', 't#', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
        '2': ['3',
              '@',
              '@-',
              'A',
              'A:',
              'A~',
              'E',
              'E2',
              'E:',
              'EI',
              'I',
              'I:',
              'N-',
              'O',
              'OY',
              'O~',
              'U',
              'UR',
              'W',
              'W~',
              'Y:',
              'a',
              'aI',
              'aU',
              'e',
              'e:',
              'i',
              'i2',
              'i:',
              'iR',
              'l-',
              'm-',
              'n-',
              'o',
              'o:',
              'oU',
              'r-',
              'u',
              'u:',
              'y',
              'y:'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l2',
              'l^',
              'r',
              'r.',
              'r/',
              'w'],
        '4': ['?', 'c', 'k', 'p', 'pF', 'q', 't', 'tS', 'tS;', 't[', 'ts'],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'g'],
        '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'g#', 'h', 'l#', 's', 's.', 's;', 'x'],
        '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n^'],
        '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']},
 'en': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
        '2': ['0',
              '0#',
              '02',
              '3',
              '3:',
              '@',
              '@#',
              '@-',
              '@2',
              '@5',
              '@L',
              'A#',
              'A:',
              'A@',
              'A~',
              'E',
              'E#',
              'E2',
              'I',
              'I#',
              'I2',
              'I2#',
              'IR',
              'N-',
              'O',
              'O2',
              'O:',
              'O@',
              'OI',
              'O~',
              'U',
              'U@',
              'V',
              'VR',
              'a',
              'a#',
              'a#2',
              'a2',
              'aI',
              'aI3',
              'aI@',
              'aU',
              'aU@',
              'aa',
              'e',
              'e#',
              'e:',
              'e@',
              'eI',
              'i',
              'i:',
              'i@',
              'i@3',
              'l-',
              'm-',
              'n-',
              'o',
              'o:',
              'o@',
              'oU',
              'oU#',
              'u',
              'u:'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l^',
              'r',
              'r-',
              'r.',
              'r/',
              'w'],
        '4': ['?', 'c', 'd#', 'k', 'p', 'q', 't', 'tS', 'tS;', 't['],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'g', 't#', 't2'],
        '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'h', 'l#', 's', 's.', 's;', 'w#', 'x', 'z#'],
        '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z/2', 'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n^'],
        '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']},
 'en-us': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', '||'],
           '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
           '2': ['0',
                 '0#',
                 '02',
                 '3',
                 '3:',
                 '@',
                 '@#',
                 '@-',
                 '@2',
                 '@5',
                 '@L',
                 'A#',
                 'A:',
                 'A@',
                 'A~',
                 'E',
                 'E#',
                 'E2',
                 'I',
                 'I#',
                 'I2',
                 'I2#',
                 'IR',
                 'N-',
                 'O',
                 'O2',
                 'O:',
                 'O@',
                 'OI',
                 'O~',
                 'U',
                 'U@',
                 'V',
                 'VR',
                 'a',
                 'a#',
                 'a#2',
                 'a2',
                 'aI',
                 'aI3',
                 'aI@',
                 'aU',
                 'aU@',
                 'aa',
                 'e',
                 'e#',
                 'e:',
                 'e@',
                 'eI',
                 'i',
                 'i:',
                 'i@',
                 'i@3',
                 'l-',
                 'm-',
                 'n-',
                 'o',
                 'o:',
                 'o@',
                 'oU',
                 'oU#',
                 'u',
                 'u:'],
           '3': ['**',
                 ';',
                 'L',
                 'L/',
                 'R',
                 'R2',
                 'R3',
                 'j',
                 'l',
                 'l.',
                 'l/',
                 'l/2',
                 'l/3',
                 'l^',
                 'r',
                 'r-',
                 'r.',
                 'r/',
                 'w'],
           '4': ['?', 'c', 'd#', 'k', 'p', 'q', 't', 't2', 'tS', 'tS;', 't['],
           '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'g', 't#'],
           '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'h', 'l#', 's', 's.', 's;', 'w#', 'x', 'z#'],
           '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z/2', 'z;'],
           '8': ['N', 'm', 'n', 'n.', 'n^'],
           '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']},
 'es': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', 't#', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
        '2': ['@',
              '@-',
              'E',
              'EI',
              'N-',
              'O',
              'U',
              'Y',
              'a',
              'a/',
              'aI',
              'aU',
              'e',
              'e/',
              'eI',
              'eU',
              'i',
              'iU',
              'l-',
              'm-',
              'n-',
              'o',
              'o/',
              'oI',
              'r-',
              'u',
              'uI',
              'y'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l^',
              'r.',
              'r/',
              'w',
              'w2'],
        '4': ['?', 'c', 'k', 'p', 'q', 't', 'tS', 'tS;', 't[', 'ts'],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'g', 'r'],
        '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'h', 'l#', 's', 's.', 's;', 'x'],
        '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n^'],
        '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']},
 'fr': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', 't#', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
        '2': ['@',
              '@-',
              'A~',
              'E',
              'E-',
              'E~',
              'I',
              'I2',
              'N-',
              'O',
              'O~',
              'V',
              'W',
              'W2',
              'W~',
              'Y',
              'a',
              'a#',
              'a-',
              'e',
              'e-',
              'i',
              'j/',
              'l-',
              'm-',
              'n-',
              'o',
              'oU',
              'r-',
              'u',
              'u:',
              'w',
              'y',
              'y-'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'j.',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l^',
              'r.',
              'r/2',
              'w/'],
        '4': ['?', 'c', 'k', 'p', 'p2', 'q', 't', 't2', 't3', 'tS', 'tS;', 't['],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'g'],
        '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'h', 'l#', 's', 's.', 's;', 'x'],
        '7': ['B',
              'D',
              'J^',
              'Q',
              'Q"',
              'Q^',
              'Z',
              'Z;',
              'r',
              'r"',
              'r/',
              'r2',
              'v',
              'v#',
              'z',
              'z.',
              'z2',
              'z3',
              'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n2', 'n^'],
        '9': ['#@', '#a', '#cFR', '#e', '#i', '#l', '#o', '#r', '#u', '-', ':']},
 'it': {'0': ['1', '_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', 't#', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '='],
        '2': ['@',
              '@-',
              'E',
              'EI',
              'I',
              'N-',
              'O',
              'U',
              'Y',
              'a',
              'a/',
              'aI',
              'aU',
              'e',
              'e/',
              'eI',
              'eU',
              'i',
              'i#',
              'i/',
              'iU',
              'l-',
              'm-',
              'n-',
              'o',
              'o/',
              'oI',
              'r-',
              'u',
              'uI',
              'y'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l^',
              'r',
              'r.',
              'r/',
              'w',
              'w2'],
        '4': ['?', 'c', 'k', 'k~', 'p', 'q', 't', 'tS', 'tS;', 'tS~', 't[', 'ts', 'ts2'],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'dZ~', 'd[', 'dz', 'g', 'g~'],
        '6': ['#X1', 'C', 'S', 'S;', 'S~', 'T', 'X', 'f', 'h', 'l#', 's', 's.', 's;', 'ss', 'x'],
        '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Q~', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n^'],
        '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']},
 'vi': {'0': ['_', '_\x01', '_!', '_:', '_::', '_;_', '_X1', '_^_', '_|', 't#', '||'],
        '1': ['%', '%%', "'", "'!", "''", ',', ',,', '1', '2', '3', '4', '5', '6', '7', '='],
        '2': ['@',
              '@-',
              '@:',
              '@:I',
              '@:U',
              '@I',
              '@U',
              'E',
              'EI',
              'EU',
              'N-',
              'O',
              'O#',
              'O+',
              'O-',
              'OI',
              'Oi',
              'a',
              'a:',
              'a:I',
              'a:U',
              'aI',
              'aU',
              'e',
              'e-',
              'eI',
              'eU',
              'i',
              'i@',
              'iE',
              'iU',
              'l-',
              'm-',
              'n-',
              'o',
              'o#',
              'o&',
              'o@',
              'oI',
              'r-',
              'u',
              'u-',
              'u@',
              'uI',
              'y',
              'y@',
              'yI'],
        '3': ['**',
              ';',
              'L',
              'L/',
              'R',
              'R2',
              'R3',
              'j',
              'l',
              'l.',
              'l/',
              'l/2',
              'l/3',
              'l^',
              'r',
              'r.',
              'r/',
              'w'],
        '4': ['?', 'c', 'cr', 'k', 'p', 'q', 't', 'tS', 'tS;', 't['],
        '5': ['*', 'J', 'b', 'd', 'dZ', 'dZ;', 'd[', 'd_', 'g'],
        '6': ['#X1', 'C', 'S', 'S;', 'T', 'X', 'f', 'h', 'kh', 'l#', 's', 's.', 's;', 'x'],
        '7': ['B', 'D', 'J^', 'Q', 'Q"', 'Q^', 'Z', 'Z;', 'r"', 'v', 'v#', 'z', 'z.', 'z;'],
        '8': ['N', 'm', 'n', 'n.', 'n^'],
        '9': ['#@', '#a', '#e', '#i', '#o', '#u', '-', ':']}}

_COMPILED_SNAPSHOT_SHA256 = '981ea552a23f2ae6beee770dedbf274984f5f3197d5b68da82bc0510d148eb4e'

_VOICE_TABLE_EVIDENCE = {'0e0a529ee8bf20406f3eb03ffb1d9fd6b5586831947fec9980941d5f22701bde': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'fr',
                                                                                            'value': 'fr-fr'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'fr-fr',
                                                                      'requested_language': 'fr',
                                                                      'table_name': 'fr',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/roa/fr',
                                                                                     'sha256': '95f44834b48c075dad13eace54d2c98ff79b81aa0074dd67eebaf66c2909eef8',
                                                                                     'size_bytes': 79},
                                                                      'voice_identifier': 'roa/fr',
                                                                      'voice_language': 'fr-fr',
                                                                      'voice_name': 'French (France)'},
 '1115b8d64b3ebb8c338c7f35b4820c1c1e937b81188dd091a717588a4b19fca9': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'en',
                                                                                            'value': 'en-gb'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'en-gb',
                                                                      'requested_language': 'en_gb',
                                                                      'table_name': 'en',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/gmw/en',
                                                                                     'sha256': '4605d5330801de3641c6e366d15f129ea1f5ffbce8722642aba01ace07ab9c83',
                                                                                     'size_bytes': 140},
                                                                      'voice_identifier': 'gmw/en',
                                                                      'voice_language': 'en-gb',
                                                                      'voice_name': 'English (Great '
                                                                                    'Britain)'},
 '17bede8306da9678dd7316874e9cff9759515fcee6be12f7de392760d748a732': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'it',
                                                                                            'value': 'it'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'it',
                                                                      'requested_language': 'it',
                                                                      'table_name': 'it',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/roa/it',
                                                                                     'sha256': '0d9069eb9a96db1c55c131b2bb7d1f5255c68fbddc2199ffd3295b52519a3256',
                                                                                     'size_bytes': 109},
                                                                      'voice_identifier': 'roa/it',
                                                                      'voice_language': 'it',
                                                                      'voice_name': 'Italian'},
 'e2002f7c810ae6295b470154eef136b829c5a36353cd62d18432341ee03adbc3': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'es',
                                                                                            'value': 'es'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'es',
                                                                      'requested_language': 'es',
                                                                      'table_name': 'es',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/roa/es',
                                                                                     'sha256': '966aa015ea5646d79f0ca4807cf5da7339aabd3782b55cfa5eb0d8c3fc8fc588',
                                                                                     'size_bytes': 63},
                                                                      'voice_identifier': 'roa/es',
                                                                      'voice_language': 'es',
                                                                      'voice_name': 'Spanish (Spain)'},
 'e706ee6c45ea20da9cbf7ed08c5d85c97a9bd6a307f0a5ece5cf945e4358f979': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'de',
                                                                                            'value': 'de'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'de',
                                                                      'requested_language': 'de',
                                                                      'table_name': 'de',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/gmw/de',
                                                                                     'sha256': 'f3cca92f94b70f8c25a29ee0a4c9ce4c7f1022241532e0647fa2b7f698bf104e',
                                                                                     'size_bytes': 42},
                                                                      'voice_identifier': 'gmw/de',
                                                                      'voice_language': 'de',
                                                                      'voice_name': 'German'},
 'ed4c8a1f3f3475c62556f71bc57f659f526175457c7b57e525fb9e92dbe1312f': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'en',
                                                                                            'value': 'en-us'},
                                                                                           {'kind': 'explicit_phonemes_override',
                                                                                            'line': 8,
                                                                                            'selected_table': 'en-us',
                                                                                            'value': 'en-us'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'en-us',
                                                                      'requested_language': 'en_us',
                                                                      'table_name': 'en-us',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/gmw/en-US',
                                                                                     'sha256': '41534c2a22df5dd4f1052ff9e1a33a3ea7bff5a26b5c02bdad5ba8ddb7524704',
                                                                                     'size_bytes': 257},
                                                                      'voice_identifier': 'gmw/en-US',
                                                                      'voice_language': 'en-us',
                                                                      'voice_name': 'English (America)'},
 'fbec2a41ba329d9838e029c911c264ece05934d5bff1f06dd60360d085a0af94': {'compiled_table_exists': True,
                                                                      'evidence_scope': 'actual_saved_GetCurrentVoice_identity_bound_voice_bytes_and_pinned_LoadVoice_control_flow',
                                                                      'loader_decisions': [{'kind': 'first_language_default',
                                                                                            'line': 2,
                                                                                            'selected_table': 'vi',
                                                                                            'value': 'vi'}],
                                                                      'loader_source': {'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197',
                                                                                        'size_bytes': 44285,
                                                                                        'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c'},
                                                                      'not_inferred_from_requested_language_alone': True,
                                                                      'provider_language': 'vi',
                                                                      'requested_language': 'vi',
                                                                      'table_name': 'vi',
                                                                      'voice_file': {'path': '/usr/lib/x86_64-linux-gnu/espeak-ng-data/lang/aav/vi',
                                                                                     'sha256': '3199c980f9e23a88a2aa693cd631bf4fcb0f3408c4272bc01b7ac0ff8e79d778',
                                                                                     'size_bytes': 111},
                                                                      'voice_identifier': 'aav/vi',
                                                                      'voice_language': 'vi',
                                                                      'voice_name': 'Vietnamese (Northern)'}}

_DECLARATIONS = ({'sha256': '7a1801eb89b752e7e78cb69eb45f78ad7241adeacc685f802b6bd70fbcb3c418',
  'size_bytes': 91129,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/dictionary.c'},
 {'sha256': 'afb4964b85eb9391935a464464bce970688a5fb742fb936edbc146077153825c',
  'size_bytes': 9195,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/phoneme.h'},
 {'sha256': '123bc70d904ab38e7ca08ec46d4e1908f35509f69178eed66d289fbf3f76760b',
  'size_bytes': 25094,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/synthdata.c'},
 {'sha256': 'cac553230cfab540e0a9589f393a53ac2ba5421f3112321b833736c0717b7e39',
  'size_bytes': 10267,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/phsource/ph_english'},
 {'sha256': '86fffee9a3a87030d42e63c4a353c707e156e7cf705048d6aa3a7d5f302bd132',
  'size_bytes': 35983,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/phsource/phonemes'},
 {'sha256': '28719e84ebe1b28feaa8b64f722f00f48d6e4b907170e346aab39a76667eed10',
  'size_bytes': 19428,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/phsource/ph_vietnam'},
 {'sha256': 'b4c43b1ba578a77d248d06891da33ab4ae47ebc26f86c55402b447a8f94ac71e',
  'size_bytes': 22228,
  'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/dictsource/vi_rules'})

_DECLARATIONS = (*_DECLARATIONS, {'url': 'https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.51/src/libespeak-ng/voices.c', 'sha256': 'bd148628479063bdf88896e55ca434f1d22402781fb015e6bfe0ca5ad6088197', 'size_bytes': 44285})


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PronunciationError("Expected a SHA256 digest")
    return value


# Display mnemonics discard '/' variants in WritePhMnemonic. Preserve all
# matching full names; no choice of a variant is inferred from flat output.
_TABLES = MappingProxyType({name: MappingProxyType({printed: tuple(sorted(
    (native, int(type_id)) for type_id, natives in groups.items() for native in natives
    if native.split("/")[0] == printed))
    for printed in {native.split("/")[0] for natives in groups.values() for native in natives}})
    for name, groups in _COMPILED_ROLES.items()})
_LOOKUP_SHA256 = hashlib.sha256(_json(_COMPILED_ROLES)).hexdigest()
_TYPE_NAMES = ("pause", "stress_or_tone", "vowel", "liquid", "unvoiced_stop", "voiced_stop",
               "unvoiced_fricative", "voiced_fricative", "nasal", "virtual")
_TONES = MappingProxyType({"1": ("ngang", "1"), "2": ("huyen", "2"), "3": ("sac", "ɜ"),
                          "4": ("hoi", "4"), "5": ("nga", "5"), "6": ("nang", "6"),
                          "7": ("ngang", "7")})


@dataclass(frozen=True)
class ReviewedNativeRoleTable:
    provider_sha256: str

    def __post_init__(self) -> None:
        if _digest(self.provider_sha256) not in _AUDITED_PROVIDERS:
            raise PronunciationError("Provider is outside the exactly reviewed native role profile")

    @classmethod
    def for_provider(cls, provider: ProviderBinding) -> ReviewedNativeRoleTable:
        if not isinstance(provider, ProviderBinding):
            raise PronunciationError("Typed provider binding required")
        result = cls(provider.sha256)
        provider.verify_assets()
        return result

    def to_dict(self) -> dict:
        return {"schema": PROFILE_SCHEMA, "provider_sha256": self.provider_sha256,
            "implementation_sha256": _IMPLEMENTATION_SHA256, "lookup_sha256": _LOOKUP_SHA256,
            "compiled_snapshot_sha256": _COMPILED_SNAPSHOT_SHA256,
            "phontab_sha256": "88799c9eedd188a63e63304b86b1abc2dc91a8ad29c19f34e3da4a3d22ab4401",
            "compiled_layout": "little_endian_4byte_file_header_4byte_table_header_32byte_name_16byte_IIH6B_entry",
            "inheritance": "one_based_include_then_override_by_code",
            "supported_native_tables": sorted(_TABLES),
            "initial_table_evidence": json.loads(_json(_VOICE_TABLE_EVIDENCE[self.provider_sha256])),
            "source_declarations": [dict(row) for row in _DECLARATIONS], "native_tone_meaning_scope": "provider_lexical_category",
            "syllable_rime_word_or_source_time_ownership": False,
            "absent_tone_is_known_ngang": False, "acoustic_vocabulary_selected": False}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()


@dataclass(frozen=True, order=True)
class NativeDecomposition:
    native_name: str
    native_type: int
    synthetic_lengthen: bool
    synthetic_syllabic: bool
    tone_marker: str  # Empty means no appended tone, not neutral tone.


@dataclass(frozen=True)
class UnitRole:
    unit_id: int
    table_name: str | None
    native_role: str | None
    role_available: bool
    native_name_candidates: tuple[str, ...]
    native_identity_available: bool
    decompositions: tuple[NativeDecomposition, ...]
    provider_stress_surface: str | None
    lexical_tone: str | None
    lexical_tone_available: bool
    tone_marker: str | None
    tone_mnemonic_span: TextSpan | None
    tone_ipa_span: TextSpan | None
    native_host_unit_id: int | None
    native_host_available: bool
    reasons: tuple[str, ...]
    syllable_attachment_available: bool = field(default=False, init=False)
    nucleus_attachment_available: bool = field(default=False, init=False)
    rime_attachment_available: bool = field(default=False, init=False)
    normalized_word_ids: tuple[int, ...] = field(default=(), init=False)
    lexical_word_attachment_available: bool = field(default=False, init=False)
    source_time_attachment_available: bool = field(default=False, init=False)
    observed_acoustic_target_available: bool = field(default=False, init=False)


def _table_name(unit: PronunciationUnit, provider: ProviderBinding) -> str | None:
    if unit.language_evidence == "explicit_native_switch":
        # The native switch names the phoneme table; retain that identity.
        return unit.phonetic_language if unit.phonetic_language in _TABLES else None
    evidence = _VOICE_TABLE_EVIDENCE.get(provider.sha256)
    return evidence["table_name"] if evidence is not None else None


def _decompositions(body: str, table: Mapping[str, tuple[tuple[str, int], ...]], *, tones: bool) -> tuple[NativeDecomposition, ...]:
    result: set[NativeDecomposition] = set()
    # Enumerate the formatter's exact ordering: base + optional length marker +
    # optional nonvowel syllabic marker + optional tone. Maximum 8 branches.
    tone_options = ("", body[-1]) if tones and body and body[-1] in _TONES else ("",)
    for tone in tone_options:
        stem = body[:-1] if tone else body
        for syllabic in (False, True):
            if syllabic and not stem.endswith("-"):
                continue
            after_syllabic = stem[:-1] if syllabic else stem
            for lengthen in (False, True):
                if lengthen and not after_syllabic.endswith(":"):
                    continue
                base = after_syllabic[:-1] if lengthen else after_syllabic
                for native, type_id in table.get(base, ()):
                    if (syllabic and type_id not in range(3, 9)) or (lengthen and type_id not in range(2, 9)):
                        continue
                    if tone and type_id not in range(2, 9):
                        continue
                    result.add(NativeDecomposition(native, type_id, lengthen, syllabic, tone))
    return tuple(sorted(result))


def _interpret_unit(unit: PronunciationUnit, table_name: str | None) -> UnitRole:
    """Pure lookup mechanics. Public admission/reader validates the producer."""
    reasons: list[str] = []
    body, ipa, stress = unit.mnemonic, unit.ipa, None
    if unit.kind == "language_switch":
        reasons.append("native_table_switch_not_a_segment_or_tone")
    if body.startswith(("'", ",")):
        expected = "ˈ" if body[0] == "'" else "ˌ"
        if not ipa.startswith(expected):
            reasons.append("paired_stress_surfaces_disagree")
        else:
            stress, body, ipa = expected, body[1:], ipa[1:]
    elif ipa.startswith(("ˈ", "ˌ")):
        reasons.append("paired_stress_surfaces_disagree")
    table = _TABLES.get(table_name) if table_name is not None else None
    if table is None:
        reasons.append("unreviewed_native_table")
    parts = _decompositions(body, table, tones=table_name == "vi") if table is not None and not reasons else ()
    if not parts and not reasons:
        reasons.append("unsupported_native_decomposition")
    names = tuple(sorted({p.native_name for p in parts}))
    types = {p.native_type for p in parts}
    role = _TYPE_NAMES[next(iter(types))] if len(types) == 1 else None
    if len(types) > 1:
        reasons.append("native_role_ambiguous_between_valid_decompositions")
    if len(parts) > 1:
        reasons.append("native_identity_or_synthetic_modifier_ambiguous")
    marker = None
    host = None
    if table_name == "vi" and parts:
        tone_options = {p.tone_marker for p in parts}
        if len(tone_options) == 1 and "" not in tone_options:
            candidate = next(iter(tone_options))
            # Vowel carriers are reviewed here. Syllabic consonant/cross-role
            # carriers remain unavailable, even if a suffix looks familiar.
            if types == {2}:
                marker = candidate
                host = unit.unit_id
            else:
                reasons.append("unreviewed_nonvowel_tone_carrier")
        elif len(tone_options) > 1:
            reasons.append("tone_presence_ambiguous_between_native_name_and_suffix")
        elif names and len(names) == 1 and names[0] in _TONES and types == {1}:
            marker = names[0]  # Explicit standalone tone symbol; no host.
            role = "lexical_tone_marker"
        else:
            reasons.append("no_explicit_tone_evidence_absence_is_not_ngang")
    elif table_name is not None and unit.kind != "language_switch":
        reasons.append("tone_semantics_not_defined_for_this_native_table")
    if marker is not None and not ipa.endswith(_TONES[marker][1]):
        marker, host = None, None
        reasons.append("paired_tone_surfaces_disagree")
    tone = _TONES[marker][0] if marker is not None else None
    return UnitRole(unit.unit_id, table_name, role, role is not None, names,
        len(parts) == 1 and bool(names), parts, stress, tone, tone is not None, marker,
        TextSpan(unit.mnemonic_span.end - 1, unit.mnemonic_span.end) if marker else None,
        TextSpan(unit.ipa_span.end - 1, unit.ipa_span.end) if marker else None,
        host, host is not None, tuple(reasons))


@dataclass(frozen=True)
class NativeRoleInterpretation:
    emission: ProviderEmission
    table: ReviewedNativeRoleTable
    unit_roles: tuple[UnitRole, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.emission, ProviderEmission) or not isinstance(self.table, ReviewedNativeRoleTable):
            raise PronunciationError("Typed emission and reviewed role table required")
        if self.emission.provider.sha256 != self.table.provider_sha256:
            raise PronunciationError("Role table belongs to a different native provider")
        # Recompute the raw contract; no typed-object shortcut to derived masks.
        verified = ProviderEmission.from_dict(self.emission.to_dict(), expected_sha256=self.emission.sha256,
            expected_provider_sha256=self.table.provider_sha256,
            expected_input_sha256=text_sha256(self.emission.input_text))
        object.__setattr__(self, "unit_roles", tuple(_interpret_unit(unit, _table_name(unit, verified.provider))
                                                   for unit in verified.units))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "emission_sha256": self.emission.sha256,
            "input_text_sha256": text_sha256(self.emission.input_text),
            "provider_sha256": self.table.provider_sha256, "role_profile": self.table.to_dict(),
            "role_profile_sha256": self.table.sha256,
            "unit_roles": [asdict(x) for x in self.unit_roles],
            "scope": "provider_unit_roles_and_explicit_lexical_tone_not_observed_prosody",
            "word_source_time_mapping_available": False, "acoustic_vocabulary_selected": False}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, emission: ProviderEmission,
                  expected_sha256: str, expected_emission_sha256: str) -> NativeRoleInterpretation:
        if emission.sha256 != _digest(expected_emission_sha256):
            raise PronunciationError("Role result does not bind the expected original emission")
        result = cls(emission, ReviewedNativeRoleTable.for_provider(emission.provider))
        if not isinstance(value, Mapping) or _json(value) != _json(result.to_dict()) or result.sha256 != _digest(expected_sha256):
            raise PronunciationError("Native roles differ from trusted identity or recomputed facts")
        return result


def interpret_native_roles(emission: ProviderEmission, *, expected_emission_sha256: str) -> NativeRoleInterpretation:
    """Admit the bound provider and interpret; never infer absent acoustic data."""
    if not isinstance(emission, ProviderEmission) or emission.sha256 != _digest(expected_emission_sha256):
        raise PronunciationError("Expected a trusted typed native emission")
    return NativeRoleInterpretation(emission, ReviewedNativeRoleTable.for_provider(emission.provider))
