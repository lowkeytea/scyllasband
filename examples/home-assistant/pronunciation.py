#!/usr/bin/env python3
"""Add/list/remove pronunciation overrides in the bind-mounted phoneme_dict.json.

Runs inside the container:
  docker exec scyllasband-wyoming python /app/pronunciation.py add pamela --say "PAM uh luh"
  docker exec scyllasband-wyoming python /app/pronunciation.py add pamela --phones "pæmˈələ"
  docker exec scyllasband-wyoming python /app/pronunciation.py list
  docker exec scyllasband-wyoming python /app/pronunciation.py remove pamela
"""
import argparse, json, os, sys
from pathlib import Path

DICT = Path('/models/litert/assets/g2p/phoneme_dict.json')
VOCAB = Path('/models/litert/assets/phone_vocab.json')
# The compose file bind-mounts the project directory here, so the backup is
# visible on the host next to phoneme_dict.json.
BACKUP = Path('/app/pronunciation/phoneme_dict.json.bak')


def _vocab():
    return json.loads(VOCAB.read_text())['token_to_id']


def _validate(phones, language):
    from scyllasband.lexicon import split_phones
    split = split_phones(phones, _vocab())
    if not split:
        raise ValueError(f'invalid phones for {language}: {phones!r} '
                         '(every phone must exist in the bundle vocabulary)')
    return ''.join(split)


def _from_say(text, language):
    from scyllasband import ScyllasBandRuntime, SynthesisRequest
    runtime = ScyllasBandRuntime.from_bundle('/models/litert')
    # The dict format is per-locale, so derive phones once with an English
    # voice and fan the result out when --language all is used.
    derive_lang = language if language in ('en_gb', 'en_us') else 'en_gb'
    result = runtime.synthesize(SynthesisRequest(
        text=text, voice_id='orpheus' if derive_lang == 'en_gb' else 'scylla',
        language=derive_lang))
    phones = result.metadata['chunks'][0]['phones']
    return ''.join(p for p in phones if not p.startswith('<'))


def _load():
    return json.loads(DICT.read_text())


def _save(data):
    # Back up the live file first (visible on the host via the directory mount).
    BACKUP.write_text(DICT.read_text())
    # The dict is a bind mount: os.replace on the target fails with EBUSY.
    # Write in place; the runtime reads it only at startup.
    DICT.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def _languages(spec, data):
    """Resolve --language: 'all' means every locale already in the dict."""
    if spec == 'all':
        return sorted(data)
    return [spec]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    add = sub.add_parser('add')
    add.add_argument('word')
    add.add_argument('--language', default='en_gb')
    src = add.add_mutually_exclusive_group(required=True)
    src.add_argument('--say', help='respelling, e.g. "PAM uh luh"')
    src.add_argument('--phones', help='IPA phones, e.g. "pæmˈələ"')
    sub.add_parser('list')
    rm = sub.add_parser('remove')
    rm.add_argument('word')
    rm.add_argument('--language', default='en_gb')
    a = ap.parse_args()
    data = _load()

    if a.cmd == 'list':
        for lang, words in sorted(data.items()):
            for word, phones in sorted(words.items()):
                print(f'{lang:6s} {word:20s} {phones}')
        return
    if a.cmd == 'remove':
        langs = _languages(a.language, data)
        word = a.word.strip().lower()
        found = [lang for lang in langs if data.get(lang, {}).pop(word, None) is not None]
        if not found:
            print(f'not found: {a.language}/{word}', file=sys.stderr)
            return 1
        _save(data)
        print(f'removed {word} from {", ".join(found)}')
        return

    langs, word = _languages(a.language, data), a.word.strip().lower()
    phones = _from_say(a.say, a.language) if a.say is not None else a.phones
    phones = _validate(phones, langs[0] if langs else 'en_gb')
    for lang in langs:
        data.setdefault(lang, {})[word] = phones
    _save(data)
    print(f'{word} = {phones} in {", ".join(langs)}  (restart container to apply)')


if __name__ == '__main__':
    sys.exit(main())
