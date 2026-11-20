#!/usr/bin/env python3
"""Add/list/remove pronunciation overrides in the named-volume word list.

Runs inside the container:
  docker exec scyllasband-wyoming python /app/pronunciation.py add jon --phones "dʒˈɒn"
  docker exec scyllasband-wyoming python /app/pronunciation.py list
  docker exec scyllasband-wyoming python /app/pronunciation.py remove jon

Use --say only as a starting point: its G2P stress can be wrong, so check the
phones it prints and pass an exact --phones string when needed.
"""
import argparse, json, os, sys
from pathlib import Path

DICT = Path('/data/phoneme_dict.json')
VOCAB = Path('/models/litert/assets/phone_vocab.json')
BACKUP = Path('/data/phoneme_dict.json.bak')


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
    # Back up the live file first, then write in place. The runtime reads the
    # dict only at startup, so a restart is needed after a change.
    BACKUP.write_text(DICT.read_text())
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
    src.add_argument('--say', help='rough respelling; check the phones it produces')
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
