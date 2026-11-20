#!/usr/bin/env python3
"""Add/list/remove pronunciation overrides in the bind-mounted phoneme_dict.json.

Runs inside the container:
  docker compose exec -T scyllasband-wyoming python /app/pronunciation.py add pamela --say "PAM uh luh"
  docker compose exec -T scyllasband-wyoming python /app/pronunciation.py add pamela --phones "pæmˈələ"
  docker compose exec -T scyllasband-wyoming python /app/pronunciation.py list
  docker compose exec -T scyllasband-wyoming python /app/pronunciation.py remove pamela
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
    result = runtime.synthesize(SynthesisRequest(
        text=text, voice_id='orpheus' if language == 'en_gb' else 'scylla',
        language=language))
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
        lang = a.language
        word = a.word.strip().lower()
        if data.get(lang, {}).pop(word, None) is None:
            print(f'not found: {lang}/{word}', file=sys.stderr)
            return 1
        _save(data)
        print(f'removed {lang}/{word}')
        return

    lang, word = a.language, a.word.strip().lower()
    phones = _from_say(a.say, lang) if a.say is not None else a.phones
    phones = _validate(phones, lang)
    data.setdefault(lang, {})[word] = phones
    _save(data)
    print(f'{lang}/{word} = {phones}  (restart container to apply)')


if __name__ == '__main__':
    sys.exit(main())
