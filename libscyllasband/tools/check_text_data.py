#!/usr/bin/env python3
"""Check scyllasband/text_data/*.json for entries that would misfire.

    python libscyllasband/tools/check_text_data.py

- ``initialisms.spell`` entries match in lower case too, so none may be a word in lower case. Each one is checked with the
  ``hunspell`` command against the language's dictionary and the English one (loanwords); a word found there belongs in
  ``capitals_only``. Languages without an installed dictionary are checked against English only, and say so.
- Two entries of one section may share a form only when their ``case`` policies keep them apart.

Exits non-zero when something is found.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scyllasband.text_data import TEXT_DATA, forms  # noqa: E402

DICTIONARIES = {"en": ["en_US", "en_GB"], "de": ["de_DE"], "fr": ["fr_FR"], "es": ["es_ES"], "it": ["it_IT"], "vi": ["vi_VN"]}
SECTIONS = {"abbreviations": "sentence", "titles": "sentence", "before_number": "any", "units": "any"}


def installed(dictionary: str) -> bool:
    return any((Path(folder) / f"{dictionary}.dic").exists() for folder in ("/usr/share/hunspell", "/usr/share/myspell/dicts"))


def known_words(words: list[str], dictionary: str) -> set[str]:
    """The words hunspell accepts (it lists the ones it does not)."""
    result = subprocess.run(["hunspell", "-d", dictionary, "-l"], input="\n".join(words), capture_output=True, text=True, check=True)
    return set(words) - set(result.stdout.split())


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []
    for lang, data in TEXT_DATA.items():
        initialisms = data.get("initialisms", {})
        checked = set(initialisms.get("lower_case_also", ()))   # read the same way in lower case ("pdf")
        spell = [word.lower() for word in initialisms.get("spell", ()) if word not in checked]
        if spell:
            if shutil.which("hunspell") is None:
                notes.append("hunspell is not installed; initialisms were not checked")
            else:
                dictionaries = [name for name in DICTIONARIES.get(lang, []) + DICTIONARIES["en"] if installed(name)]
                if not any(installed(name) for name in DICTIONARIES.get(lang, [])):
                    notes.append(f"{lang}: no {lang} dictionary installed; initialisms checked against English only")
                for dictionary in dict.fromkeys(dictionaries):
                    for word in sorted(known_words(spell, dictionary)):
                        problems.append(f"{lang}: initialism {word.upper()!r} is the word {word!r} ({dictionary}); "
                                        f"move it to capitals_only")
        for section, default_case in SECTIONS.items():
            seen: dict[str, dict[int, str]] = {}
            for index, entry in enumerate(data.get(section, ())):
                for form in forms(entry["written"]):
                    seen.setdefault(form.replace(" ", "").lower(), {})[index] = f"{form!r}:{entry.get('case', default_case)}"
            for key, by_entry in seen.items():
                uses = list(by_entry.values())
                policies = [use.rsplit(":", 1)[1] for use in uses]
                if len(uses) > 1 and any(policy != "exact" for policy in policies):
                    problems.append(f"{lang} {section}: {', '.join(uses)} share a form; make them exact or merge them")
    for note in notes:
        print("note:", note)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
