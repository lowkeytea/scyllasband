# Spoken-form tables

One JSON file per normalizer language (`en`, `de`, `fr`, `es`, `it`, `vi`). The text normalizer reads them in Python and
the C++ runtime embeds them (`libscyllasband/tools/generate_text_data.py` writes `src/scyllasband_text_data.inc`; the
Python tests fail while that file is out of date). Adding a term is an edit here, not a code change.

## Matching

Every `written` form (a string, or a list of forms) matches as a whole token: not inside a longer word, and not right
after another letter, digit, dot or apostrophe. A space after a dot inside a form is optional, so `z. B.` also matches
`z.B.`; a space between letters matches any run of spaces.

`case` says which spellings of a form match:

| `case` | Matches |
|---|---|
| `sentence` (default for abbreviations and titles) | as written and in lower case anywhere; with a capital first letter at the start of a sentence; in capitals in text written all in capitals |
| `any` (default for units and `before_number`) | any mix of upper and lower case |
| `exact` | only as written; in capitals too when the text is all capitals |

Use `exact` when another casing means something else (`M.` monsieur, `MB` megabyte against `Mb` megabit, `max.` against
the name "Max.").

A spoken form is what the G2P reads, so it may be a respelling where the usual spelling comes out wrong (French
"Jésus Kri" for /ʒezykʁi/, German letter names "beh", "zeh"). Check a new one with the release G2P in a sentence.

## Sections

- `letters`: letter names, used to spell initialisms and initials.
- `words`: words the rules insert (`and`, `percent`, `degrees`, `point`, `range` for "10-15", ...).
- `months`, `currency` (`symbol: [one, many, minor one, minor many]`), `currency_scales` ("$5bn").
- `one`: the word for 1 before a unit, by grammatical gender (`m`, `f`, `n`).
- `abbreviations`: replaced wherever they stand. `ends: true` keeps a sentence-ending dot when the sentence ends there.
- `titles`: replaced before a capitalised word (a name). `after_name` is used after a capitalised word instead ("Main
  St." Street); `otherwise` in any other position ("Smith & Co." Company).
- `before_number`: replaced when a number follows ("No. 5", "S. 12", "§ 3", "~5").
- `units`: replaced after a number, with `one` or `many` by the number (French takes `one` below 2) and `gender` for the
  word "one" ("1 t" "eine Tonne").
- `symbols`: replaced wherever they stand ("&").
- `initialisms`: spelled letter by letter. `spell` entries are never words, so they match in any case and also in text
  written all in capitals. `capitals_only` entries are also ordinary words ("US", "IT"), so they are spelled only when
  written in capitals inside text that is not all capitals. English initialisms are spelled by the pronunciation
  lexicon, the others by the normalizer with `letters`. `lower_case_also` marks `spell` entries a dictionary lists in
  lower case as the same abbreviation ("pdf"), so the check below accepts them.
- `pronunciations` (English): phones by dialect (`en_us`, `en_gb`) for words the G2P gets wrong, applied by the
  pronunciation lexicon in any case. Phones must be in the model's phone set (others are dropped).
- `spell_unlisted`: a word in capitals of `min` to `max` letters with none of `vowels` is spelled too ("PTWC").
- `sentence_openers`, `name_leaders`, `feminine_titles`, `ordinals`, `ordinal_suffixes`: word lists the initials, Roman
  numeral and ordinal rules use.

`libscyllasband/tools/check_text_data.py` lists `spell` initialisms that are dictionary words in lower case (they belong in
`capitals_only`), and forms that two entries share.
