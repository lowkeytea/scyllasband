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
- `words`: words the rules insert (`and`, `percent`, `degrees`, `point`, `range` for "10-15", `minus` for "-3", `hour` and
  `hours` for clock times in the languages that name the hour, `year` before the year of a Vietnamese date, ...).
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
- `nonverbal` (English): non-verbal words recognised by shape and rewritten to one spelling (`spoken`) before the lexicon
  and the hyphen step see them: "Hm", "hhhmmm" become "hmm"; "mm-hmm", "mm hmm", "mmhm" become "mhm"; "shhh", "sssh"
  become "shh". A shape is letters, each optionally followed by `+` for one or more ("mm+" is two or more "m"); `-` is
  where the word may be written with one hyphen or space, or closed up. A shape matches a whole token in any case
  ("HMM" is "hmm"; the dotted "H.M.M." is letters), never right after a number ("5 mm"), and not a listed initialism
  written in capitals ("ER"). `when` narrows an entry: `before_punctuation` (a comma, full stop, ellipsis, question or
  exclamation mark follows: "Er, no" but not "to err is human") or `sentence_start_before_punctuation` ("Sh!"). The
  phones of a spoken form that the G2P gets wrong go in `pronunciations`.
- `nonverbal_spellings` (English; the next model in training): typed variants of a non-verbal word that the
  model has no training rows for, rewritten before the normalizer to the spelling it was trained on ("Uhuh" "Uh-uh", "A-hem" "Ahem", "Pst"
  "Psst"). Each rule is a regular expression (`pattern`) and the spelling to write (`replace`); the first letter keeps
  its case and a variant in capitals stays in capitals.
- `nonverbal_phones` (English; the next model in training): the phones the model was trained on for each
  non-verbal sound, a copy of the
  trainer's phone table named in `table`. `words` maps the normalizer's spelling to symbols separated by spaces; ` | `
  separates the parts of a sound the normalizer writes as several words, listed under `phrases` ("uh uh" is `uhuh`:
  each word takes one part, so the break is a word boundary; the same sound typed as one word takes the parts joined).
  A bundle that uses the table must have every symbol in its phone set.
- `pronunciations` (English): phones by dialect (`en_us`, `en_gb`) for words the G2P gets wrong, applied by the
  pronunciation lexicon in any case. Phones must be in the model's phone set (others are dropped).
- `context_pronunciations` (French): phones for a word whose reading depends on the words around it, applied by the
  pronunciation lexicon after the G2P. An entry applies to `word` when the previous word is in `after`, the next word is in
  `before`, or (`phrase_end: true`) the word ends its phrase; the first entry that applies wins, and an entry with phones
  outside the model's phone set is dropped. French numbers use it where the G2P drops a sound: the t of "vingt" before
  "deux" to "neuf" (never after "quatre"), the z of "dix" before "neuf", the final consonant of "huit", "six" and "dix"
  at the end of a phrase.
- `one_before_noun` (Spanish, Italian, German): the forms of "one" when a noun follows a number read with it. `one` is
  the number word, `masculine` and `feminine` its forms before a noun ("un"/"una", "ein"/"eine"); `compounds` lets the
  rule reach numbers ending in it ("veintiuno" "veintiún"/"veintiuna", with `compound_masculine` for the irregular
  ones); `feminine_before_vowel` and `vowels` elide it ("un'ora"); `keep_before` lists the beginnings that keep the plain
  form ("uno studente"). The noun is feminine when it is in `feminine_words`, masculine when in `masculine_words`, and
  otherwise by `feminine_endings`; in German a word in lower case is an adjective, feminine by
  `adjective_feminine_endings` ("eine neue Nachricht"). `not_nouns` are words that are no noun ("1 de cada 3") and
  `labels` words before a number that make it a label ("Kapitel 1 Abschnitt 2"). After a word in `dative_after` (German
  prepositions), "one" takes `dative_masculine` or `dative_feminine` ("in einer Stunde", "seit einem Jahr").
- `spell_unlisted`: a word in capitals of `min` to `max` letters with none of `vowels` is spelled too ("PTWC").
- `sentence_openers`, `name_leaders`, `feminine_titles`, `ordinals`, `ordinal_suffixes`: word lists the initials, Roman
  numeral and ordinal rules use.

`libscyllasband/tools/check_text_data.py` lists `spell` initialisms that are dictionary words in lower case (they belong in
`capitals_only`), and forms that two entries share.
