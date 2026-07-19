# Scylla's Band Voice Samples

These clips preview the ten managed voices and the six-axis affect controls in the current Scylla's Band model.
The affect demonstrations use lines written for the requested delivery rather than neutral carrier text.
The multilingual previews use the same comparison line in each language so voice differences are easier to hear.
[Open the interactive audio gallery](https://lowkeytea.github.io/scyllasband/) to play every clip directly in the browser.

Affect strengths are normalized model inputs: `0.25`, `0.5`, `0.75`, and `1.0` correspond to human ratings `1`, `2`, `3`, and `4`.
The generated clips below use the `onnx` backend, 8-step `heun`, deterministic seed `2027`, and the per-line CFG shown in each table.

## Multilingual Voice Previews

| Voice | English | Spanish | Italian |
| --- | --- | --- | --- |
| Ariadne | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-en) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-es) | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-it) |
| Felix | [Listen](https://lowkeytea.github.io/scyllasband/#felix-en) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-es) | [Listen](https://lowkeytea.github.io/scyllasband/#felix-it) |
| Gwen | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-en) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-es) | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-it) |
| Ink | [Listen](https://lowkeytea.github.io/scyllasband/#ink-en) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-es) | [Listen](https://lowkeytea.github.io/scyllasband/#ink-it) |
| Max | [Listen](https://lowkeytea.github.io/scyllasband/#max-en) | [Listen](https://lowkeytea.github.io/scyllasband/#max-es) | [Listen](https://lowkeytea.github.io/scyllasband/#max-it) |
| Orpheus | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-en) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-es) | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-it) |
| Rex | [Listen](https://lowkeytea.github.io/scyllasband/#rex-en) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-es) | [Listen](https://lowkeytea.github.io/scyllasband/#rex-it) |
| Scylla | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-en) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-es) | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-it) |
| Stone | [Listen](https://lowkeytea.github.io/scyllasband/#stone-en) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-es) | [Listen](https://lowkeytea.github.io/scyllasband/#stone-it) |
| Tuesday | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-en) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-es) | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-it) |

## Affect Demonstrations

`calm`, `joy`, `anger`, and `sadness` are core delivery axes. `sarcasm` is demonstrated as an overlay mixed with a compatible core delivery.

### Ariadne (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The observatory is quiet tonight. Let the brass rings turn; every star will arrive exactly when it means to. | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-calm) |
| Joy | `joy=0.75` | 2.5 | I found the missing constellation! It was hiding behind the moon like a child waiting to shout surprise! | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-joy) |
| Anger | `anger=0.75` | 2.5 | You rewrote the map while we were still inside the labyrinth? Put it back before the walls decide to move again! | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-anger) |
| Sadness | `sadness=0.5` | 2 | I kept your place beside the telescope. The stars returned, one by one, but you never did. | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-sadness) |
| Sarcasm | `calm=0.5, sarcasm=0.75` | 2.5 | Oh, brilliant. You woke the ancient oracle just to ask where you left your keys. | [Listen](https://lowkeytea.github.io/scyllasband/#ariadne-sarcasm) |

### Felix (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The kettle is humming, the code is compiling, and for one miraculous minute, nothing needs fixing. | [Listen](https://lowkeytea.github.io/scyllasband/#felix-calm) |
| Joy | `joy=0.75` | 2.5 | It worked on the first try! Nobody touch anything; I want to remember this impossible moment forever! | [Listen](https://lowkeytea.github.io/scyllasband/#felix-joy) |
| Anger | `anger=0.75` | 2.5 | Who replaced my calibrated gears with painted biscuits? The machine is furious, and frankly, so am I! | [Listen](https://lowkeytea.github.io/scyllasband/#felix-anger) |
| Sadness | `sadness=0.5` | 2 | I repaired the little clock, but it still chimes for someone who is not coming home. | [Listen](https://lowkeytea.github.io/scyllasband/#felix-sadness) |
| Sarcasm | `calm=0.5, sarcasm=0.75` | 2.5 | Perfect. Another emergency caused by a button labeled, do not press. What a mystery. | [Listen](https://lowkeytea.github.io/scyllasband/#felix-sarcasm) |

### Gwen (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | Easy now. Lower the lantern, listen to the river, and let the path reveal itself. | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-calm) |
| Joy | `joy=0.75` | 2.5 | We made it over the ridge! There is the city, all gold roofs and bells, waiting for us! | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-joy) |
| Anger | `anger=0.75` | 2.5 | You locked the villagers outside their own gates? Open them now, or I will introduce your door to my axe! | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-anger) |
| Sadness | `sadness=0.5` | 2 | The banners are still flying, but there is no one left in the courtyard to see them. | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-sadness) |
| Sarcasm | `anger=0.5, sarcasm=0.75` | 2.5 | Of course your plan involved a dragon, a rope, and absolutely no exit. Inspired leadership. | [Listen](https://lowkeytea.github.io/scyllasband/#gwen-sarcasm) |

### Ink (`en_gb`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | Hush. The rain is writing tiny poems on the window, and even the ghosts have stopped to read. | [Listen](https://lowkeytea.github.io/scyllasband/#ink-calm) |
| Joy | `joy=0.75` | 2.5 | The moon stole my hat and the ravens formed a parade! This is officially an excellent evening! | [Listen](https://lowkeytea.github.io/scyllasband/#ink-joy) |
| Anger | `anger=0.75` | 2.5 | You fed my final page to the fireplace? I will haunt your footnotes for the rest of recorded history! | [Listen](https://lowkeytea.github.io/scyllasband/#ink-anger) |
| Sadness | `sadness=0.5` | 2 | The ink has dried in your last letter, but I still read it as if the words might change. | [Listen](https://lowkeytea.github.io/scyllasband/#ink-sadness) |
| Sarcasm | `joy=0.5, sarcasm=0.75` | 2.5 | Yes, release the cursed butterflies indoors. The wallpaper was far too peaceful. | [Listen](https://lowkeytea.github.io/scyllasband/#ink-sarcasm) |

### Max (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | One careful turn of the valve, one steady breath, and the whole ridiculous machine settles into rhythm. | [Listen](https://lowkeytea.github.io/scyllasband/#max-calm) |
| Joy | `joy=0.75` | 2.5 | The toaster has achieved flight! Breakfast is airborne, science is victorious, and nobody is currently on fire! | [Listen](https://lowkeytea.github.io/scyllasband/#max-joy) |
| Anger | `anger=0.75` | 2.5 | Stop pulling random levers! That one controls the gravity, and I have only just repaired the ceiling! | [Listen](https://lowkeytea.github.io/scyllasband/#max-anger) |
| Sadness | `sadness=0.5` | 2 | The workshop feels enormous without your terrible singing bouncing off the copper pipes. | [Listen](https://lowkeytea.github.io/scyllasband/#max-sadness) |
| Sarcasm | `joy=0.5, sarcasm=0.75` | 2.5 | Marvelous. You improved my self-stirring soup by teaching it to escape. | [Listen](https://lowkeytea.github.io/scyllasband/#max-sarcasm) |

### Orpheus (`en_gb`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The curtain waits in darkness. Breathe with the orchestra, and let the first note find you. | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-calm) |
| Joy | `joy=0.75` | 2.5 | They are on their feet! Listen to that thunder; tonight, even the chandeliers are applauding! | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-joy) |
| Anger | `anger=0.75` | 2.5 | You cut the final scene? That silence was the heart of the play, not an invitation for your scissors! | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-anger) |
| Sadness | `sadness=0.5` | 2 | The theater is empty now, yet your last song still seems to linger above the stage. | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-sadness) |
| Sarcasm | `calm=0.5, sarcasm=0.75` | 2.5 | A six-hour rehearsal with no script. At last, the purest possible form of theater. | [Listen](https://lowkeytea.github.io/scyllasband/#orpheus-sarcasm) |

### Rex (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The street is quiet, the trail is fresh, and the suspect has no idea we found the blue feather. | [Listen](https://lowkeytea.github.io/scyllasband/#rex-calm) |
| Joy | `joy=0.75` | 2.5 | Case closed! The jewels are back, the mayor owes us dinner, and I finally get to wear the victory hat! | [Listen](https://lowkeytea.github.io/scyllasband/#rex-joy) |
| Anger | `anger=0.75` | 2.5 | You endangered my team for a cheap headline. Hand over the evidence and get out of my way. | [Listen](https://lowkeytea.github.io/scyllasband/#rex-anger) |
| Sadness | `sadness=0.5` | 2 | I solved the case, but the old dog still waits by the station door for a partner who will not return. | [Listen](https://lowkeytea.github.io/scyllasband/#rex-sadness) |
| Sarcasm | `anger=0.5, sarcasm=0.75` | 2.5 | Sure, erase the fingerprints before I arrive. Evidence is famously improved by soap. | [Listen](https://lowkeytea.github.io/scyllasband/#rex-sarcasm) |

### Scylla (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | Let the tide carry the ship. The sea is listening, and tonight it has chosen mercy. | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-calm) |
| Joy | `joy=0.75` | 2.5 | The storm broke! Raise every sail; we are racing the sunrise all the way home! | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-joy) |
| Anger | `anger=0.75` | 2.5 | You chained my crew below deck during a storm? Unlock that hatch before I tear it from the ship! | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-anger) |
| Sadness | `sadness=0.5` | 2 | The lighthouse is dark, and every wave keeps bringing back the song we lost. | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-sadness) |
| Sarcasm | `anger=0.5, sarcasm=0.75` | 2.5 | Wonderful navigation. We have discovered the same doomed island for the third time. | [Listen](https://lowkeytea.github.io/scyllasband/#scylla-sarcasm) |

### Stone (`en_us`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The mountain does not hurry. Set down your burden, breathe, and watch the snow settle. | [Listen](https://lowkeytea.github.io/scyllasband/#stone-calm) |
| Joy | `joy=0.75` | 2.5 | The lost climbers are safe! Ring the bell; let the whole valley hear they made it home! | [Listen](https://lowkeytea.github.io/scyllasband/#stone-joy) |
| Anger | `anger=0.75` | 2.5 | You left them on the ridge to save your supplies? Take your pack and start walking. Now. | [Listen](https://lowkeytea.github.io/scyllasband/#stone-anger) |
| Sadness | `sadness=0.5` | 2 | I carved their names into the stone, because the wind was beginning to forget them. | [Listen](https://lowkeytea.github.io/scyllasband/#stone-sadness) |
| Sarcasm | `calm=0.5, sarcasm=0.75` | 2.5 | Excellent shortcut. We climbed six hours to arrive directly beneath where we started. | [Listen](https://lowkeytea.github.io/scyllasband/#stone-sarcasm) |

### Tuesday (`en_gb`)

| Delivery | Affect values | CFG | Line | Audio |
| --- | --- | ---: | --- | --- |
| Calm | `calm=0.5` | 2 | The inbox is quiet, the coffee is warm, and the calendar has graciously stopped making threats. | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-calm) |
| Joy | `joy=0.75` | 2.5 | The meeting was canceled! We have inherited a whole afternoon and absolutely no one can take it back! | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-joy) |
| Anger | `anger=0.75` | 2.5 | You scheduled a status meeting about the meeting we are currently having? Delete it. Delete all of them. | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-anger) |
| Sadness | `sadness=0.5` | 2 | The last train left without me, and even the station clock looks tired of waiting. | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-sadness) |
| Sarcasm | `calm=0.5, sarcasm=0.75` | 2.5 | Fantastic. The printer is on fire, the deadline moved closer, and somehow this is a team-building exercise. | [Listen](https://lowkeytea.github.io/scyllasband/#tuesday-sarcasm) |

## Rebuild

From the Scylla's Band repository root:

```bash
python samples/generate_affect_samples.py --include-multilingual --overwrite
```

The same command refreshes the self-contained `docs/` directory used for GitHub Pages branch publishing.

Use `--backend litert` to render the same script through the native LiteRT bundle, or `--voices ink rex` / `--deliveries anger sarcasm` for a focused subset.
