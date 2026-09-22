# After the measured v2 release

Keep the released model as the listening baseline. Its four controls and whisper already provide useful delivery changes. The next training work should address temporal prosody without sacrificing speaker identity, pronunciation or stable controls.

## What the current model learned

Requests contain one vector per audio record. Frame-level physical measurements and qualified three-second audio-model ratings are local training objectives. They do not make the input controls time-dependent. At inference, broadcasting a request of energy 1.5 does not specify the source's local rise and fall around that value.

The existing relative-timing continuation improves rhythm objectives but still predicts one deterministic duration sequence. It neither samples alternative rhythmic performances nor predicts a local energy/tension/valence/assertiveness trajectory. Small untrained slider jitter should not be treated as a substitute for that training.

## First experiment: preserve full audio and align the window labels

Keep whole utterances; chopping every recording into three-second training examples is unnecessary. Carry each qualified rating window's timestamps and confidence into aligned local conditioning. Map that coarse evidence to acoustic frames and phone/word spans, with validity masks. Overlapping windows are coarse observations, not exact per-phone emotional truth. Short/ineligible tails must remain masked for rating supervision rather than being silently assigned false labels.

Retain the global four-axis request and binary whisper. Add a separate, smoothly varying local delivery representation to the duration and vector models. Supervise it from the existing window measurements plus the physical acoustic targets. An oracle-trajectory experiment should establish whether the generator can use local conditioning without losing voice quality before training a predictor to produce that trajectory.

## Second experiment: automatic performance planning

Users should still enter text, four values and whisper. A trained planner can predict a local residual trajectory and rhythm/emphasis choices from text, punctuation, voice and the global request. Training must constrain the local trajectory's aggregate to remain consistent with the requested delivery while allowing coherent within-utterance variation. Source trajectories should supervise the planner; naive independent noise should not.

Distinguish source-conditioned reconstruction from inference. Reporting success when the model receives the source's hidden trajectory would not establish that text-only synthesis can choose a natural trajectory. Evaluate the learned planner independently, including absent controls, combinations and multiple seeds.

## Data gap: connected speech

The prepared English material is overwhelmingly single-sentence. A local variation mechanism alone cannot establish discourse-level timing if training rarely contains connected passages. Add aligned two-to-four-sentence takes with changing emphasis, questions, turn transitions and emotional development within a stable voice. Preserve neutral/light takes and alternate readings; do not force every example toward extreme emotion.

Check coverage per voice and locale. Energy/tension/valence currently have more useful spread than assertiveness/expressiveness. Treat measurement reliability and coverage as separate questions; do not rescale every voice to invent missing extremes. Neutral remains 2.

## Evaluation that can justify expansion

Use matched held-out texts and the same voice, frontend phones, controls, sampler and decoder. Compare:

1. Original source and source-latent reconstruction.
2. Generated latents with source durations and oracle local delivery.
3. Generated latents with predicted durations and oracle local delivery.
4. Fully predicted durations and local delivery from text/global controls.
5. Released measured v2 and a continuation with the same training budget but no temporal-input change.

Assess pronunciation, pauses, emphasis placement, naturalness, speaker identity and control fidelity separately. Measure word-aligned timing and local score behavior, but do not maximize variance as a quality target. Blind/randomized listening across several seeds is needed before expanding a small experiment.

Start with English and two already trained voices, including one difficult voice such as Scylla, then verify transfer to the full voice/language set. Promote only if the fully predicted path improves connected speech while preserving the current single-sentence strengths. Singing and large pitch/time stretching remain separate objectives and should not enter the speech release through untrained inference tricks.
