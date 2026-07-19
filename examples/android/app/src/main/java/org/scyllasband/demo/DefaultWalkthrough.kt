package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

/** The first-launch document, authored with the same tags accepted by imported presets. */
object DefaultWalkthrough {
    private val taggedSource =
        """
        [scylla:en_us:joy=0.6]Welcome to Scylla's Band Studio!

        [scylla:en_us:anger=0.6]You might notice I sound a bit angry here, for no great reason! Oh my god, why the heck am I so angry?

        [scylla:en_us:sadness=0.6]Or maybe just a little bit sad... Why is life so complicated?

        [max:en_us:calm=0.4]Tap the full tags to change the voice, language, emotion value, and strength. Mine is calm at forty percent, because every demo needs one responsible adult.

        [tuesday:en_gb:sarcasm=0.7]So that's the quick of the test. But we don't just roll with the emotions.

        [rex:es:joy=0.7]Ahora cambiamos al español con Rex. Mi tag selecciona español y alegría al setenta por ciento. Toca mi punto para probar calma, sarcasmo, o una C F G diferente. Un chiste: ¿por qué el sintetizador llevó un mapa? Porque no quería perder el tono.

        [ink:it:sarcasm=0.6]Ora tocca a Ink in italiano, con sarcasmo al sessanta per cento. Puoi cambiare emozione senza modificare il testo. Ecco una battuta: perché la voce artificiale ha portato una matita? Per prendere nota, naturalmente. Sì, era una battuta molto sintetica.

        [max:en_us:calm=0.35]In a tagged file, voice, language, and emotion can change together. Short tags can change only the active language or emotion. After import, those instructions become speaker points, so you can fine-tune them visually instead of memorizing syntax.

        [scylla:en_us:joy=0.8]Now press Play, tap a marker, and make us sound completely different. Final joke: why did four voices share one script? We wanted better dialogue, and none of us could agree on a group chat. Welcome to Scylla's Band.
        """.trimIndent()

    private val emotionCfgByPoint = listOf(3.0f, 2.4f, 3.0f, 1.2f, 2.5f, 1.1f, 1.2f, 0.9f, 1.15f)

    fun snapshot(
        defaults: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): SpeakerDocumentSnapshot {
        val parsed = GroupSpeakPresetParser.parse(taggedSource, defaults, bundleInfo)
        check(parsed.points.size == emotionCfgByPoint.size) {
            "Default walkthrough tags and CFG settings are out of sync"
        }
        return parsed.copy(
            points = parsed.points.mapIndexed { index, point ->
                point.copy(
                    settings = point.settings.copy(emotionCfg = emotionCfgByPoint[index]),
                )
            },
        )
    }
}
