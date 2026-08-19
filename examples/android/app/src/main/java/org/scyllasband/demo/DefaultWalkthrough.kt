package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

/**
 * The first-launch document, authored with the same tags accepted by imported presets.
 *
 * Non-English lines, translated:
 * - Felix (de, whispered): "Felix here. I'm not whispering because it's a secret. I whisper because it's what I do best."
 * - Stone (fr): "My name is Stone. Drama isn't my thing. French is."
 * - Scylla (vi): "See? Still me, just in Vietnamese."
 * - Rex (es): "And Spanish is still here! Ten voices, a pile of languages, and not a single word leaves your phone."
 * - Max (it): "And then there's Italian: every sentence a little drama. Please, no applause."
 */
object DefaultWalkthrough {
    private val taggedSource =
        """
        [gwen:en_us:whisper=0.8]Psst. Hey. You, with the headphones. Come closer... the others don't know you're here yet.

        [gwen:en_us:joy=0.7]Nothing? Fine, plan B! Big, warm, friendly voice! Everyone trusts a happy voice, right? ...Right?

        [gwen:en_us:anger=0.7]Oh, come on! I whispered, I sparkled, and you're still just poking at the screen!

        [ink:en_gb:sarcasm=0.6]Gwen. Deep breath. What is it you're actually trying to tell these nice people?

        [scylla:en_us:calm=0.6]What she's trying to say is: we got an upgrade. Version two, with new languages and new tricks. And since the whole thing is named after me, I'll do the honors.

        [tuesday:en_gb:sarcasm=0.5]Introductions, then, before she does all seven languages herself. And don't worry: everything you're hearing happens right on your Android. The cloud wasn't invited.

        [felix:de:whisper=0.7]Felix hier. Ich flüstere nicht, weil es geheim ist. Ich flüstere, weil ich es am besten kann.

        [stone:fr:calm=0.5]Je m'appelle Stone. Le drame, ce n'est pas mon genre. Le français, si.

        [scylla:vi:joy=0.5]Thấy chưa? Vẫn là tôi, chỉ là bằng tiếng Việt.

        [rex:es:joy=0.8]¡Y el español sigue aquí! Diez voces, un montón de idiomas, y ni una sola palabra sale de tu teléfono.

        [max:it:sarcasm=0.5]E poi c'è l'italiano: ogni frase un piccolo dramma. Prego, niente applausi.

        [ariadne:en_us:calm=0.6]Everyone is very impressive. But I have been here for two whole versions now, and still nobody has brought the little cookies.

        [orpheus:en_gb:sadness=0.5]And that's the demo. She waits for her cookies, I wait for a bigger part. Play it again. Perhaps we both get lucky.
        """.trimIndent()

    private val emotionCfgByPoint =
        listOf(1.3f, 2.4f, 2.8f, 1.2f, 1.15f, 2.0f, 1.25f, 1.1f, 1.2f, 1.15f, 1.2f, 1.15f, 1.4f)

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
