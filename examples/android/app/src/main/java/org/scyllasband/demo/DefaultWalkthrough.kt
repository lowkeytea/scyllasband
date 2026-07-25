package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

/** The first-launch document, authored with the same tags accepted by imported presets. */
object DefaultWalkthrough {
    private val taggedSource =
        """
        [scylla:en_us:sadness=0.6]Hey! Hey you, over there!

        [scylla:en_us:anger=0.6]Yo, I can see you reading the reddit post!

        [scylla:en_us:calm=0.6]Sorry, I'm getting ahead of myself. Hello guys, gals, this is a demo of a brand new voice model!

        [orpheus:en_gb:sarcasm=0.4]Really, a new voice model in this day and age? Why should anyone even care?

        [tuesday:en_gb:sarcasm=0.7]I think sir... sorry to butt in... but it's because this demo seems to have a bit more expression in it. And it's running on Android.

        [rex:es:joy=0.7]¡Menuda fantasía! ¿Todo esto corriendo en Android? ¡Pero si la gente dice que estos trastos no tienen ni para arrancar la lavadora!

        [ink:it:sarcasm=0.6]Mamma mia, certo che la gente vive proprio col prosciutto sugli occhi per Apple! Ma figurati se Android non è abbastanza potente da far girare i modelli vocali.

        [felix:en_us:calm=0.35]So hey, I'm glad you listened in. Scylla's Band already has a version 2 in the works with improved voices and new languages.

        [ariadne:en_us:questioning=0.8]I heard there were free cookies! Oh... sorry, I just wanted an excuse to say something. Pass the cupcake?
        
        [gwen:en_us:calm=0.8]Sorry, I think by cookies that means the author is taking suggestions while V 2 is still in the works. Thank you!
        """.trimIndent()

    private val emotionCfgByPoint = listOf(3.0f, 2.4f, 3.0f, 1.2f, 2.5f, 1.1f, 1.2f, 0.9f, 1.15f, 1.15f)

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
