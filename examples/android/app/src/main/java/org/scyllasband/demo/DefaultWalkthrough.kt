package org.scyllasband.demo

import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings

/**
 * The first-launch document: the repository's `data/walkthrough_demo.txt` (bundled as an asset), a tagged
 * multi-voice, multilingual dialogue converted into editable speaker points.
 */
object DefaultWalkthrough {
    const val ASSET_NAME = "walkthrough_demo.txt"

    fun snapshot(
        source: String,
        defaults: ScyllasBandSegmentSettings,
        bundleInfo: ScyllasBandBundleInfo,
    ): SpeakerDocumentSnapshot = GroupSpeakPresetParser.parse(source, defaults, bundleInfo)
}
