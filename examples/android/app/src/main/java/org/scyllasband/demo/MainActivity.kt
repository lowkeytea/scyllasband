package org.scyllasband.demo

import android.os.Bundle
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.scyllasband.android.ScyllasBand
import org.scyllasband.android.ScyllasBandAccelerator
import org.scyllasband.android.ScyllasBandDelivery
import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandSegmentSettings
import org.scyllasband.android.ScyllasBandVoice
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var editor: MarkerEditText
    private lateinit var exampleSpinner: Spinner
    private lateinit var defaultSettingsButton: Button
    private lateinit var playButton: Button
    private lateinit var clearMarkersButton: Button
    private lateinit var statusText: TextView
    private lateinit var progress: ProgressBar
    private lateinit var playbackTranscript: View
    private lateinit var playbackContext: TextView
    private lateinit var playbackText: TextView

    private val mainHandler = Handler(Looper.getMainLooper())
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private val stopRequested = AtomicBoolean(false)
    private val scyllasband by lazy { ScyllasBand.create(applicationContext) }

    @Volatile
    private var activePlayer: FloatAudioPlayer? = null

    private var bundleInfo: ScyllasBandBundleInfo? = null
    private var initializing = true
    private var playing = false
    private var defaultSettings = FALLBACK_SETTINGS
    private var lastPointSettings = FALLBACK_SETTINGS
    private var pendingSnapshot: SpeakerDocumentSnapshot? = null
    private var pendingDefaultWalkthrough = false

    private val examplePresets by lazy {
        listOf(
            ExamplePreset(R.string.example_walkthrough, DefaultWalkthrough.ASSET_NAME, true),
            ExamplePreset(R.string.example_test_document, "test_document.txt", false),
            ExamplePreset(R.string.example_emotional, "emotional_text.txt", false),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        applySystemBarInsets()
        bindViews()
        restoreState(savedInstanceState)
        bindInteractions()
        renderPendingSnapshot()
        updateControls()
        initializeScyllasBand()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(STATE_DEFAULT_WALKTHROUGH, pendingDefaultWalkthrough)
        if (!pendingDefaultWalkthrough) {
            outState.putString(STATE_DOCUMENT, snapshotToJson(snapshotFromEditor()).toString())
        }
        outState.putString(STATE_DEFAULTS, settingsToJson(defaultSettings).toString())
        outState.putString(STATE_LAST_POINT, settingsToJson(lastPointSettings).toString())
    }

    override fun onDestroy() {
        stopRequested.set(true)
        scyllasband.cancel()
        activePlayer?.close()
        activePlayer = null
        worker.execute { scyllasband.close() }
        worker.shutdown()
        super.onDestroy()
    }

    private fun bindViews() {
        editor = findViewById(R.id.scriptEditor)
        exampleSpinner = findViewById(R.id.exampleSpinner)
        defaultSettingsButton = findViewById(R.id.defaultSettingsButton)
        playButton = findViewById(R.id.playButton)
        clearMarkersButton = findViewById(R.id.clearMarkersButton)
        statusText = findViewById(R.id.statusText)
        progress = findViewById(R.id.progress)
        playbackTranscript = findViewById(R.id.playbackTranscript)
        playbackContext = findViewById(R.id.playbackContext)
        playbackText = findViewById(R.id.playbackText)
    }

    private fun applySystemBarInsets() {
        val content = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(content) { view, insets ->
            val safeArea = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
            )
            view.setPadding(safeArea.left, safeArea.top, safeArea.right, safeArea.bottom)
            insets
        }
        ViewCompat.requestApplyInsets(content)
    }

    private fun bindInteractions() {
        exampleSpinner.adapter = simpleAdapter(
            listOf(getString(R.string.choose_example)) +
                examplePresets.map { getString(it.titleResource) },
        )
        exampleSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position == 0) return
                val preset = examplePresets[position - 1]
                exampleSpinner.setSelection(0, false)
                confirmLoadExample(preset)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        editor.onAddMarkerRequested = { offset -> addMarker(offset) }
        editor.onEditMarkerRequested = { marker -> editMarker(marker) }
        defaultSettingsButton.setOnClickListener {
            showSettingsDialog(
                initial = defaultSettings,
                allowDelete = false,
                onSave = {
                    defaultSettings = it
                    setStatus(getString(R.string.status_default_updated, it.voiceId))
                },
            )
        }
        clearMarkersButton.setOnClickListener { clearMarkers() }
        playButton.setOnClickListener {
            if (playing) stopPlayback() else startPlayback()
        }
    }

    private fun initializeScyllasBand() {
        setStatus(getString(R.string.status_installing))
        worker.execute {
            val result = runCatching { scyllasband.initialize(accelerator = ACCELERATOR) }
            postToUi {
                result.onSuccess { info ->
                    bundleInfo = info
                    initializing = false
                    defaultSettings = validateSettings(defaultSettings, info)
                    lastPointSettings = validateSettings(lastPointSettings, info)
                    renderPendingSnapshot()
                    setStatus(
                        getString(
                            R.string.status_ready,
                            info.backend,
                            info.accelerator,
                            info.voices.size,
                        ),
                    )
                }.onFailure { error ->
                    initializing = false
                    setStatus(getString(R.string.status_init_failed, friendlyError(error)))
                }
                updateControls()
            }
        }
    }

    private fun addMarker(offset: Int) {
        if (bundleInfo == null) {
            setStatus(getString(R.string.status_wait_for_model))
            return
        }
        val editable = editor.text ?: return
        markerAtDisplayOffset(offset)?.let {
            editMarker(it)
            return
        }
        showSettingsDialog(
            initial = lastPointSettings,
            allowDelete = false,
            onSave = { settings ->
                val position = offset.coerceIn(0, editable.length)
                editable.insert(position, MARKER_TEXT)
                editable.setSpan(
                    SpeakerMarkerSpan(settings, resources.displayMetrics.density),
                    position,
                    position + MARKER_TEXT.length,
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE,
                )
                editor.setSelection((position + MARKER_TEXT.length).coerceAtMost(editable.length))
                lastPointSettings = settings
                setStatus(getString(R.string.status_marker_added, settings.voiceId))
            },
        )
    }

    private fun editMarker(marker: SpeakerMarkerSpan) {
        if (bundleInfo == null) {
            setStatus(getString(R.string.status_wait_for_model))
            return
        }
        showSettingsDialog(
            initial = marker.settings,
            allowDelete = true,
            onSave = { settings ->
                marker.settings = settings
                lastPointSettings = settings
                editor.invalidate()
                setStatus(getString(R.string.status_marker_updated, settings.voiceId))
            },
            onDelete = {
                val value = editor.text as? Spannable ?: return@showSettingsDialog
                val start = value.getSpanStart(marker)
                val end = value.getSpanEnd(marker)
                value.removeSpan(marker)
                if (start >= 0 && end > start) editor.text?.delete(start, end)
                setStatus(getString(R.string.status_marker_removed))
            },
        )
    }

    private fun showSettingsDialog(
        initial: ScyllasBandSegmentSettings,
        allowDelete: Boolean,
        onSave: (ScyllasBandSegmentSettings) -> Unit,
        onDelete: (() -> Unit)? = null,
    ) {
        val info = bundleInfo ?: return
        val dialogView = layoutInflater.inflate(R.layout.dialog_speaker_point, null)
        val voiceSpinner = dialogView.findViewById<Spinner>(R.id.voiceSpinner)
        val languageSpinner = dialogView.findViewById<Spinner>(R.id.languageSpinner)
        val deliveryControls = DeliveryControls(dialogView, layoutInflater)

        val voices = info.voices
        voiceSpinner.adapter = simpleAdapter(voices.map { it.displayName })
        deliveryControls.delivery = initial.delivery

        var displayedLanguages: List<String> = emptyList()

        fun selectedVoice(): ScyllasBandVoice = voices[voiceSpinner.selectedItemPosition]

        fun updateLanguages(preferred: String?) {
            val voice = selectedVoice()
            displayedLanguages = voice.languages
            languageSpinner.adapter = simpleAdapter(voice.languages.map(::languageLabel))
            val selected = voice.languages.indexOf(preferred).takeIf { it >= 0 }
                ?: voice.languages.indexOf(voice.defaultLanguage).coerceAtLeast(0)
            languageSpinner.setSelection(selected, false)
        }

        voiceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val previous = displayedLanguages.getOrNull(languageSpinner.selectedItemPosition)
                updateLanguages(previous)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        val initialVoiceIndex = voices.indexOfFirst { it.id == initial.voiceId }.coerceAtLeast(0)
        voiceSpinner.setSelection(initialVoiceIndex, false)
        updateLanguages(initial.language)

        val builder = MaterialAlertDialogBuilder(this)
            .setTitle(if (allowDelete) R.string.edit_speaker_point else R.string.add_speaker_point)
            .setView(dialogView)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.save, null)
        if (allowDelete) builder.setNeutralButton(R.string.remove_point, null)
        val dialog = builder.create()
        dialog.setOnShowListener {
            dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val voice = selectedVoice()
                val settings = ScyllasBandSegmentSettings(
                    voiceId = voice.id,
                    language = voice.languages[languageSpinner.selectedItemPosition],
                    delivery = deliveryControls.delivery,
                )
                onSave(settings)
                dialog.dismiss()
            }
            if (allowDelete) {
                dialog.getButton(android.app.AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                    onDelete?.invoke()
                    dialog.dismiss()
                }
            }
        }
        dialog.show()
    }

    private fun startPlayback() {
        val info = bundleInfo ?: return
        val segments = SpeakerDocument.segments(snapshotFromEditor(), defaultSettings)
        if (segments.isEmpty()) {
            setStatus(getString(R.string.status_no_text))
            return
        }
        stopRequested.set(false)
        showPlaybackText(segments.first().text, segments.first().settings)
        playing = true
        updateControls()
        val playbackStartedAt = SystemClock.elapsedRealtime()
        val nativeHeapAtStart = Debug.getNativeHeapAllocatedSize()
        worker.execute {
            var player: FloatAudioPlayer? = null
            var firstAudioMs: Long? = null
            val result = runCatching {
                segments.forEachIndexed { index, segment ->
                    if (stopRequested.get()) return@runCatching
                    var currentChunkText = segment.text
                    scyllasband.synthesizeStreaming(
                        segment.text,
                        segment.settings,
                        seed = PLAYBACK_SEED + index,
                        onChunkStarted = { chunkIndex, chunkCount, chunkText ->
                            if (stopRequested.get()) {
                                false
                            } else {
                                currentChunkText = chunkText ?: segment.text
                                if (player == null) {
                                    postToUi {
                                        setStatus(
                                            getString(
                                                R.string.status_rendering_chunk,
                                                index + 1,
                                                segments.size,
                                                chunkIndex + 1,
                                                chunkCount,
                                                segment.settings.voiceId,
                                            ),
                                        )
                                    }
                                }
                                true
                            }
                        },
                        onAudioChunk = { audio, chunkIndex, chunkCount ->
                            if (stopRequested.get()) {
                                false
                            } else {
                                if (firstAudioMs == null) {
                                    firstAudioMs = SystemClock.elapsedRealtime() - playbackStartedAt
                                    Log.i(
                                        LOG_TAG,
                                        "first audio in ${firstAudioMs}ms; native heap " +
                                            "${Debug.getNativeHeapAllocatedSize() / 1024L} KiB",
                                    )
                                }
                                val playbackPlayer = player ?: FloatAudioPlayer(info.sampleRate).also {
                                    player = it
                                    activePlayer = it
                                }
                                val audibleText = currentChunkText
                                playbackPlayer.enqueue(
                                    audio,
                                    onPlaybackStarted = {
                                        postToUi {
                                            showPlaybackText(audibleText, segment.settings)
                                            setStatus(
                                                getString(
                                                    R.string.status_playing_chunk,
                                                    index + 1,
                                                    segments.size,
                                                    chunkIndex + 1,
                                                    chunkCount,
                                                    segment.settings.voiceId,
                                                ),
                                            )
                                        }
                                    },
                                    shouldStop = { stopRequested.get() },
                                )
                            }
                        },
                    )
                }
                if (!stopRequested.get()) {
                    player?.finish { stopRequested.get() }
                }
            }
            player?.close()
            activePlayer = null
            Log.i(
                LOG_TAG,
                "playback ${when {
                    stopRequested.get() -> "stopped"
                    result.isSuccess -> "completed"
                    else -> "failed"
                }}; " +
                    "firstAudioMs=${firstAudioMs ?: -1}; " +
                    "totalMs=${SystemClock.elapsedRealtime() - playbackStartedAt}; " +
                    "nativeHeapDeltaKiB=" +
                    "${(Debug.getNativeHeapAllocatedSize() - nativeHeapAtStart) / 1024L}",
            )
            postToUi {
                playing = false
                if (stopRequested.get()) {
                    setStatus(getString(R.string.status_stopped))
                } else {
                    result.onSuccess { setStatus(getString(R.string.status_finished)) }
                    result.onFailure { setStatus(getString(R.string.status_playback_failed, friendlyError(it))) }
                }
                updateControls()
            }
        }
    }

    private fun confirmLoadExample(preset: ExamplePreset) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.replace_script_title)
            .setMessage(R.string.replace_script_message)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.replace) { _, _ -> loadExample(preset) }
            .show()
    }

    private fun loadExample(preset: ExamplePreset) {
        val info = bundleInfo ?: return
        val result = runCatching {
            val source = readExample(preset.assetName)
            if (preset.groupSpeak) {
                GroupSpeakPresetParser.parse(source, defaultSettings, info)
            } else {
                SpeakerDocumentSnapshot(source.trim(), emptyList())
            }
        }
        result.onSuccess { snapshot ->
            renderSnapshot(snapshot)
            lastPointSettings = snapshot.points.lastOrNull()?.settings ?: defaultSettings
            setStatus(
                getString(
                    R.string.status_example_loaded,
                    getString(preset.titleResource),
                    snapshot.points.size,
                ),
            )
        }.onFailure { error ->
            setStatus(getString(R.string.status_example_failed, friendlyError(error)))
        }
    }

    private fun readExample(assetName: String): String =
        assets.open("scyllasband/examples/$assetName").bufferedReader(Charsets.UTF_8).use { it.readText() }

    private fun stopPlayback() {
        stopRequested.set(true)
        scyllasband.cancel()
        setStatus(getString(R.string.status_stopping))
        activePlayer?.close()
        activePlayer = null
        updateControls()
    }

    private fun clearMarkers() {
        val value = editor.text as? Spannable ?: return
        value.getSpans(0, value.length, SpeakerMarkerSpan::class.java)
            .sortedByDescending { value.getSpanStart(it) }
            .forEach { marker ->
                val start = value.getSpanStart(marker)
                val end = value.getSpanEnd(marker)
                value.removeSpan(marker)
                if (start >= 0 && end > start) editor.text?.delete(start, end)
            }
        setStatus(getString(R.string.status_markers_cleared))
    }

    private fun updateControls() {
        val ready = bundleInfo != null && !initializing
        editor.visibility = if (playing) View.GONE else View.VISIBLE
        playbackTranscript.visibility = if (playing) View.VISIBLE else View.GONE
        exampleSpinner.isEnabled = ready && !playing
        defaultSettingsButton.isEnabled = ready && !playing
        clearMarkersButton.isEnabled = !playing
        playButton.isEnabled = (ready && editor.text?.isNotBlank() == true) || playing
        playButton.setText(if (playing) R.string.stop else R.string.play)
        progress.visibility = if (initializing || playing) View.VISIBLE else View.GONE
    }

    private fun showPlaybackText(text: String, settings: ScyllasBandSegmentSettings) {
        playbackContext.text = getString(
            R.string.now_playing_context,
            settings.voiceId,
            languageLabel(settings.language),
        )
        playbackText.text = text
    }

    private fun markerAtDisplayOffset(offset: Int): SpeakerMarkerSpan? {
        val value = editor.text as? Spannable ?: return null
        val bounded = offset.coerceIn(0, value.length)
        return value.getSpans(bounded, bounded, SpeakerMarkerSpan::class.java).firstOrNull()
            ?: if (bounded > 0) {
                value.getSpans(bounded - 1, bounded, SpeakerMarkerSpan::class.java).firstOrNull()
            } else null
    }

    private fun snapshotFromEditor(): SpeakerDocumentSnapshot {
        val value = editor.text as? Spannable ?: return SpeakerDocumentSnapshot("", emptyList())
        val pointByStart = value.getSpans(0, value.length, SpeakerMarkerSpan::class.java)
            .associateBy { value.getSpanStart(it) }
        val plain = StringBuilder()
        val points = mutableListOf<SpeakerPoint>()
        for (index in 0 until value.length) {
            val marker = pointByStart[index]
            if (marker != null) {
                points += SpeakerPoint(plain.length, marker.settings)
            } else if (value[index] != MARKER_CHARACTER) {
                plain.append(value[index])
            }
        }
        return SpeakerDocumentSnapshot(plain.toString(), points)
    }

    private fun renderSnapshot(snapshot: SpeakerDocumentSnapshot) {
        val value = SpannableStringBuilder(snapshot.text)
        var shift = 0
        snapshot.points.sortedBy { it.offset }.forEach { point ->
            val displayOffset = (point.offset + shift).coerceIn(0, value.length)
            value.insert(displayOffset, MARKER_TEXT)
            value.setSpan(
                SpeakerMarkerSpan(point.settings, resources.displayMetrics.density),
                displayOffset,
                displayOffset + MARKER_TEXT.length,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE,
            )
            shift += MARKER_TEXT.length
        }
        editor.text = value
        editor.setSelection(value.length)
    }

    private fun restoreState(state: Bundle?) {
        defaultSettings = state?.getString(STATE_DEFAULTS)
            ?.let { runCatching { settingsFromJson(JSONObject(it)) }.getOrNull() }
            ?: FALLBACK_SETTINGS
        lastPointSettings = state?.getString(STATE_LAST_POINT)
            ?.let { runCatching { settingsFromJson(JSONObject(it)) }.getOrNull() }
            ?: defaultSettings
        val savedDocument = state?.getString(STATE_DOCUMENT)
        pendingDefaultWalkthrough = state == null || state.getBoolean(
            STATE_DEFAULT_WALKTHROUGH,
            savedDocument == null,
        )
        pendingSnapshot = savedDocument
            ?.let { runCatching { snapshotFromJson(JSONObject(it)) }.getOrNull() }
    }

    private fun renderPendingSnapshot() {
        pendingSnapshot?.let { snapshot ->
            renderSnapshot(snapshot)
            pendingSnapshot = null
            return
        }
        if (!pendingDefaultWalkthrough) return
        val info = bundleInfo ?: return
        val snapshot = DefaultWalkthrough.snapshot(readExample(DefaultWalkthrough.ASSET_NAME), defaultSettings, info)
        renderSnapshot(snapshot)
        lastPointSettings = snapshot.points.lastOrNull()?.settings ?: defaultSettings
        pendingDefaultWalkthrough = false
    }

    private fun validateSettings(
        settings: ScyllasBandSegmentSettings,
        info: ScyllasBandBundleInfo,
    ): ScyllasBandSegmentSettings {
        val voice = info.voices.firstOrNull { it.id == settings.voiceId } ?: info.voices.first()
        val language = settings.language.takeIf { it in voice.languages } ?: voice.defaultLanguage
        return settings.copy(voiceId = voice.id, language = language)
    }

    private fun simpleAdapter(values: List<String>): ArrayAdapter<String> =
        ArrayAdapter(this, android.R.layout.simple_spinner_item, values).also {
            it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }

    private fun languageLabel(id: String): String = when (id) {
        "en_us" -> getString(R.string.language_en_us)
        "en_gb" -> getString(R.string.language_en_gb)
        "es" -> getString(R.string.language_es)
        "it" -> getString(R.string.language_it)
        "fr" -> getString(R.string.language_fr)
        "de" -> getString(R.string.language_de)
        "vi" -> getString(R.string.language_vi)
        else -> id
    }

    private fun setStatus(value: String) {
        statusText.text = value
    }

    private fun postToUi(action: () -> Unit) {
        mainHandler.post {
            if (!isFinishing && !isDestroyed) action()
        }
    }

    private fun friendlyError(error: Throwable): String =
        error.message?.takeIf { it.isNotBlank() } ?: error.javaClass.simpleName

    private fun settingsToJson(settings: ScyllasBandSegmentSettings) = JSONObject()
        .put("voice", settings.voiceId)
        .put("language", settings.language)
        .put(
            "delivery",
            JSONObject()
                .put("energy", settings.delivery.energy.toDouble())
                .put("tension", settings.delivery.tension.toDouble())
                .put("valence", settings.delivery.valence.toDouble())
                .put("assertiveness", settings.delivery.assertiveness.toDouble())
                .put("whisper", settings.delivery.whisper),
        )

    private fun settingsFromJson(value: JSONObject): ScyllasBandSegmentSettings {
        val delivery = value.getJSONObject("delivery")
        return ScyllasBandSegmentSettings(
            voiceId = value.getString("voice"),
            language = value.getString("language"),
            delivery = ScyllasBandDelivery(
                energy = delivery.optDouble("energy", 2.0).toFloat(),
                tension = delivery.optDouble("tension", 2.0).toFloat(),
                valence = delivery.optDouble("valence", 2.0).toFloat(),
                assertiveness = delivery.optDouble("assertiveness", 2.0).toFloat(),
                whisper = delivery.optBoolean("whisper", false),
            ),
        )
    }

    private fun snapshotToJson(snapshot: SpeakerDocumentSnapshot): JSONObject {
        val points = JSONArray()
        snapshot.points.forEach { point ->
            points.put(
                JSONObject()
                    .put("offset", point.offset)
                    .put("settings", settingsToJson(point.settings)),
            )
        }
        return JSONObject().put("text", snapshot.text).put("points", points)
    }

    private fun snapshotFromJson(value: JSONObject): SpeakerDocumentSnapshot {
        val pointsJson = value.getJSONArray("points")
        val points = buildList {
            for (index in 0 until pointsJson.length()) {
                val point = pointsJson.getJSONObject(index)
                add(SpeakerPoint(point.getInt("offset"), settingsFromJson(point.getJSONObject("settings"))))
            }
        }
        return SpeakerDocumentSnapshot(value.getString("text"), points)
    }

    private companion object {
        const val MARKER_CHARACTER = '\uFFFC'
        const val LOG_TAG = "ScyllasBandPlayback"
        const val MARKER_TEXT = "\uFFFC"
        const val PLAYBACK_SEED = 31_415L

        // CPU: on the Galaxy Z Fold 8 it runs about 27 times faster than real time, and LiteRT's GPU accelerator either
        // fails (AUTO falls back to the CPU) or hangs (GPU). See the README.
        val ACCELERATOR = ScyllasBandAccelerator.CPU
        const val STATE_DOCUMENT = "speaker_document"
        const val STATE_DEFAULT_WALKTHROUGH = "default_walkthrough"
        const val STATE_DEFAULTS = "default_settings"
        const val STATE_LAST_POINT = "last_point_settings"
        val FALLBACK_SETTINGS = ScyllasBandSegmentSettings("scylla", "en_us")
    }
}

private data class ExamplePreset(
    val titleResource: Int,
    val assetName: String,
    val groupSpeak: Boolean,
)
