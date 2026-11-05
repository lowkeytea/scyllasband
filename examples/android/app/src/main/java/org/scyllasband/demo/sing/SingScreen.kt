package org.scyllasband.demo.sing

import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.GridLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.PopupMenu
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.MaterialColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONObject
import org.scyllasband.android.ScyllasBand
import org.scyllasband.android.ScyllasBandBundleInfo
import org.scyllasband.android.ScyllasBandScoreException
import org.scyllasband.android.ScyllasBandSingOptions
import org.scyllasband.android.ScyllasBandSingingDecoder
import org.scyllasband.demo.FloatAudioPlayer
import org.scyllasband.demo.R
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt

/** What the Sing screen shares with the activity: the runtime, its worker thread, and the claim on the runtime. */
interface SingHost {
    val runtime: ScyllasBand
    val worker: ExecutorService
    val bundleInfo: ScyllasBandBundleInfo?

    /** Claims the runtime for [task] ("singing", "previewing"); false while speech or another task holds it. */
    fun begin(task: String): Boolean
    fun end(task: String)
}

/**
 * The Sing screen: a song being written (lines of note chips), its fit to the voice, a tone preview, and sung takes.
 *
 * Every check of the song goes through the runtime, so the app never re-implements the score rules: the score is validated
 * with [ScyllasBand.parseScore], its range read with `singFit` and previewed with `preview` (no graph runs, so they answer at
 * once even while a take renders, on their own thread), and it is sung with `singStreaming` on the activity's worker.
 */
class SingScreen(
    private val activity: AppCompatActivity,
    private val root: View,
    private val host: SingHost,
    savedState: Bundle?,
) {
    private data class Selection(val line: String, val note: String)
    private data class Take(val key: String, val samples: FloatArray, val sampleRate: Int)

    private val inflater = LayoutInflater.from(activity)
    private val store = SongStore(activity)
    private val checks: ExecutorService = Executors.newSingleThreadExecutor()
    private val examples: List<ExampleScore> =
        ExampleScore.sorted(activity.assets.list(MELODIES).orEmpty().toList())

    private val examplesButton: MaterialButton = root.findViewById(R.id.examplesButton)
    private val songsButton: MaterialButton = root.findViewById(R.id.songsButton)
    private val runtimeSummary: TextView = root.findViewById(R.id.singRuntimeSummary)
    private val notice: TextView = root.findViewById(R.id.singNotice)
    private val songCard: MaterialCardView = root.findViewById(R.id.songCard)
    private val songTitle: TextView = root.findViewById(R.id.songTitle)
    private val songDetails: TextView = root.findViewById(R.id.songDetails)
    private val songFit: TextView = root.findViewById(R.id.songFit)
    private val linesContainer: LinearLayout = root.findViewById(R.id.linesContainer)
    private val addLineButton: MaterialButton = root.findViewById(R.id.addLineButton)
    private val soundToggle: MaterialButtonToggleGroup = root.findViewById(R.id.soundToggle)
    private var renderingSound = false
    private val previewButton: MaterialButton = root.findViewById(R.id.previewButton)
    private val singButton: MaterialButton = root.findViewById(R.id.singButton)
    private val anotherTakeButton: MaterialButton = root.findViewById(R.id.anotherTakeButton)
    private val saveButton: MaterialButton = root.findViewById(R.id.saveSongButton)
    private val shareButton: MaterialButton = root.findViewById(R.id.shareSongButton)
    private val progressBar: ProgressBar = root.findViewById(R.id.singProgressBar)
    private val progressText: TextView = root.findViewById(R.id.singProgress)
    private val statusText: TextView = root.findViewById(R.id.singStatus)

    private val accent = MaterialColors.getColor(root, com.google.android.material.R.attr.colorPrimary, Color.BLUE)
    private val errorColor = MaterialColors.getColor(root, com.google.android.material.R.attr.colorError, Color.RED)
    private val outline = activity.getColor(R.color.scyllasband_outline)
    private val density = activity.resources.displayMetrics.density

    private var song: SongDocument = savedState?.getString(STATE_SONG)
        ?.let { runCatching { SongDocument.fromJson(JSONObject(it)) }.getOrNull() }
        ?: SongDocument.blank("gwen")
    private var selection: Selection? = null
    private var problem: String? = null
    private var problemNote: String? = null
    private var fitSummary = ""
    private var suggestedTranspose: Int? = null
    private var singing = false
    private var previewing = false
    private var take: Take? = null
    private var takeSeed = 31_415L
    private var checkGeneration = 0
    private var warmedUp = false
    private val stopFlag = AtomicBoolean(false)

    private var inspector: BottomSheetDialog? = null
    private var inspectorView: View? = null
    private var inspectorOctave = 4
    private var settingsView: View? = null

    init {
        examplesButton.setOnClickListener { showExamples() }
        songsButton.setOnClickListener { showSongs() }
        songCard.setOnClickListener { showSettings() }
        addLineButton.setOnClickListener { change(SongEditing.addLine(song)) }
        soundToggle.addOnButtonCheckedListener { _, id, checked ->
            if (checked && !renderingSound) change(song.copy(decoder = if (id == R.id.soundQuality) SongDecoder.QUALITY else SongDecoder.TURBO))
        }
        previewButton.setOnClickListener { preview() }
        singButton.setOnClickListener { singOrStop() }
        anotherTakeButton.setOnClickListener { anotherTake() }
        saveButton.setOnClickListener { save() }
        shareButton.setOnClickListener { share() }
        setStatus(activity.getString(R.string.sing_status_start))
        render()
    }

    private val info: ScyllasBandBundleInfo? get() = host.bundleInfo
    private val canSing: Boolean get() = info?.singing?.supported == true
    private val hasBigVgan: Boolean get() = info?.singing?.hasBigVgan == true
    private val busy: Boolean get() = singing || previewing

    // Lifecycle

    /**
     * Debug builds: what an `adb shell am start` asked for (extras `sing_example` <melody name>, `sing_turbo`, `autosing`),
     * applied once the runtime is ready; `autosing` logs the outcome as "SBSING <status>" (tag ScyllasBandSing).
     */
    var launchOptions: Bundle? = null

    /** The runtime is loaded: pick a voice it has and, for a fresh screen, the first whole-song example. */
    fun onRuntimeReady() {
        val bundle = info ?: return
        if (bundle.voices.none { it.id == song.voice }) song = song.copy(voice = bundle.voices.firstOrNull()?.id ?: song.voice)
        runtimeSummary.text = "${bundle.backend} · ${bundle.accelerator}"
        if (!bundle.singing.supported) {
            setStatus(activity.getString(R.string.sing_speech_only))
        } else if (song.title == SongDocument.blank(song.voice).title) {
            examples.firstOrNull { it.voice != null }?.let { load(it) }
        }
        songChanged()
        applyLaunchOptions()
    }

    private fun applyLaunchOptions() {
        val options = launchOptions ?: return
        launchOptions = null
        options.getString("sing_example")?.let { name -> examples.firstOrNull { it.assetName == "$name.json" }?.let { load(it) } }
        if (options.getBoolean("sing_turbo")) change(song.copy(decoder = SongDecoder.TURBO))
        if (options.getBoolean("autosing")) {
            autosing = true
            root.postDelayed({ singOrStop() }, 1000)
        }
    }

    private var autosing = false

    /** The tab became visible: load the singing graphs once (the GPU program compiles on the first launch). */
    fun onShown() {
        if (warmedUp || !canSing) return
        warmedUp = true
        val voice = song.voice
        host.worker.execute {
            runCatching { host.runtime.warmUpSinging(voice, ScyllasBandSingingDecoder.DEFAULT) }
                .onFailure { Log.w(TAG, "Singing warm-up failed: ${it.message}") }
        }
    }

    fun saveState(out: Bundle) {
        out.putString(STATE_SONG, song.toJson().toString())
    }

    fun stop() {
        stopFlag.set(true)
        if (singing) host.runtime.cancel()
    }

    fun close() {
        stop()
        inspector?.dismiss()
        checks.shutdown()
        checks.awaitTermination(2, TimeUnit.SECONDS)
    }

    // Editing

    private fun change(updated: SongDocument) {
        if (updated == song) return
        song = updated
        songChanged()
    }

    private fun editSelected(edit: (SongDocument, String, String) -> SongDocument) {
        val target = selection ?: return
        change(edit(song, target.line, target.note))
    }

    /** After every change: the screen redraws, the take is kept only for the song it was sung from, and the runtime checks it. */
    private fun songChanged() {
        render()
        refreshInspector()
        refreshSettings()
        val generation = ++checkGeneration
        val current = song
        val bundle = info ?: return
        if (!bundle.singing.supported) return
        val score = current.score()
        checks.execute {
            SystemClock.sleep(CHECK_DELAY_MS)
            if (generation != checkGeneration) return@execute
            var objection: String? = null
            var row: Int? = null
            var fit: JSONObject? = null
            try {
                ScyllasBand.parseScore(score.json, 0)
                fit = host.runtime.singFit(score.json, current.voice, null, options(current))
            } catch (error: ScyllasBandScoreException) {
                objection = error.message
                row = error.noteIndex
            } catch (error: Exception) {
                objection = error.message ?: error.javaClass.simpleName
            }
            root.post {
                if (generation != checkGeneration) return@post
                problem = objection
                problemNote = row?.let { score.rows.getOrNull(it) }
                updateFit(fit)
                render()
                refreshSettings()
            }
        }
    }

    private fun updateFit(fit: JSONObject?) {
        if (fit == null) {
            fitSummary = ""
            suggestedTranspose = null
            return
        }
        val low = fit.optString("low_note")
        val high = fit.optString("high_note")
        val outside = fit.optDouble("outside_semitones", 0.0)
        val voice = displayName(song.voice)
        val suggested = fit.optInt("suggested_transpose", song.transpose)
        if (!fit.isNull("voice_center") && outside > 0) {
            val direction = activity.getString(
                if (fit.optDouble("mean_midi") > fit.optDouble("voice_center")) R.string.sing_fit_high else R.string.sing_fit_low,
            )
            fitSummary = activity.getString(R.string.sing_fit_outside, low, high, formatNumber(outside, 1), direction, voice)
            suggestedTranspose = suggested.takeIf { it != song.transpose }
        } else {
            fitSummary = activity.getString(R.string.sing_fit_in_range, low, high, voice)
            suggestedTranspose = null
        }
    }

    private fun options(current: SongDocument = song, seed: Long? = null) = ScyllasBandSingOptions(
        decoder = if (current.decoder == SongDecoder.TURBO || !hasBigVgan) ScyllasBandSingingDecoder.VOCOS else ScyllasBandSingingDecoder.BIGVGAN,
        transpose = current.transpose,
        seed = seed,
    )

    // Rendering

    private fun render() {
        val bundle = info
        notice.visibility = if (bundle != null && !bundle.singing.supported) View.VISIBLE else View.GONE
        songTitle.text = song.title
        val transpose = if (song.transpose == 0) "" else " · transpose ${signed(song.transpose)}"
        songDetails.text = "${displayName(song.voice)} · ${song.style.title} · ${song.bpm.roundToInt()} bpm · ${song.decoder.title}$transpose"
        songFit.visibility = if (fitSummary.isEmpty()) View.GONE else View.VISIBLE
        songFit.text = fitSummary
        songFit.setTextColor(if (suggestedTranspose == null) songDetails.currentTextColor else activity.getColor(android.R.color.holo_orange_dark))
        renderLines()
        renderingSound = true
        soundToggle.check(if (song.decoder == SongDecoder.QUALITY && hasBigVgan) R.id.soundQuality else R.id.soundTurbo)
        renderingSound = false
        for (i in 0 until soundToggle.childCount) soundToggle.getChildAt(i).isEnabled = hasBigVgan && !singing
        val ready = bundle != null && canSing
        previewButton.text = activity.getString(if (previewing) R.string.stop else R.string.sing_preview)
        previewButton.isEnabled = ready && !singing && (previewing || problem == null)
        singButton.text = activity.getString(if (singing) R.string.stop else R.string.sing_sing)
        singButton.isEnabled = ready && (busy || problem == null)
        anotherTakeButton.isEnabled = ready && !busy
        addLineButton.isEnabled = !singing
        examplesButton.isEnabled = !singing && examples.isNotEmpty()
        songsButton.isEnabled = !singing
        songCard.isEnabled = !singing
        if (problem != null && !busy) setStatus(problem.orEmpty())
    }

    private fun renderLines() {
        while (linesContainer.childCount > song.lines.size) linesContainer.removeViewAt(linesContainer.childCount - 1)
        while (linesContainer.childCount < song.lines.size) linesContainer.addView(newLineCard())
        song.lines.forEachIndexed { index, line -> bindLine(linesContainer.getChildAt(index), index, line) }
    }

    private fun newLineCard(): View {
        val card = inflater.inflate(R.layout.item_song_line, linesContainer, false)
        val lyrics = card.findViewById<EditText>(R.id.lineLyrics)
        val apply = card.findViewById<MaterialButton>(R.id.lineApply)
        lyrics.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: Editable?) {
                val line = song.lines.firstOrNull { it.id == card.tag } ?: return
                apply.visibility = if (s?.toString() != line.lyrics) View.VISIBLE else View.GONE
            }
        })
        val applyLyrics = {
            (card.tag as? String)?.let { id ->
                lyrics.clearFocus()
                hideKeyboard(lyrics)
                change(SongEditing.updateLyrics(song, id, lyrics.text.toString()))
            }
        }
        apply.setOnClickListener { applyLyrics() }
        lyrics.setOnEditorActionListener { _, action, _ ->
            if (action == EditorInfo.IME_ACTION_DONE) {
                applyLyrics()
                true
            } else {
                false
            }
        }
        card.findViewById<MaterialButton>(R.id.lineOptions).setOnClickListener { button ->
            (card.tag as? String)?.let { showLineOptions(button, it) }
        }
        return card
    }

    private fun bindLine(card: View, index: Int, line: SongLine) {
        val rebound = card.tag != line.id
        card.tag = line.id
        card.findViewById<TextView>(R.id.lineTitle).text = activity.getString(R.string.sing_line, index + 1)
        val lyrics = card.findViewById<EditText>(R.id.lineLyrics)
        if (rebound || !lyrics.hasFocus()) {
            if (lyrics.text.toString() != line.lyrics) lyrics.setText(line.lyrics)
            card.findViewById<View>(R.id.lineApply).visibility = View.GONE
        }
        lyrics.isEnabled = !singing
        card.findViewById<View>(R.id.lineOptions).isEnabled = !singing
        card.findViewById<ContourView>(R.id.lineContour).setNotes(line.notes, selection?.note, problemNote)
        val notes = card.findViewById<LinearLayout>(R.id.lineNotes)
        notes.removeAllViews()
        line.notes.forEach { note -> notes.addView(noteChip(line, note)) }
        val restAfter = card.findViewById<TextView>(R.id.lineRestAfter)
        val isLast = index == song.lines.size - 1
        restAfter.visibility = if (isLast) View.GONE else View.VISIBLE
        restAfter.text = activity.getString(R.string.sing_rest_after, beatsTitle(max(line.restAfter, song.minimumLineRest)))
    }

    private fun noteChip(line: SongLine, note: SongNote): View {
        val chip = inflater.inflate(R.layout.item_note_chip, null, false) as MaterialCardView
        val selected = selection?.note == note.id
        val isProblem = problemNote == note.id
        chip.layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            .apply { marginEnd = (6 * density).roundToInt() }
        chip.findViewById<LinearLayout>(R.id.chipBody).minimumWidth = (max(48.0, 36.0 * note.beats) * density).roundToInt()
        chip.findViewById<TextView>(R.id.chipSyllable).text = when {
            note.isRest -> activity.getString(R.string.sing_rest)
            note.syllable.isEmpty() -> "–"
            else -> note.syllable
        }
        chip.findViewById<TextView>(R.id.chipPitch).text = note.midi?.let(SongDocument::noteName) ?: " "
        chip.findViewById<TextView>(R.id.chipLength).text = lengthSymbol(note.beats)
        chip.setCardBackgroundColor(
            if (note.isRest) Color.argb(28, 128, 128, 128) else Color.argb(31, Color.red(accent), Color.green(accent), Color.blue(accent)),
        )
        chip.strokeColor = when {
            isProblem -> errorColor
            selected -> accent
            else -> Color.argb(120, Color.red(outline), Color.green(outline), Color.blue(outline))
        }
        chip.strokeWidth = ((if (selected || isProblem) 2f else 0.75f) * density).roundToInt()
        chip.contentDescription = if (note.isRest) {
            "Rest, ${beatsTitle(note.beats)}"
        } else {
            "${note.syllable.ifEmpty { "held" }}, ${SongDocument.noteName(note.midi ?: 60)}, ${beatsTitle(note.beats)}"
        }
        chip.isEnabled = !singing
        chip.setOnClickListener {
            selection = Selection(line.id, note.id)
            note.midi?.let { inspectorOctave = it / 12 - 1 }
            render()
            showInspector()
        }
        return chip
    }

    private fun showLineOptions(anchor: View, lineId: String) {
        val line = song.lines.firstOrNull { it.id == lineId } ?: return
        val isLast = song.lines.last().id == lineId
        val menu = PopupMenu(activity, anchor)
        val choices = listOf(song.minimumLineRest, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0).filter { it >= song.minimumLineRest }.distinct().sorted()
        if (!isLast) {
            val rests = menu.menu.addSubMenu(activity.getString(R.string.sing_rest_after_menu))
            choices.forEachIndexed { index, beats ->
                rests.add(0, index, index, beatsTitle(beats)).apply {
                    isCheckable = true
                    isChecked = beats == line.restAfter
                }
            }
        }
        menu.menu.add(1, 0, choices.size, activity.getString(R.string.sing_delete_line)).isEnabled = song.lines.size > 1
        menu.setOnMenuItemClickListener { item ->
            when (item.groupId) {
                0 -> change(SongEditing.setRestAfter(song, lineId, choices[item.itemId]))
                1 -> {
                    if (selection?.line == lineId) selection = null
                    change(SongEditing.removeLine(song, lineId))
                }
            }
            true
        }
        menu.show()
    }

    // Note inspector

    private fun showInspector() {
        if (inspector?.isShowing == true) {
            refreshInspector()
            return
        }
        val view = inflater.inflate(R.layout.sheet_note_inspector, null, false)
        inspectorView = view
        val sheet = BottomSheetDialog(activity)
        sheet.setContentView(view)
        sheet.setOnDismissListener {
            inspector = null
            inspectorView = null
            selection = null
            render()
        }
        view.findViewById<View>(R.id.inspectorDone).setOnClickListener { sheet.dismiss() }
        val syllable = view.findViewById<EditText>(R.id.inspectorSyllable)
        syllable.setOnEditorActionListener { _, action, _ ->
            if (action == EditorInfo.IME_ACTION_DONE) {
                editSelected { s, l, n -> SongEditing.setSyllable(s, l, n, syllable.text.toString()) }
                hideKeyboard(syllable)
                true
            } else {
                false
            }
        }
        syllable.setOnFocusChangeListener { _, focused ->
            if (!focused) editSelected { s, l, n -> SongEditing.setSyllable(s, l, n, syllable.text.toString()) }
        }
        view.findViewById<PianoKeysView>(R.id.inspectorKeys).onPick = { midi ->
            editSelected { s, l, n -> SongEditing.setPitch(s, l, n, midi) }
        }
        view.findViewById<View>(R.id.inspectorOctaveDown).setOnClickListener {
            inspectorOctave = (inspectorOctave - 1).coerceAtLeast(1)
            refreshInspector()
        }
        view.findViewById<View>(R.id.inspectorOctaveUp).setOnClickListener {
            inspectorOctave = (inspectorOctave + 1).coerceAtMost(6)
            refreshInspector()
        }
        val grid = view.findViewById<GridLayout>(R.id.inspectorLengths)
        LENGTHS.forEach { beats ->
            val button = MaterialButton(activity, null, com.google.android.material.R.attr.materialButtonOutlinedStyle)
            button.text = lengthSymbol(beats)
            button.tag = beats
            button.contentDescription = beatsTitle(beats)
            button.setOnClickListener { editSelected { s, l, n -> SongEditing.setBeats(s, l, n, beats) } }
            grid.addView(button, GridLayout.LayoutParams(GridLayout.spec(GridLayout.UNDEFINED), GridLayout.spec(GridLayout.UNDEFINED, 1f)).apply {
                width = 0
                setMargins((3 * density).roundToInt(), 0, (3 * density).roundToInt(), 0)
            })
        }
        view.findViewById<View>(R.id.inspectorToggleRest).setOnClickListener { editSelected(SongEditing::toggleRest) }
        view.findViewById<View>(R.id.inspectorSplit).setOnClickListener { editSelected(SongEditing::split) }
        view.findViewById<View>(R.id.inspectorJoin).setOnClickListener { editSelected(SongEditing::mergeWithNext) }
        view.findViewById<View>(R.id.inspectorInsertRest).setOnClickListener { editSelected(SongEditing::insertRestAfter) }
        view.findViewById<View>(R.id.inspectorDelete).setOnClickListener {
            editSelected(SongEditing::delete)
            sheet.dismiss()
        }
        inspector = sheet
        refreshInspector()
        sheet.show()
    }

    private fun refreshInspector() {
        val view = inspectorView ?: return
        val target = selection ?: return
        val note = SongEditing.note(song, target.line, target.note) ?: run {
            inspector?.dismiss()
            return
        }
        view.findViewById<View>(R.id.inspectorSungSection).visibility = if (note.isRest) View.GONE else View.VISIBLE
        val syllable = view.findViewById<EditText>(R.id.inspectorSyllable)
        if (!syllable.hasFocus() && syllable.text.toString() != note.syllable) syllable.setText(note.syllable)
        view.findViewById<TextView>(R.id.inspectorWordHint).setText(
            if (note.continuesWord) R.string.sing_continues_word else R.string.sing_starts_word,
        )
        view.findViewById<TextView>(R.id.inspectorPitchTitle).text =
            activity.getString(R.string.sing_pitch, note.midi?.let(SongDocument::noteName).orEmpty())
        view.findViewById<PianoKeysView>(R.id.inspectorKeys).apply {
            octave = inspectorOctave
            selected = note.midi
        }
        view.findViewById<TextView>(R.id.inspectorOctave).text = activity.getString(R.string.sing_octave, inspectorOctave)
        view.findViewById<TextView>(R.id.inspectorLengthTitle).text = activity.getString(R.string.sing_length, beatsTitle(note.beats))
        val grid = view.findViewById<GridLayout>(R.id.inspectorLengths)
        for (i in 0 until grid.childCount) {
            val button = grid.getChildAt(i) as MaterialButton
            val current = button.tag == note.beats
            button.strokeColor = ColorStateList.valueOf(if (current) accent else outline)
            button.strokeWidth = ((if (current) 2f else 1f) * density).roundToInt()
        }
        view.findViewById<MaterialButton>(R.id.inspectorToggleRest).setText(if (note.isRest) R.string.sing_make_note else R.string.sing_make_rest)
        view.findViewById<View>(R.id.inspectorSplit).visibility = if (note.isRest) View.GONE else View.VISIBLE
    }

    // Song settings

    private fun showSettings() {
        val bundle = info
        val view = inflater.inflate(R.layout.dialog_song_settings, null, false)
        settingsView = view
        val title = view.findViewById<EditText>(R.id.settingsTitle)
        title.setText(song.title)
        val voices = bundle?.voices.orEmpty()
        val voiceSpinner = view.findViewById<Spinner>(R.id.settingsVoice)
        voiceSpinner.adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, voices.map { it.displayName })
        voices.indexOfFirst { it.id == song.voice }.takeIf { it >= 0 }?.let { voiceSpinner.setSelection(it, false) }
        voiceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, item: View?, position: Int, id: Long) {
                voices.getOrNull(position)?.let { if (it.id != song.voice) change(song.copy(voice = it.id)) }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        view.findViewById<MaterialButtonToggleGroup>(R.id.settingsStyle).apply {
            check(if (song.style == SongStyle.POP) R.id.settingsStylePop else R.id.settingsStyleBallad)
            addOnButtonCheckedListener { _, id, checked ->
                if (checked) change(song.copy(style = if (id == R.id.settingsStylePop) SongStyle.POP else SongStyle.BALLAD))
            }
        }
        view.findViewById<View>(R.id.settingsTempoDown).setOnClickListener { change(song.copy(bpm = (song.bpm - 2).coerceAtLeast(40.0))) }
        view.findViewById<View>(R.id.settingsTempoUp).setOnClickListener { change(song.copy(bpm = (song.bpm + 2).coerceAtMost(200.0))) }
        view.findViewById<MaterialButtonToggleGroup>(R.id.settingsSound).apply {
            check(if (song.decoder == SongDecoder.QUALITY && hasBigVgan) R.id.settingsSoundQuality else R.id.settingsSoundTurbo)
            isEnabled = hasBigVgan
            for (i in 0 until childCount) getChildAt(i).isEnabled = hasBigVgan
            addOnButtonCheckedListener { _, id, checked ->
                if (checked) change(song.copy(decoder = if (id == R.id.settingsSoundQuality) SongDecoder.QUALITY else SongDecoder.TURBO))
            }
        }
        view.findViewById<TextView>(R.id.settingsSoundHelp).setText(if (hasBigVgan) R.string.sing_sound_help else R.string.sing_sound_vocos_only)
        view.findViewById<View>(R.id.settingsTransposeDown).setOnClickListener { change(song.copy(transpose = (song.transpose - 1).coerceAtLeast(-12))) }
        view.findViewById<View>(R.id.settingsTransposeUp).setOnClickListener { change(song.copy(transpose = (song.transpose + 1).coerceAtMost(12))) }
        view.findViewById<View>(R.id.settingsUseSuggested).setOnClickListener {
            suggestedTranspose?.let { change(song.copy(transpose = it)) }
        }
        refreshSettings()
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.sing_song)
            .setView(view)
            .setPositiveButton(R.string.sing_done, null)
            .setOnDismissListener {
                val typed = title.text.toString().trim()
                settingsView = null
                if (typed.isNotEmpty() && typed != song.title) change(song.copy(title = typed)) else render()
            }
            .show()
    }

    private fun refreshSettings() {
        val view = settingsView ?: return
        view.findViewById<TextView>(R.id.settingsTempo).text = activity.getString(R.string.sing_tempo, song.bpm.roundToInt())
        view.findViewById<TextView>(R.id.settingsTranspose).text = activity.getString(R.string.sing_transpose, signed(song.transpose))
        view.findViewById<TextView>(R.id.settingsFit).text = fitSummary
        view.findViewById<MaterialButton>(R.id.settingsUseSuggested).apply {
            visibility = if (suggestedTranspose != null) View.VISIBLE else View.GONE
            text = suggestedTranspose?.let { activity.getString(R.string.sing_use_transpose, signed(it)) }
        }
    }

    // Examples and saved songs

    private fun showExamples() {
        val menu = PopupMenu(activity, examplesButton)
        examples.forEachIndexed { index, example -> menu.menu.add(0, index, index, example.title) }
        menu.setOnMenuItemClickListener { item ->
            load(examples[item.itemId])
            true
        }
        menu.show()
    }

    private fun load(example: ExampleScore) {
        try {
            val text = activity.assets.open("$MELODIES/${example.assetName}").bufferedReader().use { it.readText() }
            val summary = ScyllasBand.parseScore(text, 0)
            val voice = example.voice?.takeIf { id -> info?.voices?.any { it.id == id } == true } ?: song.voice
            val loaded = SongDocument.fromSummary(summary, example.title, voice) ?: return
            stop()
            replaceSong(loaded)
            setStatus(activity.getString(R.string.sing_status_loaded, example.title, loaded.lines.size, loaded.noteCount))
        } catch (error: Exception) {
            setStatus(activity.getString(R.string.sing_status_load_failed, example.title, error.message ?: error.javaClass.simpleName))
        }
    }

    private fun showSongs() {
        val saved = store.saved()
        val menu = PopupMenu(activity, songsButton)
        menu.menu.add(0, NEW_SONG, 0, R.string.sing_new_song)
        saved.forEachIndexed { index, item -> menu.menu.add(1, index, index + 1, item.title) }
        if (saved.isNotEmpty()) menu.menu.add(2, DELETE_SONG, saved.size + 1, R.string.sing_delete_saved)
        menu.setOnMenuItemClickListener { item ->
            when (item.groupId) {
                0 -> {
                    stop()
                    replaceSong(SongDocument.blank(info?.voices?.firstOrNull()?.id ?: song.voice))
                    setStatus(activity.getString(R.string.sing_status_new))
                }
                1 -> {
                    stop()
                    replaceSong(saved[item.itemId])
                    setStatus(activity.getString(R.string.sing_status_opened, saved[item.itemId].title))
                }
                2 -> confirmDelete(saved)
            }
            true
        }
        menu.show()
    }

    private fun confirmDelete(saved: List<SongDocument>) {
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.sing_delete_saved)
            .setItems(saved.map { it.title }.toTypedArray()) { _, index ->
                store.delete(saved[index])
                setStatus(activity.getString(R.string.sing_status_deleted, saved[index].title))
            }
            .show()
    }

    private fun replaceSong(updated: SongDocument) {
        song = updated
        selection = null
        take = null
        songChanged()
    }

    private fun save() {
        runCatching { store.save(song) }
            .onSuccess { setStatus(activity.getString(R.string.sing_status_saved, song.title)) }
            .onFailure { setStatus(activity.getString(R.string.sing_status_save_failed, it.message.orEmpty())) }
    }

    private fun share() {
        runCatching {
            val current = take?.takeIf { it.key == takeKey() }
            store.shareIntent(song, current?.samples, current?.sampleRate ?: (info?.sampleRate ?: SAMPLE_RATE))
        }.onSuccess { intent ->
            activity.startActivity(android.content.Intent.createChooser(intent, activity.getString(R.string.sing_share_title)))
        }.onFailure { setStatus(activity.getString(R.string.sing_status_share_failed, it.message.orEmpty())) }
    }

    // Preview and singing

    /** Plays the melody as tones at the pitch it will be sung; no model runs, so it starts at once. */
    private fun preview() {
        if (previewing) {
            stop()
            return
        }
        val bundle = info ?: return
        if (problem != null || !host.begin(TASK_PREVIEW)) return
        val json = song.scoreJson()
        val voice = song.voice
        val settings = options()
        stopFlag.set(false)
        previewing = true
        setStatus(activity.getString(R.string.sing_status_previewing))
        render()
        checks.execute {
            val failure = runCatching {
                val tones = host.runtime.preview(json, voice, settings)
                FloatAudioPlayer(bundle.sampleRate).use { player ->
                    player.enqueue(tones, {}, { stopFlag.get() })
                    player.finish { stopFlag.get() }
                }
            }.exceptionOrNull()
            root.post {
                previewing = false
                host.end(TASK_PREVIEW)
                setStatus(
                    failure?.let { activity.getString(R.string.sing_status_preview_failed, it.message.orEmpty()) }
                        ?: activity.getString(if (stopFlag.get()) R.string.sing_status_stopped else R.string.sing_status_preview_done),
                )
                render()
            }
        }
    }

    private fun takeKey(seed: Long = takeSeed): String =
        "${song.scoreJson()}|${song.voice}|${song.transpose}|${song.decoder.id}|$seed"

    /**
     * Sings the song. With Turbo, or when the first phrase renders faster than real time, it plays while the rest renders;
     * otherwise it renders the whole song, then plays. The take is kept until the song changes.
     */
    private fun singOrStop() {
        if (busy) {
            stop()
            return
        }
        val bundle = info ?: run {
            setStatus(activity.getString(R.string.sing_status_wait))
            return
        }
        if (!canSing) return
        problem?.let {
            setStatus(it)
            return
        }
        if (!host.begin(TASK_SING)) {
            setStatus(activity.getString(R.string.sing_status_busy))
            return
        }
        val key = takeKey()
        stopFlag.set(false)
        singing = true
        take?.takeIf { it.key == key }?.let { last ->
            replay(last)
            return
        }
        val json = song.scoreJson()
        val voice = song.voice
        val settings = options(seed = takeSeed)
        val turbo = settings.decoder == ScyllasBandSingingDecoder.VOCOS
        val sampleRate = bundle.sampleRate
        setProgress(activity.getString(R.string.sing_progress_preparing))
        setStatus(activity.getString(if (turbo) R.string.sing_status_singing_turbo else R.string.sing_status_singing_quality))
        render()
        val started = SystemClock.elapsedRealtime()
        host.worker.execute {
            val pieces = ArrayList<FloatArray>()
            val buffered = ArrayList<FloatArray>()
            var sampleCount = 0
            var streaming = turbo
            var decided = turbo
            var finished = false
            val failure = runCatching {
                // A queue as long as the song, so rendering never waits for playback (the take is kept whole anyway).
                FloatAudioPlayer(sampleRate, SONG_CHUNKS).use { player ->
                    val sung = host.runtime.singStreaming(json, voice, null, settings, { index, count, text ->
                        root.post {
                            setProgress(
                                if (text.isNullOrBlank()) {
                                    activity.getString(R.string.sing_progress_phrase, index + 1, count)
                                } else {
                                    activity.getString(R.string.sing_progress_phrase_text, index + 1, count, text)
                                },
                            )
                        }
                        !stopFlag.get()
                    }) { phrase ->
                        pieces += phrase.samples
                        sampleCount += phrase.samples.size
                        if (!decided) {
                            // Stream from now on only if rendering keeps ahead of playback.
                            val rendered = (SystemClock.elapsedRealtime() - started) / 1000.0
                            streaming = rendered < sampleCount.toDouble() / sampleRate * STREAM_MARGIN
                            decided = true
                            if (!streaming) root.post { setProgress(activity.getString(R.string.sing_progress_rendering)) }
                        }
                        if (streaming) {
                            for (pending in buffered + listOf(phrase.samples)) {
                                if (!player.enqueue(pending, {}, { stopFlag.get() })) return@singStreaming false
                            }
                            buffered.clear()
                        } else {
                            buffered += phrase.samples
                        }
                        !stopFlag.get()
                    }
                    val renderSeconds = (SystemClock.elapsedRealtime() - started) / 1000.0
                    val audioSeconds = sampleCount.toDouble() / sampleRate
                    Log.i(TAG, "Sang %.1f s in %.1f s".format(audioSeconds, renderSeconds))
                    if (sung && !stopFlag.get()) {
                        val samples = joinPieces(pieces, sampleCount)
                        root.post {
                            take = Take(key, samples, sampleRate)
                            setProgress(activity.getString(R.string.sing_progress_playing))
                            setStatus(activity.getString(R.string.sing_status_sang, audioSeconds, renderSeconds))
                        }
                        for (pending in buffered) if (!player.enqueue(pending, {}, { stopFlag.get() })) break
                        player.finish { stopFlag.get() }
                        finished = true
                    }
                }
            }.exceptionOrNull()
            root.post { finishSinging(cancelled = stopFlag.get() || (!finished && failure == null), failure = failure) }
        }
    }

    private fun replay(last: Take) {
        setStatus(activity.getString(R.string.sing_status_replaying))
        setProgress(activity.getString(R.string.sing_progress_playing))
        render()
        checks.execute {
            val failure = runCatching {
                FloatAudioPlayer(last.sampleRate).use { player ->
                    player.enqueue(last.samples, {}, { stopFlag.get() })
                    player.finish { stopFlag.get() }
                }
            }.exceptionOrNull()
            root.post { finishSinging(cancelled = stopFlag.get(), failure = failure) }
        }
    }

    private fun finishSinging(cancelled: Boolean, failure: Throwable?) {
        singing = false
        host.end(TASK_SING)
        setProgress(null)
        when {
            failure != null && !cancelled -> setStatus(activity.getString(R.string.sing_status_failed, failure.message ?: failure.javaClass.simpleName))
            cancelled -> setStatus(activity.getString(R.string.sing_status_stopped))
        }
        render()
        if (autosing) {
            autosing = false
            Log.i(TAG, "SBSING ${statusText.text}")
        }
    }

    /** A new seed: the next Sing renders a different take of the same song. */
    private fun anotherTake() {
        takeSeed += 1
        take = null
        setStatus(activity.getString(R.string.sing_status_new_take))
    }

    // Helpers

    private fun setStatus(text: String) {
        statusText.text = text
    }

    private fun setProgress(text: String?) {
        progressBar.visibility = if (text == null) View.GONE else View.VISIBLE
        progressText.visibility = if (text == null) View.GONE else View.VISIBLE
        progressText.text = text.orEmpty()
    }

    private fun displayName(voice: String): String =
        info?.voices?.firstOrNull { it.id == voice }?.displayName ?: voice.replaceFirstChar { it.uppercase() }

    private fun beatsTitle(beats: Double): String =
        if (beats == 1.0) activity.getString(R.string.sing_one_beat) else activity.getString(R.string.sing_beats, formatNumber(beats, 2))

    private fun hideKeyboard(view: View) {
        activity.getSystemService(InputMethodManager::class.java)?.hideSoftInputFromWindow(view.windowToken, 0)
    }

    private companion object {
        const val TAG = "ScyllasBandSing"
        const val MELODIES = "scyllasband/melodies"
        const val STATE_SONG = "sing_song"
        const val TASK_SING = "singing"
        const val TASK_PREVIEW = "previewing"
        const val CHECK_DELAY_MS = 150L
        const val SAMPLE_RATE = 24_000
        const val STREAM_MARGIN = 0.9
        const val SONG_CHUNKS = 512
        const val NEW_SONG = -1
        const val DELETE_SONG = -2
        val LENGTHS = listOf(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

        fun signed(value: Int): String = if (value > 0) "+$value" else value.toString()

        fun formatNumber(value: Double, decimals: Int): String =
            "%.${decimals}f".format(value).trimEnd('0').trimEnd('.')

        /** 1/16, 1/8, 1/8., 1/4, 1/4., 1/2, 1/2., 1 (beats are quarter notes). */
        fun lengthSymbol(beats: Double): String = when (beats) {
            0.25 -> "1/16"
            0.5 -> "1/8"
            0.75 -> "1/8."
            1.0 -> "1/4"
            1.5 -> "1/4."
            2.0 -> "1/2"
            3.0 -> "1/2."
            4.0 -> "1"
            else -> formatNumber(beats, 2)
        }

        fun joinPieces(pieces: List<FloatArray>, size: Int): FloatArray {
            val out = FloatArray(size)
            var at = 0
            for (piece in pieces) {
                piece.copyInto(out, at)
                at += piece.size
            }
            return out
        }
    }
}
