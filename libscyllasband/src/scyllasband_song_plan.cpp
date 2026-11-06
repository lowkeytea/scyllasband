#include "scyllasband_song_plan.h"

#include "scyllasband_numeric.h"
#include "scyllasband_unicode.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace scyllasband {

namespace {

constexpr int kMaxConsonantFrames = 8;
// Vowel symbols of the phone vocabulary (the first character of a phone decides): the same set the trainer uses.
const unicode::Text kVowels = unicode::decode("aeiouyɑɐɒæəɚɛɜɝɪɔʊʌɨʉøœɯɤɵɘɞɶ");
const char* const kNeedsV4 = "Singing needs a v4 bundle with delivery modes (controls.delivery_modes); this bundle has none";

// Median share of a word's acoustic frames its vowels take, measured by forced alignment on the training corpus: sung 0.51,
// rap 0.42 (rap at a higher share stretched vowels and squeezed consonants).
double vowel_share_of(const std::string& mode) {
    if (mode == "rap") return 0.42;
    return 0.51;   // sung, ballad, spoken
}

std::string strip(const std::string& text) { return unicode::encode(unicode::strip(unicode::decode(text))); }
std::string lower(const std::string& text) { return unicode::encode(unicode::lower(unicode::decode(text))); }

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? separator : "") + parts[i];
    return out;
}

std::string quote(const std::string& text) { return "'" + text + "'"; }

Json note_json(const ScoreNote& note) {
    Json out{Json::Object{}};
    out.set("label", Json(note.label));
    out.set("note", Json(note.name()));
    out.set("start", Json(round_to(note.start, 4)));
    out.set("end", Json(round_to(note.end, 4)));
    return out;
}

void check_style(const SingingControls& controls, const Score& score) {
    const std::vector<std::string> allowed = singing_styles(controls);
    if (std::find(allowed.begin(), allowed.end(), score.style) == allowed.end()) {
        throw ScoreError("Style " + quote(score.style) + " (" + score.mode + ") is not a singing style of this bundle; choose one of " + join(allowed, ", "));
    }
}

std::optional<double> voice_center(const SingingControls& controls, const std::string& voice) {
    auto found = controls.voice_center.find(voice);
    return found == controls.voice_center.end() ? std::nullopt : std::optional<double>(found->second);
}

}  // namespace

std::vector<std::string> singing_modes(const SingingControls& controls) {
    std::vector<std::string> listed;
    for (const std::string& mode : controls.modes) {
        if (controls.mode_ids.count(mode)) listed.push_back(mode);
    }
    if (!listed.empty()) return listed;
    for (const auto& [mode, id] : controls.mode_ids) {   // std::map: sorted, as Python's sorted()
        if (mode != "spoken") listed.push_back(mode);
    }
    return listed;
}

std::vector<std::string> singing_styles(const SingingControls& controls) {
    const std::vector<std::string> modes = singing_modes(controls);
    std::vector<std::string> out;
    for (const auto& [style, mode] : style_modes()) {
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) out.push_back(style);
    }
    return out;
}

// --- phones, durations and melody ----------------------------------------------------------------------------------------------

std::pair<std::vector<std::string>, std::vector<int>> word_phones(const std::vector<std::string>& words, SongFrontend& frontend,
                                                                  const std::string& language) {
    // The phrase is phonemized as one text so the G2P sees its context; when its words do not line up one-to-one with the
    // score's, each word is phonemized on its own, so the count always matches.
    const Phonemized result = frontend.phonemize(join(words, " "), language);
    // Punctuation phones are dropped (a score has none; the G2P may still produce some), with their silence when it duplicates a
    // neighbouring one, keeping the word starts on the same phones.
    std::vector<std::string> kept;
    std::vector<int> starts;
    const std::set<int> drop(result.word_starts.begin(), result.word_starts.end());
    for (std::size_t index = 0; index < result.phones.size(); ++index) {
        const std::string& phone = result.phones[index];
        if (is_punctuation_pause(phone) || (phone == kSilencePhone && !kept.empty() && kept.back() == kSilencePhone)) continue;
        if (drop.count(static_cast<int>(index))) starts.push_back(static_cast<int>(kept.size()));
        kept.push_back(phone);
    }
    if (starts.size() + 1 == words.size()) return {kept, starts};
    std::vector<std::string> phones = {kSilencePhone};
    starts.clear();
    for (const std::string& word : words) {
        const Phonemized single = frontend.phonemize(word, language);
        std::vector<std::string> inner;
        for (const std::string& phone : single.phones) {
            if (phone != kSilencePhone && !is_punctuation_pause(phone)) inner.push_back(phone);
        }
        if (inner.empty()) throw ScoreError("No phones for the word " + quote(word));
        if (phones.size() > 1) {
            phones.push_back(kSilencePhone);
            starts.push_back(static_cast<int>(phones.size()));
        }
        phones.insert(phones.end(), inner.begin(), inner.end());
    }
    phones.push_back(kSilencePhone);
    return {phones, starts};
}

bool is_vowel(const std::string& phone) {
    if (phone.empty() || phone[0] == '<') return false;
    const unicode::Text text = unicode::decode(phone);
    return !text.empty() && kVowels.find(text[0]) != unicode::Text::npos;
}

std::vector<int> phone_durations(const std::vector<std::string>& phones, const std::vector<int>& word_starts, const std::vector<ScoreWord>& words,
                                 int64_t total_frames, double vowel_share, int hop, int sample_rate) {
    std::size_t first = 0;
    while (first < phones.size() && phones[first] == kSilencePhone) ++first;
    std::vector<std::size_t> starts = {first};
    for (int start : word_starts) starts.push_back(static_cast<std::size_t>(start));
    starts.push_back(phones.size());
    if (starts.size() - 1 != words.size()) {
        std::vector<std::string> texts;
        for (const ScoreWord& word : words) texts.push_back(word.text);
        throw ScoreError(std::to_string(starts.size() - 1) + " phonemized words for " + std::to_string(words.size()) + " score words: " + join(texts, " "));
    }
    std::vector<std::pair<int64_t, int64_t>> bounds;
    for (const ScoreWord& word : words) {
        bounds.emplace_back(round_half_even(word.start * sample_rate / hop), round_half_even(word.end * sample_rate / hop));
    }
    std::vector<int64_t> durations(phones.size(), 0);
    for (std::size_t i = 0; i < first; ++i) {
        if (phones[i] == kSilencePhone) durations[i] = bounds[0].first;
    }
    for (std::size_t gi = 0; gi + 1 < starts.size(); ++gi) {
        const std::size_t g0 = starts[gi], g1 = starts[gi + 1];
        const int64_t f0 = bounds[gi].first, f1 = bounds[gi].second;
        std::vector<std::size_t> vowels, consonants;
        int64_t acoustic = 0;
        for (std::size_t i = g0; i < g1; ++i) {
            const std::string& phone = phones[i];
            if (phone == kSilencePhone || is_punctuation_pause(phone) || is_non_acoustic_modifier(phone)) continue;
            ++acoustic;
            (is_vowel(phone) ? vowels : consonants).push_back(i);
        }
        const int64_t n = std::max(f1 - f0, acoustic);
        const int64_t consonant_count = static_cast<int64_t>(consonants.size());
        int64_t c = 0;
        if (!vowels.empty()) {
            c = std::min<int64_t>(kMaxConsonantFrames, std::max<int64_t>(1, round_half_even(static_cast<double>(n) * (1 - vowel_share) /
                                                                                            static_cast<double>(std::max<int64_t>(1, consonant_count)))));
        } else {
            c = std::max<int64_t>(1, floor_div(n, std::max<int64_t>(1, consonant_count)));
        }
        for (std::size_t i : consonants) durations[i] = c;
        const int64_t left = n - c * consonant_count;
        if (!vowels.empty()) {
            const int64_t count = static_cast<int64_t>(vowels.size());
            const int64_t share = floor_div(left, count), extra = floor_mod(left, count);
            for (std::size_t k = 0; k < vowels.size(); ++k) durations[vowels[k]] = std::max<int64_t>(1, share + (static_cast<int64_t>(k) < extra ? 1 : 0));
        } else if (!consonants.empty()) {
            durations[consonants.back()] += left;
        }
        const int64_t following = gi + 1 < bounds.size() ? bounds[gi + 1].first : std::max(f1 + 2, total_frames);
        std::optional<std::size_t> last_silence;
        for (std::size_t i = g0; i < g1; ++i) {
            if (phones[i] == kSilencePhone) last_silence = i;
        }
        if (last_silence) durations[*last_silence] = std::max<int64_t>(0, following - (f0 + n));
    }
    return std::vector<int>(durations.begin(), durations.end());
}

std::vector<float> frame_melody(const std::vector<ScoreNote>& notes, int64_t frames, int hop, int sample_rate, double midi_center, double midi_scale) {
    const std::size_t T = static_cast<std::size_t>(frames);
    std::vector<float> out(T * 3, 0.0f);
    std::vector<double> centres(T);
    for (std::size_t t = 0; t < T; ++t) centres[t] = (static_cast<double>(t) * hop + static_cast<double>(hop / 2)) / sample_rate;
    for (const ScoreNote& note : notes) {
        if (!note.midi || note.end <= note.start) continue;
        const float pitch = static_cast<float>((*note.midi - midi_center) / midi_scale);
        for (std::size_t t = 0; t < T; ++t) {
            if (centres[t] >= note.start && centres[t] < note.end) {
                out[t * 3] = pitch;
                out[t * 3 + 1] = static_cast<float>((centres[t] - note.start) / (note.end - note.start));
                out[t * 3 + 2] = 1.0f;
            }
        }
    }
    return out;
}

// --- the plan ----------------------------------------------------------------------------------------------------------------

Json PhrasePlan::to_json(bool with_melody) const {
    Json out{Json::Object{}};
    out.set("index", Json(index));
    out.set("start", Json(round_to(phrase.start, 4)));
    out.set("end", Json(round_to(phrase.end, 4)));
    out.set("text", Json(phrase.text));
    out.set("frames", Json(frames));
    out.set("phones", Json::array_of(sentence.phones));
    out.set("durations", Json::array_of(durations));
    Json::Array note_items;
    for (const ScoreNote& note : phrase.notes) note_items.push_back(note_json(note));
    out.set("notes", Json(std::move(note_items)));
    if (with_melody) {
        Json::Array values;
        values.reserve(melody.size());
        for (float value : melody) values.emplace_back(static_cast<double>(value));
        out.set("melody", Json(std::move(values)));
    }
    return out;
}

double SongPlan::seconds() const {
    double out = 0.0;
    for (const PhrasePlan& phrase : phrases) out = std::max(out, phrase.phrase.end);
    return out;
}

Json SongPlan::to_json(bool with_melody) const {
    Json out{Json::Object{}};
    out.set("voice", Json(voice));
    out.set("language", Json(language));
    out.set("mode", Json(mode));
    out.set("mode_id", Json(mode_id));
    out.set("register_shift", Json(register_shift));
    out.set("transpose", Json(transpose));
    out.set("bpm", Json(score.bpm));
    out.set("style", Json(score.style));
    out.set("text", Json(score.text));
    out.set("seconds", Json(round_to(seconds(), 3)));
    Json::Array items;
    for (const PhrasePlan& phrase : phrases) items.push_back(phrase.to_json(with_melody));
    out.set("phrases", Json(std::move(items)));
    return out;
}

SongPlan plan_song(const Score& input, SongFrontend& frontend, const std::string& voice, const std::string& language, const PlanSettings& settings) {
    const SingingControls& controls = frontend.singing();
    if (!controls.enabled()) throw std::invalid_argument(kNeedsV4);
    check_style(controls, input);
    const int sample_rate = frontend.sample_rate(), hop = frontend.latent_hop();
    const int64_t max_frames = frontend.max_phrase_frames();
    Score score = input.transposed(settings.transpose);
    const int shift = settings.use_register ? register_shift(score, voice_center(controls, voice)) : 0;
    score = score.transposed(shift);
    const double longest = static_cast<double>((max_frames - 2) * hop) / sample_rate - 2 * settings.pad;
    const double max_seconds = std::min(settings.max_seconds.value_or(kMaxPhraseSeconds), longest);
    const double share = settings.vowel_share.value_or(vowel_share_of(score.mode));
    SongPlan plan;
    plan.voice = voice;
    plan.language = language;
    plan.mode = score.mode;
    plan.mode_id = controls.mode_ids.at(score.mode);
    plan.register_shift = shift;
    plan.transpose = settings.transpose;
    const std::vector<Phrase> phrases = split_phrases(score, settings.phrase_rest, max_seconds, settings.pad);
    for (std::size_t index = 0; index < phrases.size(); ++index) {
        const Phrase& phrase = phrases[index];
        std::vector<std::string> words;
        for (const ScoreWord& word : phrase.words) words.push_back(word.text);
        auto [phones, starts] = word_phones(words, frontend, language);
        const int64_t total = round_half_even(phrase.notes.back().end * sample_rate / hop) + 2;
        std::vector<int> durations = phone_durations(phones, starts, phrase.words, total, share, hop, sample_rate);
        int64_t frames = 0;
        for (int value : durations) frames += value;
        if (frames > max_frames) {
            throw OverlongError("Phrase " + std::to_string(index + 1) + " ('" + unicode::encode(unicode::decode(phrase.text).substr(0, 40)) + "') needs " +
                                      std::to_string(frames) + " latent frames, but this bundle holds at most " + std::to_string(max_frames) +
                                      "; add a rest inside it or shorten its notes");
        }
        PhrasePlan item;
        item.index = static_cast<int>(index);
        item.phrase = phrase;
        item.sentence.phones = std::move(phones);
        item.sentence.word_starts = std::move(starts);
        item.durations = std::move(durations);
        item.frames = frames;
        item.melody = frame_melody(phrase.notes, frames, hop, sample_rate, controls.midi_center, controls.midi_scale);
        plan.phrases.push_back(std::move(item));
    }
    plan.score = std::move(score);
    return plan;
}

// --- requests ------------------------------------------------------------------------------------------------------------------

SingSettings resolve_sing_settings(const SongFrontend& frontend, const SingRequest& request) {
    const SingingControls& singing = frontend.singing();
    SingSettings out;
    if (request.steps < 0) throw std::invalid_argument("steps must be positive");
    out.steps = request.steps > 0 ? request.steps : (singing.steps > 0 ? singing.steps : 8);
    out.sampler = lower(request.sampler.empty() ? (singing.sampler.empty() ? std::string("heun") : singing.sampler) : request.sampler);
    if (out.sampler != "heun" && out.sampler != "euler") throw std::invalid_argument("Unsupported sampler '" + out.sampler + "'; choose heun or euler");
    const std::string preferred = singing.decoder.empty() ? std::string("vocos") : singing.decoder;
    out.decoder = request.decoder.empty() ? (frontend.has_decoder(preferred) ? preferred : std::string("vocos")) : lower(request.decoder);
    if (!frontend.has_decoder(out.decoder)) {
        throw std::invalid_argument("This bundle has no '" + out.decoder + "' decoder; it has " + join(frontend.available_decoders(), ", "));
    }
    out.guidance = request.guidance.value_or(singing.guidance);
    if (!std::isfinite(out.guidance)) throw std::invalid_argument("guidance must be finite");
    return out;
}

ResolvedSing resolve_sing_request(SongFrontend& frontend, const Score& score, const SingRequest& request) {
    if (!frontend.singing().enabled()) throw std::invalid_argument(kNeedsV4);
    ResolvedSing out;
    out.voice = strip(request.voice);
    if (out.voice.empty()) throw std::invalid_argument("A voice is required");
    out.language = frontend.resolve_language(out.voice, request.language);
    check_style(frontend.singing(), score);
    out.score = score;
    out.settings = resolve_sing_settings(frontend, request);
    return out;
}

// --- fit -----------------------------------------------------------------------------------------------------------------------

Json SongFit::to_json() const {
    Json out{Json::Object{}};
    out.set("voice_center", voice_center ? Json(*voice_center) : Json());
    Json band{Json::Array{}};
    if (voice_center) {
        band.push(Json(band_low));
        band.push(Json(band_high));
    }
    out.set("band", band);
    out.set("transpose", Json(transpose));
    out.set("register_shift", Json(register_shift));
    out.set("mean_midi", Json(round_to(mean_midi, 3)));
    out.set("low_midi", Json(low_midi));
    out.set("high_midi", Json(high_midi));
    out.set("low_note", Json(midi_to_note(low_midi)));
    out.set("high_note", Json(midi_to_note(high_midi)));
    out.set("outside_semitones", Json(round_to(outside_semitones, 3)));
    out.set("suggested_transpose", Json(suggested_transpose));
    return out;
}

SongFit song_fit(const SingingControls& controls, const Score& score, const std::string& voice, int transpose, bool use_register) {
    SongFit fit;
    fit.voice_center = voice_center(controls, voice);
    fit.transpose = transpose;
    auto sung_mean = [&](int t, int* shift_out) {
        const Score moved = score.transposed(t);
        const int shift = use_register ? register_shift(moved, fit.voice_center) : 0;
        if (shift_out) *shift_out = shift;
        const std::vector<int> pitched = moved.pitched();
        long long sum = 0;
        for (int midi : pitched) sum += midi;
        return pitched.empty() ? 0.0 : static_cast<double>(sum) / static_cast<double>(pitched.size()) + shift;
    };
    fit.mean_midi = sung_mean(transpose, &fit.register_shift);
    const std::vector<int> pitched = score.transposed(transpose + fit.register_shift).pitched();
    if (!pitched.empty()) {
        fit.low_midi = *std::min_element(pitched.begin(), pitched.end());
        fit.high_midi = *std::max_element(pitched.begin(), pitched.end());
    }
    fit.suggested_transpose = transpose;
    if (fit.voice_center && !pitched.empty()) {
        fit.band_low = *fit.voice_center - kRegisterBelow;
        fit.band_high = *fit.voice_center + kRegisterAbove;
        auto outside = [&](double mean) { return std::max({0.0, fit.band_low - mean, mean - fit.band_high}); };
        fit.outside_semitones = outside(fit.mean_midi);
        if (fit.outside_semitones > 0.0) {
            double best = INFINITY;
            for (int step = 0; step <= 12; ++step) {   // 0, 1, -1, 2, -2, ... 6, -6: the first that fits, else the nearest
                const int candidate = step % 2 ? (step + 1) / 2 : -(step / 2);
                const double distance = outside(sung_mean(candidate, nullptr));
                if (distance < best - 1e-9) {
                    best = distance;
                    fit.suggested_transpose = candidate;
                }
                if (distance == 0.0) break;
            }
        }
    }
    return fit;
}

}  // namespace scyllasband
