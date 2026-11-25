// What a v4 or v5 bundle says about singing (controls.delivery_modes), read once by the engine and used by the song planner.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scyllasband {

struct SingingControls {
    std::map<std::string, int64_t> mode_ids;  // delivery mode -> the flow's delivery_mode_id; empty: the bundle does not sing
    std::vector<std::string> modes;           // the modes it sings in (controls.delivery_modes.singing.modes)
    std::string sampler = "heun";
    int steps = 8;
    std::string decoder = "vocos";            // the preferred singing decoder
    double guidance = 1.0;
    double midi_center = 60.0, midi_scale = 24.0;  // the melody layout (note_phase_presence)
    // The MIDI pitch each voice sings around: the manifest's voice_midi_center, else the middle of the voice's spoken pitch
    // edges (which sat within 3 semitones of every voice's measured sung median).
    std::map<std::string, double> voice_center;
    // v5: the score styles are the bundle's own style rows, each its own mode, plus `aliases` (alias, style); lyrics are read
    // and timed the way its sung training rows were, and a note written "x" is a word sung with no pitch.
    bool v5 = false;
    std::vector<std::pair<std::string, std::string>> aliases;
    bool guidance_stated = false;      // the manifest gives singing.guidance (a v5 bundle must, unless the request does)
    std::string guidance_steps = "all";  // the evaluations guidance runs on: all, or first_half (flow time below 0.5)
    std::optional<int> register_floor;   // the lowest MIDI note the register may put a song on (singing.register_floor)
    // Whether the preferred decoder must be there (v5, unless controls.decoders marks it "required": false); otherwise a
    // preferred decoder this backend lacks gives way to vocos.
    bool decoder_required = false;

    bool enabled() const { return !mode_ids.empty(); }
};

}  // namespace scyllasband
