// What a v4 bundle says about singing (controls.delivery_modes), read once by the engine and used by the song planner.
#pragma once

#include <cstdint>
#include <map>
#include <string>
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

    bool enabled() const { return !mode_ids.empty(); }
};

}  // namespace scyllasband
