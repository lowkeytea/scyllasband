/* ABI 3.x stays compatible with 3.0.0: the 3.0.0 types keep their layout and enum values (fixtures/abi_3_0_layout.txt,
   taken from the 3.0.0 header on a 64-bit platform) and the library still exports every 3.0.0 symbol
   (fixtures/abi_3_0_symbols.txt). Usage: scyllasband_abi_test <libscyllasband_native> <fixtures dir> */
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "scyllasband.h"

#define FIELD(type, field) \
    used += snprintf(out + used, size - used, "  %-28s offset %3zu size %zu\n", #field, offsetof(type, field), sizeof(((type*)0)->field))

static size_t layout(char* out, size_t size) {
    size_t used = 0;
    used += snprintf(out + used, size - used, "ScyllasBandRuntimeOptions %zu\n", sizeof(ScyllasBandRuntimeOptions));
    FIELD(ScyllasBandRuntimeOptions, bundle_dir); FIELD(ScyllasBandRuntimeOptions, threads); FIELD(ScyllasBandRuntimeOptions, accelerator);
    used += snprintf(out + used, size - used, "ScyllasBandRequest %zu\n", sizeof(ScyllasBandRequest));
    FIELD(ScyllasBandRequest, text); FIELD(ScyllasBandRequest, voice_id); FIELD(ScyllasBandRequest, language); FIELD(ScyllasBandRequest, delivery);
    FIELD(ScyllasBandRequest, speed); FIELD(ScyllasBandRequest, steps); FIELD(ScyllasBandRequest, sampler); FIELD(ScyllasBandRequest, seed);
    FIELD(ScyllasBandRequest, has_seed); FIELD(ScyllasBandRequest, temperature); FIELD(ScyllasBandRequest, normalize_text);
    used += snprintf(out + used, size - used, "ScyllasBandAudio %zu\n", sizeof(ScyllasBandAudio));
    FIELD(ScyllasBandAudio, samples); FIELD(ScyllasBandAudio, sample_count); FIELD(ScyllasBandAudio, sample_rate); FIELD(ScyllasBandAudio, metadata_json);
    used += snprintf(out + used, size - used, "ScyllasBandEvent %zu\n", sizeof(ScyllasBandEvent));
    FIELD(ScyllasBandEvent, type); FIELD(ScyllasBandEvent, chunk_index); FIELD(ScyllasBandEvent, chunk_count); FIELD(ScyllasBandEvent, chunk_id);
    FIELD(ScyllasBandEvent, metadata_json); FIELD(ScyllasBandEvent, samples); FIELD(ScyllasBandEvent, sample_count); FIELD(ScyllasBandEvent, sample_rate);
    used += snprintf(out + used, size - used, "enums status %d %d %d %d accelerator %d %d %d %d sampler %d %d %d events %d %d %d %d\n",
                     SCYLLASBAND_OK, SCYLLASBAND_ERROR_INVALID_ARGUMENT, SCYLLASBAND_ERROR_RUNTIME, SCYLLASBAND_ERROR_CANCELLED,
                     SCYLLASBAND_ACCELERATOR_CPU, SCYLLASBAND_ACCELERATOR_GPU, SCYLLASBAND_ACCELERATOR_AUTO, SCYLLASBAND_ACCELERATOR_NEURAL_ENGINE,
                     SCYLLASBAND_SAMPLER_DEFAULT, SCYLLASBAND_SAMPLER_HEUN, SCYLLASBAND_SAMPLER_EULER, SCYLLASBAND_EVENT_PLAN,
                     SCYLLASBAND_EVENT_CHUNK_STARTED, SCYLLASBAND_EVENT_AUDIO, SCYLLASBAND_EVENT_DONE);
    return used;
}

static size_t read_file(const char* dir, const char* name, char* out, size_t size) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    size_t count = fread(out, 1, size - 1, file);
    fclose(file);
    out[count] = '\0';
    return count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: scyllasband_abi_test <library> <fixtures dir>\n");
        return 2;
    }
    int failures = 0;
    static char got[8192], want[8192], symbols[8192];
    if (sizeof(void*) == 8) {   /* the golden layout is a 64-bit one */
        layout(got, sizeof(got));
        if (!read_file(argv[2], "abi_3_0_layout.txt", want, sizeof(want)) || strcmp(got, want) != 0) {
            fprintf(stderr, "FAIL 3.0.0 layouts differ from fixtures/abi_3_0_layout.txt:\n%s", got);
            ++failures;
        }
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "FAIL cannot load %s: %s\n", argv[1], dlerror());
        return 1;
    }
    if (!read_file(argv[2], "abi_3_0_symbols.txt", symbols, sizeof(symbols))) {
        fprintf(stderr, "FAIL no fixtures/abi_3_0_symbols.txt\n");
        return 1;
    }
    int checked = 0;
    for (char* name = strtok(symbols, "\n"); name; name = strtok(NULL, "\n")) {
        ++checked;
        if (!dlsym(library, name)) {
            fprintf(stderr, "FAIL 3.0.0 symbol %s is not exported\n", name);
            ++failures;
        }
    }
    const char* (*version)(void) = (const char* (*)(void))dlsym(library, "scyllasband_version");
    if (version && strncmp(version(), "3.", 2) != 0) {
        fprintf(stderr, "FAIL version %s is not 3.x\n", version());
        ++failures;
    }
    if (failures) return 1;
    printf("abi: 3.0.0 layouts and %d symbols ok (library %s)\n", checked, version ? version() : "?");
    return 0;
}
