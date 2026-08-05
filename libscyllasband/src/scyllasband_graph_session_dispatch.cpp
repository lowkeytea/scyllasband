/**
 * @file scyllasband_graph_session_dispatch.cpp
 * @brief Public graph-session ABI for Core AI builds.
 *
 * Core AI's Swift implementation exports scyllasband_coreai_session_* and the
 * ONNX translation unit renames itself to scyllasband_onnx_session_* when
 * both backends are compiled together. This translation unit owns the public
 * scyllasband_litert_session_* surface and routes each session by model
 * artifact: *.aimodel directories run on Core AI, everything else on ONNX.
 */

#include "scyllasband.h"
#include "scyllasband_error.h"

#ifdef SCYLLASBAND_WITH_COREAI

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

extern "C" {

void* scyllasband_coreai_session_create(const char* model_path, int32_t accelerator, int32_t max_threads);
void scyllasband_coreai_session_destroy(void* session);
int scyllasband_coreai_session_has_signature(const void* session, const char* signature_name);
int scyllasband_coreai_session_run(
    const void* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);
int scyllasband_coreai_session_run_resized(
    const void* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);

#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
ScyllasBandLiteRtSession* scyllasband_onnx_session_create(const char* model_path, int32_t accelerator, int32_t max_threads);
void scyllasband_onnx_session_destroy(ScyllasBandLiteRtSession* session);
int scyllasband_onnx_session_has_signature(const ScyllasBandLiteRtSession* session, const char* signature_name);
int scyllasband_onnx_session_run(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);
int scyllasband_onnx_session_run_resized(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
);
#endif

}  // extern "C"

namespace {

enum class GraphSessionKind { coreai, onnx };

struct DispatchedGraphSession {
    GraphSessionKind kind;
    void* inner;
};

bool is_coreai_artifact(const char* model_path) {
    if (model_path == nullptr) {
        return false;
    }
    const std::string path(model_path);
    const std::string suffix = ".aimodel";
    return path.size() >= suffix.size() &&
           path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const DispatchedGraphSession* unwrap(const ScyllasBandLiteRtSession* session) {
    return reinterpret_cast<const DispatchedGraphSession*>(session);
}

}  // namespace

extern "C" {

ScyllasBandLiteRtSession* scyllasband_litert_session_create(
    const char* model_path,
    int32_t accelerator,
    int32_t max_threads
) {
    DispatchedGraphSession* wrapper = nullptr;
    if (is_coreai_artifact(model_path)) {
        void* inner = scyllasband_coreai_session_create(model_path, accelerator, max_threads);
        if (inner == nullptr) {
            return nullptr;
        }
        wrapper = new (std::nothrow) DispatchedGraphSession{GraphSessionKind::coreai, inner};
        if (wrapper == nullptr) {
            scyllasband_coreai_session_destroy(inner);
        }
    } else {
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
        ScyllasBandLiteRtSession* inner = scyllasband_onnx_session_create(model_path, accelerator, max_threads);
        if (inner == nullptr) {
            return nullptr;
        }
        wrapper = new (std::nothrow) DispatchedGraphSession{GraphSessionKind::onnx, inner};
        if (wrapper == nullptr) {
            scyllasband_onnx_session_destroy(inner);
        }
#else
        scyllasband_graph_session_set_error(
            "This build executes only Core AI (.aimodel) graphs; rebuild with ONNX Runtime for other artifacts."
        );
        return nullptr;
#endif
    }
    if (wrapper == nullptr) {
        scyllasband_graph_session_set_error("graph session dispatch: allocation failed");
        return nullptr;
    }
    return reinterpret_cast<ScyllasBandLiteRtSession*>(wrapper);
}

void scyllasband_litert_session_destroy(ScyllasBandLiteRtSession* session) {
    if (session == nullptr) {
        return;
    }
    DispatchedGraphSession* wrapper = reinterpret_cast<DispatchedGraphSession*>(session);
    if (wrapper->kind == GraphSessionKind::coreai) {
        scyllasband_coreai_session_destroy(wrapper->inner);
    } else {
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
        scyllasband_onnx_session_destroy(static_cast<ScyllasBandLiteRtSession*>(wrapper->inner));
#endif
    }
    delete wrapper;
}

int scyllasband_litert_session_has_signature(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name
) {
    if (session == nullptr) {
        return 0;
    }
    const DispatchedGraphSession* wrapper = unwrap(session);
    if (wrapper->kind == GraphSessionKind::coreai) {
        return scyllasband_coreai_session_has_signature(wrapper->inner, signature_name);
    }
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
    return scyllasband_onnx_session_has_signature(
        static_cast<const ScyllasBandLiteRtSession*>(wrapper->inner), signature_name);
#else
    return 0;
#endif
}

int scyllasband_litert_session_run(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    if (session == nullptr) {
        scyllasband_graph_session_set_error("graph session dispatch: session is required");
        return -1;
    }
    const DispatchedGraphSession* wrapper = unwrap(session);
    if (wrapper->kind == GraphSessionKind::coreai) {
        return scyllasband_coreai_session_run(
            wrapper->inner, signature_name, inputs, input_count, out_tensors, out_tensor_count);
    }
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
    return scyllasband_onnx_session_run(
        static_cast<const ScyllasBandLiteRtSession*>(wrapper->inner),
        signature_name, inputs, input_count, out_tensors, out_tensor_count);
#else
    return -1;
#endif
}

int scyllasband_litert_session_run_resized(
    const ScyllasBandLiteRtSession* session,
    const char* signature_name,
    const ScyllasBandTensorView* inputs,
    int32_t input_count,
    ScyllasBandOwnedTensor** out_tensors,
    int32_t* out_tensor_count
) {
    if (session == nullptr) {
        scyllasband_graph_session_set_error("graph session dispatch: session is required");
        return -1;
    }
    const DispatchedGraphSession* wrapper = unwrap(session);
    if (wrapper->kind == GraphSessionKind::coreai) {
        return scyllasband_coreai_session_run_resized(
            wrapper->inner, signature_name, inputs, input_count, out_tensors, out_tensor_count);
    }
#ifdef SCYLLASBAND_WITH_ONNXRUNTIME
    return scyllasband_onnx_session_run_resized(
        static_cast<const ScyllasBandLiteRtSession*>(wrapper->inner),
        signature_name, inputs, input_count, out_tensors, out_tensor_count);
#else
    return -1;
#endif
}

}  // extern "C"

#endif  // SCYLLASBAND_WITH_COREAI
