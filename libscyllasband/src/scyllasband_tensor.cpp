#include "scyllasband.h"

#include <cstdlib>

uint64_t scyllasband_tensor_element_size(int32_t data_type) {
    switch (data_type) {
        case SCYLLASBAND_TENSOR_FLOAT32:
            return 4;
        case SCYLLASBAND_TENSOR_FLOAT16:
            return 2;
        case SCYLLASBAND_TENSOR_FLOAT64:
            return 8;
        case SCYLLASBAND_TENSOR_INT64:
            return 8;
        case SCYLLASBAND_TENSOR_INT32:
            return 4;
        case SCYLLASBAND_TENSOR_INT16:
            return 2;
        case SCYLLASBAND_TENSOR_INT8:
            return 1;
        case SCYLLASBAND_TENSOR_UINT8:
            return 1;
        case SCYLLASBAND_TENSOR_BOOL:
            return 1;
        default:
            return 0;
    }
}

void scyllasband_tensors_destroy(ScyllasBandOwnedTensor* tensors, int32_t tensor_count) {
    if (tensors == nullptr) {
        return;
    }
    for (int32_t index = 0; index < tensor_count; ++index) {
        std::free(tensors[index].name);
        std::free(tensors[index].shape);
        std::free(tensors[index].data);
        tensors[index].name = nullptr;
        tensors[index].shape = nullptr;
        tensors[index].data = nullptr;
        tensors[index].rank = 0;
        tensors[index].byte_length = 0;
    }
    std::free(tensors);
}
