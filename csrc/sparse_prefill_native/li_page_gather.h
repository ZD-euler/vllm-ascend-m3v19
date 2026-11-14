// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace SparsePrefillLI {
// Two logical-page contracts are supported:
//   M3v19 native KV:             16384 B data, 256 B scale, 16384 B stride
//   target packed mixed indexer: 32768 B data, 512 B scale, 32768 B stride
// The UB staging buffer and the launch bounds use the larger contract.
constexpr uint32_t MAX_DATA_PAGE_BYTES = 32768U;
constexpr uint32_t MAX_SCALE_PAGE_BYTES = 512U;
constexpr uint32_t PAGE_ALIGN_BYTES = 32U;
}

extern "C" int32_t launch_dsa_selected_li_pages(void* stream, void* mainPageBases,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset,
    uint32_t dataBytes, uint32_t scaleBytes, uint32_t destinationStrideBytes);
