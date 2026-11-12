// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace SparsePrefillLI {
constexpr uint32_t DATA_PAGE_BYTES = 16384U, SCALE_PAGE_BYTES = 256U;
}

extern "C" int32_t launch_dsa_selected_li_pages(void* stream, void* mainPageBases,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset);
