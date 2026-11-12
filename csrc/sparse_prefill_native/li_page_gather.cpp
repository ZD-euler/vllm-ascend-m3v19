// SPDX-License-Identifier: Apache-2.0
// Full LI pages share the original main-KV page IDs and whole-object lease.
#include "selected_kv_assert.h"
#include "li_page_gather.h"

using namespace AscendC;
using namespace SparsePrefillLI;

extern "C" __global__ __aicore__ void dsa_selected_li_pages(
    GM_ADDR mainPageBases, GM_ADDR targetK, GM_ADDR targetScale,
    uint32_t domain, uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint64_t> bases;
    GlobalTensor<uint8_t> destinationK, destinationScale, source;
    bases.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(mainPageBases));
    destinationK.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(targetK));
    destinationScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(targetScale));
    TPipe pipe;
    TBuf<TPosition::VECCALC> bytes;
    pipe.InitBuffer(bytes, DATA_PAGE_BYTES);
    auto local = bytes.Get<uint8_t>();
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    for (uint64_t page = GetBlockIdx(); page < domain; page += GetBlockNum()) {
        const uint64_t main = bases.GetValue(page);
        // FULL/shared/partial/append pages are left to the original SDK load.
        if (main == 0) continue;
        ASSERT_MSG(main > mainOffset, "Invalid leased LI object base");
        if (main <= mainOffset) return;
        const uint64_t blob = main - mainOffset;
        const bool valid = blob <= UINT64_MAX - dataOffset - DATA_PAGE_BYTES &&
            blob <= UINT64_MAX - scaleOffset - SCALE_PAGE_BYTES;
        ASSERT_MSG(valid, "LI component address overflows");
        if (!valid) return;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + dataOffset));
        DataCopyPad(local, source, {1, DATA_PAGE_BYTES, 0, 0, 0}, pad);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(destinationK[page * DATA_PAGE_BYTES], local,
                    {1, DATA_PAGE_BYTES, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + scaleOffset));
        DataCopyPad(local, source, {1, SCALE_PAGE_BYTES, 0, 0, 0}, pad);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(destinationScale[page * SCALE_PAGE_BYTES], local,
                    {1, SCALE_PAGE_BYTES, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int32_t launch_dsa_selected_li_pages(void* stream, void* mainPageBases,
    void* targetK, void* targetScale, uint32_t domain, uint32_t destinationPages,
    uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset)
{
    if (!stream || !mainPageBases || !targetK || !targetScale || domain == 0U ||
        domain > destinationPages || dataOffset > UINT32_MAX - DATA_PAGE_BYTES ||
        scaleOffset > UINT32_MAX - SCALE_PAGE_BYTES) return -1;
    dsa_selected_li_pages<<<8, nullptr, stream>>>(
        static_cast<uint8_t*>(mainPageBases), static_cast<uint8_t*>(targetK),
        static_cast<uint8_t*>(targetScale), domain, mainOffset, dataOffset, scaleOffset);
    return 0;
}
