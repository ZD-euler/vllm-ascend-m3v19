// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "li_page_gather.h"

namespace {
using namespace SparsePrefillLI;

void LIPageGather(const at::Tensor& mainPageBases, const at::Tensor& targetK,
                  const at::Tensor& targetScale, int64_t mainOffset,
                  int64_t dataOffset, int64_t scaleOffset)
{
    TORCH_CHECK(mainPageBases.device().is_privateuseone() &&
                mainPageBases.scalar_type() == at::kUInt64 && mainPageBases.dim() == 1 &&
                mainPageBases.numel() > 0 && mainPageBases.numel() <= UINT32_MAX,
                "LI source must be an NPU uint64 page-base table");
    for (const auto& tensor : {mainPageBases, targetK, targetScale}) {
        TORCH_CHECK(tensor.device() == mainPageBases.device() && tensor.is_contiguous(),
                    "LI source and native caches must be contiguous on one NPU");
    }
    TORCH_CHECK(targetK.scalar_type() == at::kChar && targetK.dim() >= 2 &&
                targetScale.scalar_type() == at::kHalf && targetScale.dim() >= 2 &&
                targetK.size(0) > 0 && targetK.size(0) <= UINT32_MAX &&
                targetScale.size(0) == targetK.size(0) &&
                targetK.numel() / targetK.size(0) == DATA_PAGE_BYTES &&
                targetScale.numel() / targetScale.size(0) == SCALE_PAGE_BYTES / 2 &&
                mainPageBases.numel() <= targetK.size(0),
                "LI destination must retain native 16384-byte K and 256-byte scale pages");
    TORCH_CHECK(mainOffset >= 0 && mainOffset <= UINT32_MAX && dataOffset >= 0 &&
                dataOffset <= UINT32_MAX - DATA_PAGE_BYTES && scaleOffset >= 0 &&
                scaleOffset <= UINT32_MAX - SCALE_PAGE_BYTES,
                "LI component offsets exceed the registered object geometry");
    c10_npu::NPUGuard guard(mainPageBases.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("DsaLIPageGather", [=]() -> int {
        return launch_dsa_selected_li_pages(stream, mainPageBases.data_ptr(), targetK.data_ptr(),
            targetScale.data_ptr(), mainPageBases.numel(), targetK.size(0),
            mainOffset, dataOffset, scaleOffset);
    });
}
} // namespace

TORCH_LIBRARY_FRAGMENT(dsa_prefill_queue, m) {
    m.def("li_page_gather(Tensor main_page_bases, Tensor(a!) target_k, Tensor(b!) target_scale, int main_offset, int data_offset, int scale_offset) -> ()");
}
TORCH_LIBRARY_IMPL(dsa_prefill_queue, PrivateUse1, m) {
    m.impl("li_page_gather", LIPageGather);
}
