# SPDX-License-Identifier: Apache-2.0
"""Exercise the LI page gather kernel with a host model.

The model checks address ranges and native destination layout. CANN code
generation, device barriers and DMA behavior require an NPU build and run.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "csrc/sparse_prefill_native"


class SelectedLIGatherTest(unittest.TestCase):
    def compiler(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        return compiler

    def test_real_copy_body_skips_full_pages_and_preserves_native_layout(self):
        compiler = self.compiler()
        with tempfile.TemporaryDirectory(prefix="selected-li-body-") as directory:
            build = Path(directory)
            (build / "kernel_operator.h").write_text(
                '#include "selected_bitmap_host_stub.h"\n'
                "#define __global__\n#define KERNEL_TASK_TYPE_DEFAULT(...)\n"
                "namespace AscendC { using TPosition=QuePosition; constexpr int PIPE_ALL=1; "
                "inline uint32_t GetBlockNum(){return 8;} "
                "template<class T> using DataCopyPadExtParams=PaddingParams; }\n"
            )
            # Remove only the SDK launch syntax; keep the device body unchanged.
            body = (
                (SOURCE / "li_page_gather.cpp")
                .read_text()
                .split('extern "C" int32_t launch_dsa_selected_li_pages', 1)[0]
            )
            (build / "body.cpp").write_text(
                body
                + r"""
#include <cassert>
#include <vector>

void Check(uint32_t domain, uint32_t mainOffset, uint32_t dataOffset,
           uint32_t scaleOffset, bool allFull=false) {
    using namespace AscendC;
    regions.clear();
    const uint32_t pages=domain+3, objectBytes=std::max({mainOffset+1,
        dataOffset+DATA_PAGE_BYTES,scaleOffset+SCALE_PAGE_BYTES})+32;
    std::vector<uint8_t> objects(17*objectBytes), k(pages*DATA_PAGE_BYTES,0xa5),
        scale(pages*SCALE_PAGE_BYTES,0x73), expectedK=k, expectedScale=scale;
    for(size_t i=0;i<objects.size();++i) objects[i]=(i*19+i/objectBytes*7)%251;
    std::vector<uint64_t> bases(domain,0);
    for(uint32_t p=0;p<domain;++p) {
        // Native/shared/partial/append boundary pages have zero source entries.
        if(allFull || p%3==1 || p==domain-1) continue;
        auto* object=objects.data()+(p*7%17)*objectBytes;
        bases[p]=reinterpret_cast<uintptr_t>(object)+mainOffset;
        std::memcpy(expectedK.data()+p*DATA_PAGE_BYTES,object+dataOffset,DATA_PAGE_BYTES);
        std::memcpy(expectedScale.data()+p*SCALE_PAGE_BYTES,object+scaleOffset,SCALE_PAGE_BYTES);
    }
    Register(objects.data(),objects.size()); Register(bases.data(),bases.size()*8);
    Register(k.data(),k.size()); Register(scale.data(),scale.size());
    for(blockIndex=0;blockIndex<8;++blockIndex)
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(bases.data()),k.data(),scale.data(),
                           domain,mainOffset,dataOffset,scaleOffset);
    assert(k==expectedK && scale==expectedScale);
}

void Reject(uint64_t source, uint32_t mainOffset, uint32_t dataOffset, uint32_t scaleOffset) {
    using namespace AscendC;
    regions.clear(); blockIndex=0;
    std::vector<uint8_t> k(DATA_PAGE_BYTES,0xa5), scale(SCALE_PAGE_BYTES,0x73);
    Register(&source,sizeof(source)); Register(k.data(),k.size()); Register(scale.data(),scale.size());
    bool rejected=false;
    try {
        dsa_selected_li_pages(reinterpret_cast<uint8_t*>(&source),k.data(),scale.data(),
                           1,mainOffset,dataOffset,scaleOffset);
    } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
    assert(std::all_of(k.begin(),k.end(),[](auto v){return v==0xa5;}));
    assert(std::all_of(scale.begin(),scale.end(),[](auto v){return v==0x73;}));
}

int main() {
    for(uint32_t domain:{1U,7U,8U,9U,33U,4097U}) {
        Check(domain,8192,256,17408);
        Check(domain,8192,256,17408,true);
    }
    Check(37,0,512,18000);  // Main component is first in its registered object.
    Check(37,32768,256,16896);  // LI components can precede the main component.
    Reject(1,4096,8192,24576);
    Reject(4096,4096,8192,24576);  // A null whole-object base is not a valid lease.
    Reject(UINT64_MAX,0,0,32768);
    Reject(UINT64_MAX-32768-SCALE_PAGE_BYTES+1,0,0,32768);  // Scale end alone overflows.
}
"""
            )
            executable = build / "body"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-DASCENDC_DUMP=1",
                    f"-I{build}",
                    f"-I{SOURCE}",
                    f"-I{Path(__file__).parent}",
                    str(build / "body.cpp"),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
