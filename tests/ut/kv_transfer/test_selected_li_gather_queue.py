# SPDX-License-Identifier: Apache-2.0
"""Exercise the LI queue wrapper with modeled torch_npu calls."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "csrc/sparse_prefill_native"


class SelectedLIGatherQueueTest(unittest.TestCase):
    def compiler(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        return compiler

    def test_real_binding_accepts_both_page_layouts_and_keeps_current_stream_and_tensors(self):
        import torch
        from torch.utils.cpp_extension import include_paths

        compiler = self.compiler()
        with tempfile.TemporaryDirectory(prefix="selected-li-binding-") as directory:
            build = Path(directory)
            npu = build / "torch_npu/csrc"
            (npu / "core/npu").mkdir(parents=True)
            (npu / "framework").mkdir()
            (npu / "core/npu/NPUStream.h").write_text(
                "#pragma once\nextern void* testStream;\n"
                "namespace c10_npu { struct Stream { void* stream(bool){return testStream;} };"
                "inline Stream getCurrentNPUStream(){return {};} }\n"
            )
            (npu / "core/npu/NPUGuard.h").write_text(
                "#pragma once\nnamespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device){} }; }\n"
            )
            (npu / "framework/OpCommand.h").write_text(
                "#pragma once\n#include <functional>\n#include <vector>\n"
                "extern std::vector<std::function<int()>> commands;\n"
                "namespace at_npu::native { struct OpCommand { template<class F> "
                "static void RunOpApi(const char*,F fn){commands.push_back(fn);} }; }\n"
            )
            (build / "binding.cpp").write_text(
                f'#include "{SOURCE / "li_page_gather_queued.cpp"}"\n'
                + r"""
#include <cassert>
#include <cstdlib>
#include <iostream>
void* testStream=reinterpret_cast<void*>(0x123);
std::vector<std::function<int()>> commands;
int calls=0, frees=0, launchStatus=0;
void* expected[3]{};
uint32_t expectedData=16384, expectedScaleBytes=256, expectedStride=16384;
at::Tensor Make(at::ScalarType dtype, std::vector<int64_t> shape,
    std::vector<int64_t> stride={}, c10::Device device={c10::DeviceType::PrivateUse1,0}) {
    size_t size=8; for(auto value:shape) size*=std::max<int64_t>(1,value);
    if(!stride.empty()) size+=8*std::max<int64_t>(1,shape[0])*stride[0];
    auto builder=at::for_blob(std::malloc(size),shape);
    builder.options(at::TensorOptions().dtype(dtype).device(device)).target_device(device)
        .deleter([](void* ptr){std::free(ptr);++frees;});
    if(!stride.empty()) builder.strides(stride);
    return builder.make_tensor();
}
extern "C" int32_t launch_dsa_selected_li_pages(void* stream,void* bases,void* k,void* scale,
    uint32_t domain,uint32_t pages,uint32_t mainOffset,uint32_t dataOffset,uint32_t scaleOffset,
    uint32_t dataBytes,uint32_t scaleBytes,uint32_t destinationStrideBytes) {
    assert(stream==reinterpret_cast<void*>(0x123));
    assert(bases==expected[0] && k==expected[1] && scale==expected[2]);
    assert(domain==7 && pages==9 && mainOffset==8192 && dataOffset==256 && scaleOffset==17408);
    assert(dataBytes==expectedData && scaleBytes==expectedScaleBytes &&
           destinationStrideBytes==expectedStride);
    ++calls;return launchStatus;
}
template<class F> void Reject(F fn) {
    const auto count=commands.size(); bool rejected=false;
    try{fn();}catch(const c10::Error&){rejected=true;}
    assert(rejected && commands.size()==count);
}
int main(){
    {
        auto bases=Make(at::kUInt64,{7}),k=Make(at::kChar,{9,128,1,128}),s=Make(at::kHalf,{9,128,1,1});
        expected[0]=bases.data_ptr();expected[1]=k.data_ptr();expected[2]=s.data_ptr();
        LIPageGather(bases,k,s,8192,256,17408);
        assert(commands.size()==1 && calls==0);
        // The command owns all tensor references even after the caller drops them.
    }
    assert(frees==0);testStream=reinterpret_cast<void*>(0x456);
    assert(commands.front()()==0 && calls==1);commands.clear();assert(frees==3);
    {
        // Target packed mixed indexer: 32 KiB K and 512 B scale per logical page.
        testStream=reinterpret_cast<void*>(0x123);
        auto bases=Make(at::kUInt64,{7}),k=Make(at::kChar,{9,256,128}),s=Make(at::kHalf,{9,256});
        expected[0]=bases.data_ptr();expected[1]=k.data_ptr();expected[2]=s.data_ptr();
        expectedData=32768;expectedScaleBytes=512;expectedStride=32768;
        LIPageGather(bases,k,s,8192,256,17408);
        assert(commands.size()==1 && calls==1);
        assert(commands.front()()==0 && calls==2);commands.clear();
        expectedData=16384;expectedScaleBytes=256;expectedStride=16384;
    }
    auto bases=Make(at::kUInt64,{7}),k=Make(at::kChar,{9,128,128}),s=Make(at::kHalf,{9,128});
    auto call=[&](const at::Tensor& b,const at::Tensor& key,const at::Tensor& scale,
                  int64_t m=8192,int64_t d=256,int64_t z=17408){LIPageGather(b,key,scale,m,d,z);};
    Reject([&]{call(Make(at::kLong,{7}),k,s);});
    Reject([&]{call(Make(at::kUInt64,{0}),k,s);});
    Reject([&]{call(Make(at::kUInt64,{1,7}),k,s);});
    Reject([&]{call(Make(at::kUInt64,{7},{2}),k,s);});
    Reject([&]{call(Make(at::kUInt64,{7},{},{c10::DeviceType::CPU}),k,s);});
    Reject([&]{call(bases,Make(at::kChar,{9,128,128},{},{c10::DeviceType::PrivateUse1,1}),s);});
    Reject([&]{call(bases,Make(at::kHalf,{9,128,128}),s);});
    Reject([&]{call(bases,Make(at::kChar,{147456}),s);});
    Reject([&]{call(bases,Make(at::kChar,{9,128,127}),s);});
    Reject([&]{call(bases,Make(at::kChar,{9,256,127}),s);});
    Reject([&]{call(bases,Make(at::kChar,{9,256,128}),s);});  // 32 KiB K needs 512 B scale.
    Reject([&]{call(bases,Make(at::kChar,{9,128,128},{32768,128,1}),s);});
    Reject([&]{call(bases,k,Make(at::kChar,{9,256}));});
    Reject([&]{call(bases,k,Make(at::kHalf,{9,127}));});
    Reject([&]{call(bases,k,Make(at::kHalf,{9,257}));});
    Reject([&]{call(bases,k,Make(at::kHalf,{8,128}));});
    Reject([&]{call(bases,k,Make(at::kHalf,{9,128},{256,1}));});
    Reject([&]{call(bases,k,Make(at::kHalf,{9,256}));});  // 512 B scale needs 32 KiB K.
    Reject([&]{call(Make(at::kUInt64,{10}),k,s);});
    Reject([&]{call(bases,k,s,-1);});
    Reject([&]{call(bases,k,s,INT64_MAX);});
    Reject([&]{call(bases,k,s,8192,-1);});
    Reject([&]{call(bases,k,s,8192,UINT32_MAX-16384+1ULL);});
    Reject([&]{call(bases,k,s,8192,256,-1);});
    Reject([&]{call(bases,k,s,8192,256,UINT32_MAX-256+1ULL);});
    testStream=reinterpret_cast<void*>(0x123);launchStatus=37;
    expected[0]=bases.data_ptr();expected[1]=k.data_ptr();expected[2]=s.data_ptr();
    call(bases,k,s);
    assert(commands.size()==1 && commands.front()()==37);commands.clear();
    const auto op=c10::Dispatcher::singleton().findSchemaOrThrow("dsa_prefill_queue::li_page_gather","");
    assert(op.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1));
    std::cout<<"LI host contracts passed\n";
}
"""
            )
            lib = Path(torch.__file__).parent / "lib"
            executable = build / "binding"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    f"-I{build}",
                    *(f"-I{path}" for path in include_paths()),
                    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
                    str(build / "binding.cpp"),
                    f"-L{lib}",
                    f"-Wl,-rpath,{lib}",
                    "-ltorch_cpu",
                    "-ltorch",
                    "-lc10",
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "LI host contracts passed\n")


if __name__ == "__main__":
    unittest.main()
