# V-up 首次分配与异步临时内存短测

这个测试直接检查 `batch_matmul_transpose` 首次调用时，tiling 的 H2D 写入是否覆盖
前序异步读取的临时内存。只使用一张空闲 NPU，无需模型、P/D 服务或重新编译。
运行当前已安装的 vllm-ascend、LMCache-Ascend 扩展。

## 已有对照

- 原事件位置：`p_sfa` 正常；`p_use, group=0` 正常；`p_vup, group=0` 有乱码。
- 两个 group 的事件后移到下一层、关闭全部 fence：仍有乱码。
- SFA/V-up 独立测试的小、大 token 分支均通过；该测试先运行串行校验，已预热 V-up。

下一步检查 V-up 首次调用的内存写入。事件后移控制 bank 复用，这里的对象是
缓存分配器管理的临时内存。当前服务结果支持优先检查这段异步访问时序。

## 检查的具体操作

`csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h` 首次分配
`72 * 1024 = 73728` 字节的静态 tiling 缓冲区。每次调用用直接 `aclrtMemcpy`
写入 `(tokens - 1) * 72` 偏移处的 72 字节。

脚本先建立同一计算流上的任务积压，再排入一次读取固定 `0xA5` 数据的 D2H。
随后释放数据的 Tensor 对象，调用真正的 V-up 算子。缓存分配器保留底层存储，
正常的同流队列写入会排在读取之后；这里检查直接 H2D 写入是否提前改变它。
整个试验结束后才检查读回数据。输入、输出、权重都预先分配。

接口的立即复制语义见 [CANN runtime 文档](https://github.com/hicann/runtime/blob/master/docs/zh/api_ref/11-03_memory_copy_and_set.md#aclrtMemcpy)，
同流缓存复用规则见 [torch_npu 分配器](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L63-L87)。

若读回结果在预期偏移处出现 `[heads, tokens, 512, 128]` 四个 uint32，且所有
改写局限于该 72 字节区间，输出 `tiling_overwrite_reproduced`。这给出具体被覆盖
的地址、偏移和写入者的参数签名。

## 运行

在 vllm-ascend 仓库内，用运行服务的 Python 环境执行。`0` 是一张空闲卡的编号。

```bash
git pull --ff-only
bash tools/run_prefill_vup_init_race.sh 0
```

指定 Python 路径时：

```bash
PYTHON=/usr/local/python3.11.14/bin/python3 bash tools/run_prefill_vup_init_race.sh 0
```

四个模式各启动独立进程，保证静态 tiling 分配状态独立。每个进程限时 90 秒，
超时后留 5 秒退出。日志与 JSON 保存在输出的 `result_dir`。

| 模式 | 操作 | 用途 |
| --- | --- | --- |
| `cold` | 读取排队后释放 Tensor，首次调用 V-up | 检查首次分配和直接写入 |
| `sync` | 先等读取完成，再释放并首次调用 V-up | 检查完成顺序的影响 |
| `hold` | 保留 Tensor 至全部计算完成 | 检查内存复用的影响 |
| `warm` | 提前完成一次 V-up，再进行相同测试 | 区分首次分配和后续调用 |

## 结果与后续行动

- `cold=tiling_overwrite_reproduced`，其他三个 `control_clean`：定位到首次
  tiling 分配的直接写入覆盖前序读取；下一步修复该写入的流顺序和内存生命周期，
  先用相同短测验证。
- `cold=overwrite_not_observed`：本次读取完整，保留 JSON 中的排队状态；继续
  排查服务中其他提交期间的写入，优先使用地址与数据比对。
- `reader_finished_before_vup`：本次读取已提前完成。单独增加 `--backlog-ops`
  即可延长积压，先查看现有日志，无需启动服务。
- `control_failed`、`unexpected_readback`、`projection_failed`：保留完整日志，
  分别检查读回损坏范围、DMA 基础路径或算子输出。

`cold` 成功抓到覆盖时进程退出码为 **1**；其他错误为 **2**。包装脚本会继续
执行对照模式。该实验定位首次调用时的具体内存写入风险；服务修复验证再沿用
原来的高并发负载。

## CPU 检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 -m pytest -q \
  --confcutdir=tests/standalone tests/standalone/test_prefill_vup_init_race.py
```

CPU 测试使用实际 Tensor 缓冲区和可控的延迟 DMA 模型，验证完整 CLI、参数边界、
四种模式、结构体布局、字节签名、异常收尾和结果文件。NPU 的实际地址复用与
执行时序由上面的短测采集。
