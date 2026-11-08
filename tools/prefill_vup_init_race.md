# V-up 社区算子移植与异步内存短测

默认测试实际 serving 的 `AscendSFAImpl._v_up_proj`。只使用一张空闲 NPU，
无需模型或 P/D 服务。运行环境沿用服务安装的 torch_npu、vllm-ascend 和
LMCache-Ascend。

## 本次移植

来源是社区 [PR #17141](https://github.com/vllm-project/vllm-ascend/pull/17141)
中 SFA 的 V-up 替换，保留旧 CANN 的维度兼容分支：

- FP16/BF16 使用 `torch_npu.npu_transpose_batchmatmul`。
- `heads * kv_lora_rank < 65536` 时由算子内部转置输入；其余情况显式转置输入。
- 权重保持 `(heads, kv_lora_rank, v_head_dim)`，沿用原有加载方式。
- torch_npu 缺少接口或输入为其他 dtype 时沿用 `torch.bmm`。
- bank、event、KV 传输和 Q/K 投影沿用现有实现。

SFA serving 已绕开旧自定义算子及其首次初始化同步。旧 C++ 实现保留用于
历史问题对照，本次改动仅涉及 Python，拉取后重启服务即可生效。
本地 MLA 已经使用 torch_npu 的同名接口。

## 运行

在仓库内使用服务的 Python 环境。`0` 是空闲卡编号：

```bash
export PATH="/usr/local/python3.11.14/bin:$PATH"
git pull --ff-only
bash tools/run_prefill_vup_init_race.sh 0
```

四个模式各启动独立进程，每个进程限时 90 秒。日志与 JSON 保存在输出的
`result_dir`。本次测试默认目标为 `serving`，结果应包含：

- `projection_branch: "npu_transpose_batchmatmul"`：实际安装的 torch_npu 提供新接口。
- `exit_code: 0`、`changed_bytes: 0`：读回数据完整。
- `output_zero: true`、`numerical.passed: true`：零权重和非零权重数值检查通过。
- `numerical.cases` 包含 1、384、1536 token，覆盖原来 1024 token 分界两侧。

若 `projection_branch` 为 `native_torch_bmm`，说明当前 torch_npu 缺少新接口，
此次验证的是兼容回退路径。保留 runtime 日志中的版本号以检查部署依赖。

`reader_pending_before_vup` 标识测试建立了异步读取积压。
`reader_pending_after_vup` 标识 V-up 返回时该读取是否仍在排队。
`project_call_ms` 是 CPU 调用 V-up 的墙钟耗时，包含主机开销及可能的等待；
它用于观察这次短测中的提交行为，与服务 TTFT 分别记录。

数值检查安排在冷启动检查之后，防止提前初始化算子。非零权重按 head 选择
不同输入列并改变符号，期望值直接由输入计算，覆盖输出布局与 head 顺序。

## 四个模式

| 模式 | 操作 | 用途 |
| --- | --- | --- |
| `cold` | 读取排队后释放 Tensor，首次调用 V-up | 检查首次调用时的缓存复用 |
| `sync` | 先等读取完成，再释放并首次调用 V-up | 检查完成顺序的影响 |
| `hold` | 保留 Tensor 至全部计算完成 | 检查内存复用的影响 |
| `warm` | 提前完成一次 V-up，再进行相同测试 | 区分首次调用和后续调用 |

## 旧算子对照

显式指定 `legacy` 才会运行旧自定义算子：

```bash
bash tools/run_prefill_vup_init_race.sh 0 vup-legacy-check legacy
```

旧算子首次分配 `72 * 1024 = 73728` 字节的静态 tiling 缓冲区，每次调用
通过直接 `aclrtMemcpy` 写入 `(tokens - 1) * 72` 偏移处的 72 字节。
2026-11-08 的修复前 NPU 短测在 `cold` 模式复现了精确覆盖：偏移 27576，
shape 字段 `[64, 384, 512, 128]`；`sync`、`hold`、`warm` 均为零改写。

旧 C++ 修复在首次分配之后等待当前流完成再写入，因此其 `cold` 模式可显示
`reader_pending_after_vup: false`。验证该旧实现的 C++ 改动仍需重新编译扩展。

立即复制语义见 [CANN runtime 文档](https://github.com/hicann/runtime/blob/master/docs/zh/api_ref/11-03_memory_copy_and_set.md#aclrtMemcpy)，
同流缓存复用见 [torch_npu 分配器](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L63-L87)。

## CPU 验证

`tests/standalone/test_native_vup_backport.py` 执行 serving 的实际 Python 方法，
检查维度边界、权重布局、dtype/API 回退，以及完整短测 CLI 的延迟 DMA 和
非零数值校验。NPU 算子用 CPU BMM 替身执行；硬件部分由上面的短测覆盖。
