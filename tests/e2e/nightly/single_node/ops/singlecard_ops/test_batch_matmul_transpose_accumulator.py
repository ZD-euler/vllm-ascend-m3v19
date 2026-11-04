# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check V-up accumulator initialization across consecutive kernel launches."""

import importlib
import importlib.util
import unittest

import torch

LOCAL_HEADS = 16
LATENT_DIM = 512
VALUE_DIM = 128
TOKEN_COUNTS = (1, 367, 1024)
# Nonzero outputs followed by zero outputs expose an uncleared accumulator.
# Repeating the sequence also exercises different data with identical tiling.
VARIANTS = (0, 1, 2, 1, 0, 2) * 2


def _make_exact_case(tokens, dtype):
    """Return CPU inputs, a two-hot weight, and an exact gather-based oracle.

    Each output adds two small integers with coefficients +1 and -1. Both
    FP16 and BF16 can represent every input, intermediate sum, and result
    exactly, so a changed bit is a numerical error for this fixture.
    """
    token = torch.arange(tokens, dtype=torch.int32).view(-1, 1, 1)
    head = torch.arange(LOCAL_HEADS, dtype=torch.int32).view(1, -1, 1)
    latent = torch.arange(LATENT_DIM, dtype=torch.int32).view(1, 1, -1)
    pattern = token * (latent % 7 + 1) + head * 5 + latent * 3
    first = (pattern % 29 - 14).to(dtype)
    second = -((pattern + 11) % 29 - 14).to(dtype)
    inputs = torch.stack((first, second, torch.zeros_like(first)))

    heads = torch.arange(LOCAL_HEADS).view(-1, 1)
    columns = torch.arange(VALUE_DIM).view(1, -1)
    # Exercise both halves of K, including multiple internal matmul tiles.
    first_k = (columns * 3 + heads * 5) % (LATENT_DIM // 2)
    second_k = LATENT_DIM // 2 + (columns * 7 + heads * 11) % (LATENT_DIM // 2)
    weights = torch.zeros(LOCAL_HEADS, LATENT_DIM, VALUE_DIM, dtype=dtype)
    weights[heads, first_k, columns] = 1
    weights[heads, second_k, columns] = -1

    gather_shape = (inputs.shape[0], tokens, LOCAL_HEADS, VALUE_DIM)
    first_indices = first_k.view(1, 1, LOCAL_HEADS, VALUE_DIM).expand(gather_shape)
    second_indices = second_k.view(1, 1, LOCAL_HEADS, VALUE_DIM).expand(gather_shape)
    expected = inputs.gather(-1, first_indices) - inputs.gather(-1, second_indices)
    return inputs, weights, expected


class TestBatchMatmulTransposeAccumulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the CPU oracle importable without loading vLLM or an NPU
        # extension. An installed but broken extension must still fail.
        if importlib.util.find_spec("torch_npu") is None:
            raise unittest.SkipTest("requires torch_npu and an available NPU")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            raise unittest.SkipTest("requires an available NPU")

        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
        if not hasattr(torch.ops._C_ascend, "batch_matmul_transpose"):
            raise RuntimeError("batch_matmul_transpose custom operator is not installed")

    def test_consecutive_vup_calls(self):
        # The largest case uses about 98 MiB of explicit device tensors. No
        # model, historical KV, graph capture, or second stream is required.
        for dtype in (torch.float16, torch.bfloat16):
            for tokens in TOKEN_COUNTS:
                with self.subTest(dtype=dtype, tokens=tokens):
                    inputs_cpu, weights_cpu, expected = _make_exact_case(tokens, dtype)
                    inputs = inputs_cpu.to("npu")
                    weights = weights_cpu.to("npu")
                    outputs = torch.full(
                        (len(VARIANTS), tokens, LOCAL_HEADS, VALUE_DIM),
                        float("nan"),
                        dtype=dtype,
                        device="npu",
                    )
                    for call, variant in enumerate(VARIANTS):
                        torch.ops._C_ascend.batch_matmul_transpose(inputs[variant], weights, outputs[call])

                    # No per-launch CPU reads/waits: all kernels above are
                    # submitted before checking any result. NaN poison also
                    # catches output elements that the kernel never writes.
                    torch.npu.synchronize()
                    observed = outputs.cpu()
                    for call, variant in enumerate(VARIANTS):
                        with self.subTest(call=call, variant=variant):
                            torch.testing.assert_close(observed[call], expected[variant], rtol=0, atol=0)
                    del inputs, weights, outputs
                    del inputs_cpu, weights_cpu, expected, observed


if __name__ == "__main__":
    unittest.main()
