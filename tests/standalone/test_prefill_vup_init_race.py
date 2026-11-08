# SPDX-License-Identifier: Apache-2.0
"""Run the complete probe with CPU tensors and an explicit queued DMA model."""

import ctypes
import importlib.util
import json
import re
import struct
import weakref
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vup_init_race", ROOT / "tools/prefill_vup_init_race.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class CpuRuntime:
    """Storage survives tensor destruction, as in the NPU caching allocator.

    DMA is deferred until synchronization. The simulated operator's unstreamed
    tiling write reuses dead tensor storage. No freed CPU memory is accessed.
    """

    def __init__(
        self,
        *,
        overwrite=True,
        early_read=False,
        bad_read=False,
        bad_projection=False,
        raise_projection=False,
        fence_first_allocation=False,
    ):
        self.torch, self.device, self.npu = torch, "cpu", self
        self.overwrite, self.early_read = overwrite, early_read
        self.bad_read, self.bad_projection = bad_read, bad_projection
        self.raise_projection = raise_projection
        self.fence_first_allocation = fence_first_allocation
        self.copies = []
        self.initialized = False
        self.syncs = 0
        self.source_alive_at_project = []
        self.event_syncs = 0
        self.initialization_fences = 0

    def copy_out(self, destination, source):
        assert source.dtype == destination.dtype == torch.uint8
        self.source_ref = weakref.ref(source)
        self.source_storage = source.untyped_storage()
        self.copies.append((destination.untyped_storage(), self.source_storage, source.numel()))
        if self.early_read:
            self.synchronize()

    def synchronize(self):
        for destination, source, count in self.copies:
            ctypes.memmove(destination.data_ptr(), source.data_ptr(), count)
            if self.bad_read:
                ctypes.memset(destination.data_ptr(), 0, 1)
        self.copies.clear()
        self.syncs += 1

    def empty_cache(self):
        assert not self.copies

    def current_stream(self):
        return self

    def Event(self):
        rt = self

        class Event:
            def record(self, stream):
                assert stream is rt

            def query(self):
                return not rt.copies

            def synchronize(self):
                rt.event_syncs += 1
                rt.synchronize()

        return Event()

    def project(self, a, b, out):
        tokens, heads, width = a.shape
        assert b.shape == (heads, width, PROBE.V_WIDTH)
        assert out.shape == (tokens, heads, PROBE.V_WIDTH)
        alive = self.source_ref() is not None
        self.source_alive_at_project.append(alive)
        if not self.initialized and self.fence_first_allocation:
            self.synchronize()
            self.initialization_fences += 1
        if not self.initialized and not alive and self.overwrite:
            payload = struct.pack("<18I", heads, tokens, width, PROBE.V_WIDTH, *([0] * 14))
            offset = (tokens - 1) * PROBE.TILING_BYTES
            ctypes.memmove(self.source_storage.data_ptr() + offset, payload, len(payload))
        self.initialized = True
        if self.raise_projection:
            raise RuntimeError("operator failure")
        out.fill_(1 if self.bad_projection else 0)


@pytest.fixture(autouse=True)
def asynchronous_environment(monkeypatch):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)


def test_layout_matches_current_cpp_allocation_and_slot():
    root = ROOT / "csrc/batch_matmul_transpose/op_host"
    declarations = (root / "tiling/tiling_data.h").read_text()
    shape = declarations.split("struct OpShape {", 1)[1].split("\n};", 1)[0]
    tiling = declarations.split("struct PpMatmulTilingData {", 1)[1].split("\n};", 1)[0]
    fields = re.findall(r"uint32_t (\w+)\{", shape + tiling)
    assert fields[:4] == ["batchSize", "m", "k", "n"]
    assert "OpShape opShape{};" in tiling
    assert len(fields) * 4 == PROBE.TILING_BYTES
    allocation = (root / "batch_matmul_transpose.h").read_text()
    assert int(re.search(r"MAX_CAPTURE_NUM = (\d+)", allocation).group(1)) == PROBE.TILING_SLOTS
    assert "int32_t batchIdx = opShape.m - 1;" in allocation
    assert "{tilingSize * MAX_CAPTURE_NUM}" in allocation


def test_cpp_first_allocation_fence_is_checked_and_outside_the_hot_path():
    source = (ROOT / "csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h").read_text()
    # Guard the actual C++ lifetime boundary: task-queue flush and device fence
    # must be inside the one-time initializer and precede the direct H2D write.
    initializer, rest = source.split("static auto global_tiling_data = [&] {", 1)[1].split("}();", 1)
    assert initializer.index("at::empty(") < initializer.index("getCurrentNPUStream().stream()")
    assert initializer.index("getCurrentNPUStream().stream()") < initializer.index("aclrtSynchronizeStream(stream)")
    assert initializer.index("aclrtSynchronizeStream(stream)") < initializer.index(
        "TORCH_CHECK(status == ACL_ERROR_NONE"
    )
    assert initializer.index("TORCH_CHECK(status == ACL_ERROR_NONE") < initializer.index("return data;")
    assert "aclrtMemcpy(" in rest
    assert "aclrtSynchronizeStream(" not in rest


@pytest.mark.parametrize("mode", PROBE.MODES)
def test_first_allocation_fence_preserves_queued_reads_and_only_waits_once(mode):
    rt = CpuRuntime(fence_first_allocation=True)
    args = PROBE.parse_args(
        ["--operator", "legacy", "--mode", mode, "--tokens", "384", "--heads", "64", "--backlog-size", "16"]
    )
    report = PROBE.run_probe(rt, args)
    assert report["exit_code"] == 0
    assert report["changed_bytes"] == 0
    assert report["output_zero"] is True
    assert report["reader_pending_before_vup"] is True
    assert report["reader_pending_after_vup"] == (mode == "warm")
    assert rt.initialization_fences == 1
    # Another call in the same process uses the existing static allocation.
    again = PROBE.run_probe(rt, args)
    assert again["exit_code"] == 0
    assert again["changed_bytes"] == 0
    assert rt.initialization_fences == 1


@pytest.mark.parametrize("mode", PROBE.MODES)
@pytest.mark.parametrize("tokens,heads", [(1, 1), (384, 64), (1024, 2)])
def test_full_cli_with_real_cpu_buffers_and_deferred_dma(monkeypatch, tmp_path, capsys, mode, tokens, heads):
    rt = CpuRuntime()
    monkeypatch.setattr(PROBE, "load_runtime", lambda args: rt)
    output = tmp_path / "nested" / f"{mode}.json"
    code = PROBE.main(
        [
            "--operator",
            "legacy",
            "--mode",
            mode,
            "--tokens",
            str(tokens),
            "--heads",
            str(heads),
            "--backlog-size",
            "16",
            "--backlog-ops",
            "2",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["reader_pending_before_vup"] is True
    assert report["output_zero"] is True
    assert report["slot_offset"] == (tokens - 1) * PROBE.TILING_BYTES
    assert report["changed_bytes_outside_slot"] == 0
    assert rt.event_syncs == int(mode == "sync")
    assert not rt.copies
    assert rt.source_alive_at_project[-1] == (mode == "hold")
    if mode == "cold":
        assert code == 1
        assert report["status"] == "tiling_overwrite_reproduced"
        assert report["tiling_signature"] is True
        assert report["slot_shape_words"] == [heads, tokens, 512, 128]
        assert report["changed_bytes"] == PROBE.TILING_BYTES
    else:
        assert code == 0
        assert report["status"] == "control_clean"
        assert report["changed_bytes"] == 0
    records = [json.loads(line.removeprefix("[VUP_INIT] ")) for line in capsys.readouterr().out.splitlines()]
    assert records[-1] == {"event": "result", **report}


@pytest.mark.parametrize(
    "options,status,code",
    [
        ({"overwrite": False}, "overwrite_not_observed", 0),
        ({"early_read": True}, "reader_finished_before_vup", 0),
        ({"bad_read": True}, "unexpected_readback", 2),
        ({"bad_projection": True}, "projection_failed", 2),
    ],
)
def test_distinguishes_no_overlap_other_corruption_and_projection_failure(options, status, code):
    args = PROBE.parse_args(
        ["--operator", "legacy", "--mode", "cold", "--tokens", "2", "--heads", "1", "--backlog-size", "16"]
    )
    report = PROBE.run_probe(CpuRuntime(**options), args)
    assert report["status"] == status
    assert report["exit_code"] == code


@pytest.mark.parametrize("mode", ["sync", "hold", "warm"])
def test_control_corruption_is_a_failure(mode):
    args = PROBE.parse_args(
        ["--operator", "legacy", "--mode", mode, "--tokens", "2", "--heads", "1", "--backlog-size", "16"]
    )
    report = PROBE.run_probe(CpuRuntime(bad_read=True), args)
    assert report["status"] == "control_failed"
    assert report["exit_code"] == 2


def test_operator_exception_still_drains_dma_with_destination_alive():
    args = PROBE.parse_args(
        ["--operator", "legacy", "--mode", "cold", "--tokens", "2", "--heads", "1", "--backlog-size", "16"]
    )
    rt = CpuRuntime(raise_projection=True)
    with pytest.raises(RuntimeError, match="operator failure"):
        PROBE.run_probe(rt, args)
    assert not rt.copies
    assert rt.syncs == 3


@pytest.mark.parametrize(
    "option,value",
    [
        ("tokens", "0"),
        ("tokens", "1025"),
        ("heads", "0"),
        ("device", "-1"),
        ("backlog-size", "17"),
        ("backlog-size", "16384"),
        ("backlog-ops", "0"),
        ("mode", "before_vup"),
    ],
)
def test_invalid_cli_fails_before_runtime_loading(monkeypatch, option, value):
    def unexpected_load(args):
        pytest.fail("invalid arguments reached runtime loading")

    monkeypatch.setattr(PROBE, "load_runtime", unexpected_load)
    with pytest.raises(SystemExit) as error:
        PROBE.main(["--operator", "legacy", "--mode", "cold", f"--{option}", value])
    assert error.value.code == 2


def test_rejects_blocking_launch(monkeypatch):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    with pytest.raises(SystemExit) as error:
        PROBE.parse_args(["--operator", "legacy", "--mode", "cold"])
    assert error.value.code == 2


def test_rejects_truncated_readback():
    with pytest.raises(ValueError, match="readback bytes"):
        PROBE.inspect_readback(b"", 384, 64)
