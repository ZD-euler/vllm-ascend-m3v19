# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/summarize_prefill_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_prefill_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1024])
@pytest.mark.parametrize("wrapped", [False, True])
def test_streaming_json_at_arbitrary_boundaries(tool, tmp_path, chunk_size, wrapped):
    rows = [{"name": 'value with } ] \\" escaped and 汉字', "args": {"nested": [1, 2]}}, {"dur": 60000}]
    path = tmp_path / "trace_view.json"
    value = {"traceEvents": rows, "metadata": "trailing"} if wrapped else rows
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
    assert list(tool.events(path, chunk_size)) == rows


def test_truncated_event_reports_error(tool, tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text('{"traceEvents":[{"name":"partial', encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        list(tool.events(path, 4))


def test_preserves_device_tasks_and_cpu_ranges_separately(tool, tmp_path):
    rows = [
        {"name": "hcom_allReduce", "ph": "X", "pid": 1, "tid": 3, "ts": 10, "dur": 180000},
        {"name": "hcom_allReduce", "ph": "X", "pid": 2, "tid": 4, "ts": 20, "dur": 90000},
        {"name": "Memcpy H2D", "ph": "X", "pid": 2, "tid": 5, "ts": 5, "dur": 60000},
        {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "CPU"}},
        {"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 5, "args": {"name": "Stream 5"}},
    ]
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps({"traceEvents": rows}))
    result = tool.trace_summary(path, 1)
    assert len(result["allreduce"]) == 2
    assert result["allreduce"][0]["duration_ms"] == 180
    assert result["copy"][0]["duration_ms"] == 60
    assert result["copy"][0]["thread"] == "Stream 5"


def test_hccl_summary_retains_wait_and_transit(tool, tmp_path):
    path = tmp_path / "communication.json"
    data = {
        "step0": {
            "hcom_allReduce__1@group": {
                "Communication Time Info": {"Elapse Time(ms)": 180, "Wait Time(ms)": 170, "Transit Time(ms)": 10},
                "Communication Bandwidth Info": {
                    "SDMA": {"Transit Size(MB)": 2, "Bandwidth(GB/s)": 0.2, "Size Distribution": {"2": [1, 10]}}
                },
            }
        }
    }
    path.write_text(json.dumps(data))
    result = tool.communication_summary(path, 1)["allreduce"][0]
    assert result["time"]["Wait Time(ms)"] == 170
    assert result["time"]["Transit Time(ms)"] == 10
    assert "Size Distribution" not in result["links"]["SDMA"]
