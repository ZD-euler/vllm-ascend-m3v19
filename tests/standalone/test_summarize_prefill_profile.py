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


def make_capture(root, stamp, run_id, prefix=512000, new=128):
    profile_root = root / "vllm_profile" / "P_dp0"
    workers = []
    for rank in range(2):
        worker = profile_root / f"rank{rank}_{stamp}_{run_id}_ascend_pt"
        output = worker / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        (output / "trace_view.json").write_text(json.dumps({"traceEvents": []}))
        workers.append(worker)
    result_dir = root / f"prefill-profile-{stamp}-{run_id}"
    result_dir.mkdir()
    data = {
        "run_id": run_id,
        "prefix_tokens": prefix,
        "new_prompt_tokens": new,
        "status": "exported",
        "raw_profiles": list(map(str, workers)),
    }
    (result_dir / "result.json").write_text(json.dumps(data))
    return profile_root, result_dir, workers


def test_lists_chronological_runs_without_parsing_traces(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-150000", "new", prefix=20480, new=4096)
    make_capture(tmp_path, "20261109-140000", "old")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "trace_summary", lambda *args: pytest.fail("Listing parsed a trace"))
    tool.main([str(profile_root), "--list"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("1\t2026-11-09 14:00:00\t512000\t128\t2/2")
    assert lines[2].startswith("2\t2026-11-09 15:00:00\t20480\t4096\t2/2")


@pytest.mark.parametrize("selector", [["2"], ["--run", "new"], ["--run", "ne"]])
def test_selects_exact_workers_for_one_run(tool, tmp_path, monkeypatch, capsys, selector):
    profile_root, _, old_workers = make_capture(tmp_path, "20261109-140000", "old")
    _, result_dir, new_workers = make_capture(tmp_path, "20261109-150000", "new", new=4096)
    monkeypatch.chdir(tmp_path)
    tool.main([str(profile_root), *selector])
    result = json.loads(capsys.readouterr().out)
    assert result["selection"]["number"] == 2
    assert result["selection"]["manifest"] == str(result_dir / "result.json")
    assert result["selection"]["new_prompt_tokens"] == 4096
    assert len(result["traces"]) == 2
    assert all(str(path) in result["selection"]["profiles"] for path in new_workers)
    assert all(str(path) not in json.dumps(result) for path in old_workers)


def test_selection_by_full_result_directory_name(tool, tmp_path):
    profile_root, result_dir, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    runs = tool.discover_runs(profile_root)
    index, run = tool.select_run(runs, result_dir.name)
    assert index == 1 and run["run_id"] == "abcd"


def test_missing_worker_profile_fails_selection(tool, tmp_path, capsys):
    profile_root, _, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    trace = workers[1] / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    trace.unlink()
    trace.parent.rmdir()
    workers[1].rmdir()
    with pytest.raises(SystemExit) as error:
        tool.main([str(profile_root), "1"])
    assert error.value.code == 2
    assert "1/2 worker profiles" in capsys.readouterr().err


def test_relocated_raw_paths_match_by_unique_basename(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    path = result_dir / "result.json"
    data = json.loads(path.read_text())
    data["raw_profiles"] = [str(tmp_path / "previous-host" / worker.name) for worker in workers]
    path.write_text(json.dumps(data))
    runs = tool.discover_runs(profile_root)
    assert len(runs) == 1
    assert runs[0]["paths"] == workers


def test_falls_back_to_raw_profile_manifest(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    path = result_dir / "result.json"
    data = json.loads(path.read_text())
    references = data.pop("raw_profiles")
    path.write_text(json.dumps(data))
    (result_dir / "raw_profiles.json").write_text(json.dumps(references))
    assert tool.discover_runs(profile_root)[0]["paths"] == workers


def test_custom_manifest_directory(tool, tmp_path):
    profile_root, result_dir, workers = make_capture(tmp_path, "20261109-140000", "abcd")
    custom_root = tmp_path / "other-client" / "results"
    custom_root.mkdir(parents=True)
    result_dir.rename(custom_root / "custom-output")
    runs = tool.discover_runs(profile_root, custom_root)
    assert len(runs) == 1 and runs[0]["paths"] == workers


def test_unrecorded_capture_is_listed_individually(tool, tmp_path):
    raw = tmp_path / "legacy_ascend_pt"
    raw.mkdir()
    runs = tool.discover_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["paths"] == [raw]
    assert runs[0]["prefix_tokens"] is None
    assert runs[0]["status"] == "raw"


@pytest.mark.parametrize("selector", ["0", "2", "missing"])
def test_invalid_selection_is_reported(tool, tmp_path, selector):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    with pytest.raises(ValueError):
        tool.select_run(tool.discover_runs(profile_root), selector)


def test_ambiguous_id_is_reported(tool, tmp_path):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd1")
    make_capture(tmp_path, "20261109-150000", "abcd2")
    with pytest.raises(ValueError, match="matched 2"):
        tool.select_run(tool.discover_runs(profile_root), "abcd")


def test_legacy_multiple_files_still_work(tool, tmp_path, capsys):
    paths = []
    for rank in range(2):
        directory = tmp_path / f"rank{rank}"
        directory.mkdir()
        path = directory / "trace_view.json"
        path.write_text(json.dumps({"traceEvents": []}))
        paths.append(path)
    tool.main(list(map(str, paths)))
    result = json.loads(capsys.readouterr().out)
    assert len(result["traces"]) == 2
    assert "selection" not in result


def test_incomplete_manifest_leaves_other_captures_available(tool, tmp_path, capsys):
    profile_root, result_dir, _ = make_capture(tmp_path, "20261109-140000", "broken")
    _, _, workers = make_capture(tmp_path, "20261109-150000", "ready")
    (result_dir / "result.json").write_text('{"run_id":')
    runs = tool.discover_runs(profile_root)
    ready = next(run for run in runs if run["run_id"] == "ready")
    assert ready["paths"] == workers
    assert "Skipping unreadable capture record" in capsys.readouterr().err
