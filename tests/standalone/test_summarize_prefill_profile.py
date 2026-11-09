# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import shutil
import subprocess
import sys
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


def test_no_arguments_prompts_for_number_and_saves_summary(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "old")
    make_capture(tmp_path, "20261109-150000", "new", new=4096)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.chdir(tmp_path)
    prompts = []

    def choose(prompt):
        prompts.append(prompt)
        return "2"

    monkeypatch.setattr("builtins.input", choose)
    tool.main([])
    assert prompts == ["请输入目录编号（PID_时间戳）："]
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "new"
    assert report["selection"]["new_prompt_tokens"] == 4096
    output = capsys.readouterr().out
    assert "Profile 根目录" in output and "摘要已保存到" in output


def test_number_only_uses_default_directory_without_prompt(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected prompt"))
    monkeypatch.chdir(tmp_path)
    tool.main(["1"])
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "abcd"


def test_explicit_directory_interactive_and_output(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    output = tmp_path / "selected.json"
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    tool.main([str(profile_root), "--interactive", "--output", str(output)])
    assert json.loads(output.read_text())["selection"]["number"] == 1


def test_default_directory_list_does_not_prompt_or_write_summary(tool, tmp_path, monkeypatch):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Unexpected prompt"))
    monkeypatch.chdir(tmp_path)
    tool.main(["--list"])
    assert not (tmp_path / "profile-summary.json").exists()


def test_number_is_supplied_once(tool, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        tool.main(["1", "--run", "2"])
    assert error.value.code == 2
    assert "Supply the capture identifier once" in capsys.readouterr().err


def test_interactive_eof_does_not_create_summary(tool, tmp_path, monkeypatch, capsys):
    profile_root, _, _ = make_capture(tmp_path, "20261109-140000", "abcd")
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", profile_root)
    monkeypatch.chdir(tmp_path)

    def end_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    with pytest.raises(SystemExit) as error:
        tool.main([])
    assert error.value.code == 2
    assert "No profile identifier was entered" in capsys.readouterr().err
    assert not (tmp_path / "profile-summary.json").exists()


def test_interactive_script_entry_point_on_cpu(tool, tmp_path):
    make_capture(tmp_path, "20261109-140000", "abcd")
    script = tmp_path / "tools" / "summarize_prefill_profile.py"
    script.parent.mkdir()
    shutil.copyfile(tool.__file__, script)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script)],
        input="1\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=tmp_path,
        check=True,
        timeout=15,
    )
    assert "请输入目录编号" in result.stdout
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["run_id"] == "abcd"
    assert len(report["traces"]) == 2


def raw_captures(root):
    paths = []
    for pid in (162830, 162950):
        path = root / f"dp0_pp0_tp1_dcp0_ep1_rank1_{pid}_20261109154305160_ascend_pt"
        path.mkdir(parents=True)
        (path / "trace_view.json").write_text(json.dumps({"traceEvents": []}))
        paths.append(path)
    return paths


@pytest.mark.parametrize("arguments", [[], ["--run"]])
def test_raw_identifier_filters_one_directory_without_client_records(tool, tmp_path, monkeypatch, capsys, arguments):
    target, other = raw_captures(tmp_path)
    identifier = "162830_20261109154305160"
    monkeypatch.setattr(tool, "discover_runs", lambda *args: pytest.fail("Raw selection read other run records"))
    tool.main([str(tmp_path), *arguments, identifier])
    result = json.loads(capsys.readouterr().out)
    assert result["selection"]["profile_identifier"] == identifier
    assert result["selection"]["profiles"] == [str(target)]
    assert len(result["traces"]) == 1
    assert str(other) not in json.dumps(result)


def test_identifier_only_uses_default_root(tool, tmp_path, monkeypatch):
    target, _ = raw_captures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", tmp_path)
    tool.main(["162830_20261109154305160"])
    result = json.loads((tmp_path / "profile-summary.json").read_text())
    assert result["selection"]["profiles"] == [str(target)]


def test_interactive_raw_identifier(tool, tmp_path, monkeypatch):
    target, _ = raw_captures(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "DEFAULT_PROFILE_DIRECTORY", tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "162830_20261109154305160")
    tool.main([])
    result = json.loads((tmp_path / "profile-summary.json").read_text())
    assert result["selection"]["profiles"] == [str(target)]


def test_duplicate_raw_identifier_requires_unique_directory(tool, tmp_path):
    raw_captures(tmp_path / "copy1")
    raw_captures(tmp_path / "copy2")
    with pytest.raises(ValueError, match="matched 2 directories"):
        tool.select_profile_directory(tmp_path, "162830_20261109154305160")


def test_missing_raw_identifier_reports_zero_matches(tool, tmp_path):
    raw_captures(tmp_path)
    with pytest.raises(ValueError, match="matched 0 directories"):
        tool.select_profile_directory(tmp_path, "999999_20261109154305160")


def test_existing_raw_directory_path_preserves_direct_input(tool, tmp_path, capsys):
    target, _ = raw_captures(tmp_path)
    tool.main([str(target)])
    result = json.loads(capsys.readouterr().out)
    assert len(result["traces"]) == 1
    assert "selection" not in result


def test_raw_identifier_match_is_exact(tool, tmp_path):
    raw_captures(tmp_path)
    with pytest.raises(ValueError, match="matched 0 directories"):
        tool.select_profile_directory(tmp_path, "62830_20261109154305160")


@pytest.mark.parametrize("interactive", [False, True])
def test_user_pid_timestamp_through_real_script_entry_point(tool, tmp_path, interactive):
    target, other = raw_captures(tmp_path / "vllm_profile" / "P_dp0")
    identifier = "162830_20261109154305160"
    script = tmp_path / "tools" / "summarize_prefill_profile.py"
    script.parent.mkdir()
    shutil.copyfile(tool.__file__, script)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script), *([] if interactive else [identifier])],
        input=identifier + "\n" if interactive else "",
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=tmp_path,
        check=True,
        timeout=15,
    )
    report = json.loads((tmp_path / "profile-summary.json").read_text())
    assert report["selection"]["profiles"] == [str(target)]
    assert str(other) not in json.dumps(report)
    assert "摘要已保存到" in result.stdout
