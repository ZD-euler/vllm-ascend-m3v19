# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import kv_replay_transport as T  # noqa: E402


def worker(tmp_path, pid, values):
    folder = tmp_path / "proc" / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cmdline").write_bytes(b"VLLM::Worker\0")
    (folder / "status").write_text(f"Name:\tVLLM::Worker\nPPid:\t{pid + 1000}\n")
    # Reproduce the real failure: the renamed worker has a partial environment,
    # while its service launcher still has every setting used at exec time.
    (folder / "environ").write_bytes(b"HCCL_IF_IP=7.150.3.60\0ASCEND_BUFFER_POOL=4:8\0")
    launcher = tmp_path / "proc" / str(pid + 1000)
    launcher.mkdir(parents=True, exist_ok=True)
    (launcher / "cmdline").write_bytes(b"/usr/bin/python3\0/usr/local/bin/vllm\0serve\0/model\0")
    (launcher / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in dict(PATH="/usr/bin", **values).items())
    )


def test_captures_latest_workers_and_only_transport_fields(tmp_path):
    log = tmp_path / "D.log"
    log.write_text(
        "(Worker_DP0_TP0_EP0 pid=10) Mooncake payload layout: old\n"
        "(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout: current\n"
        "(Worker_DP0_TP1_EP1 pid=21) Mooncake payload layout: current\n"
    )
    values = dict(
        HCCL_INTRA_ROCE_ENABLE="1",
        ASCEND_BUFFER_POOL="4:8",
        HCCL_IF_IP="7.150.3.60",
        HCCL_SOCKET_IFNAME="enp67s0f5",
        ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
        PRIVATE_PASSWORD="not-for-the-plan",
        LMCACHE_PREFILL_STORE_DEBUG="1",
    )
    for pid in (20, 21):
        worker(tmp_path, pid, values)
    result = T.read_transport_environment([log], tmp_path / "proc")
    assert result["source_pids"] == [1020, 1021]  # old PID need not exist
    assert result["worker_pids"] == [20, 21]
    assert result["source"] == "service_launcher"
    assert result["d_visible_devices"] == [0, 1, 2, 3]
    assert result["environment"] == {key: values.get(key) for key in T.TRANSPORT_ENV_KEYS}
    assert "PRIVATE_PASSWORD" not in json.dumps(result)
    assert "LMCACHE_PREFILL_STORE_DEBUG" not in json.dumps(result)


@pytest.mark.parametrize("problem", ["missing", "different", "no_pid"])
def test_missing_or_inconsistent_process_fails_before_injection(tmp_path, problem):
    log = tmp_path / "D.log"
    log.write_text(
        "(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout:\n(Worker_DP0_TP1_EP1 pid=21) Mooncake payload layout:\n"
        if problem != "no_pid"
        else "old log without workers\n"
    )
    worker(tmp_path, 20, dict(HCCL_INTRA_ROCE_ENABLE="1"))
    if problem == "different":
        worker(tmp_path, 21, dict(HCCL_INTRA_ROCE_ENABLE="0"))
    with pytest.raises(ValueError, match="Cannot read live|disagree|Cannot locate"):
        T.read_transport_environment([log], tmp_path / "proc")


def config():
    values = dict(HCCL_INTRA_ROCE_ENABLE="1", ASCEND_BUFFER_POOL="4:8", HCCL_IF_IP="7.150.3.60")
    return dict(
        environment={key: values.get(key) for key in T.TRANSPORT_ENV_KEYS},
        source_pids=[32124],
        d_visible_devices=[0, 1, 2, 3],
    )


def test_applies_and_unsets_in_fresh_native_import_process(tmp_path):
    # Real child/import ordering, without needing an NPU. The fake native module
    # observes the environment at import, exactly where the settings matter.
    (tmp_path / "torch.py").write_text("raise RuntimeError('stop at native import')\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(TOOLS))
    env.update(ASCEND_ENABLE_USE_FABRIC_MEM="1", ASCEND_RT_VISIBLE_DEVICES="4,5,6,7")
    for key in ("HCCL_INTRA_ROCE_ENABLE", "ASCEND_BUFFER_POOL", "HCCL_IF_IP"):
        env.pop(key, None)
    plan = dict(transport=config(), seed_device=0)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from prefill_kv_seed import seed; seed(json.loads(sys.argv[1]))",
            json.dumps(plan),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert "stop at native import" in child.stderr
    line = next(line for line in child.stdout.splitlines() if '"status": "transport_config"' in line)
    actual = json.loads(line.split("] ", 1)[1])
    assert actual["environment"] == dict(HCCL_INTRA_ROCE_ENABLE="1", ASCEND_BUFFER_POOL="4:8", HCCL_IF_IP="7.150.3.60")
    assert actual["seed_physical_device"] == 4
    assert actual["seed_visibility"] == "4,5,6,7"  # never copied D's 0,1,2,3


@pytest.mark.parametrize("visible,device", [(None, 0), ("4,0", 1), ("4,5", 2)])
def test_rejects_same_physical_device_or_invalid_visibility(monkeypatch, visible, device):
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="overlaps|outside"):
        T.apply_transport_environment(config(), device)


def test_rejects_arbitrary_environment_in_plan():
    value = config()
    value["environment"]["LD_PRELOAD"] = "bad"
    with pytest.raises(ValueError, match="Invalid captured"):
        T.apply_transport_environment(value, 4)


def test_prefill_discovery_ignores_stopped_workers_but_requires_live_source(tmp_path):
    log = tmp_path / "D.log"
    log.write_text(
        "(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout: current\n"
        "(Worker_DP1_TP0_EP4 pid=30) Mooncake payload layout: stopped\n"
    )
    worker(tmp_path, 20, dict(HCCL_INTRA_ROCE_ENABLE="1"))
    result = T.read_transport_environment([log], tmp_path / "proc", skip_stopped=True)
    assert result["source_pids"] == [1020]
    (tmp_path / "proc/20/cmdline").unlink()
    with pytest.raises(ValueError, match="Cannot locate live"):
        T.read_transport_environment([log], tmp_path / "proc", skip_stopped=True)


@pytest.mark.parametrize("live", [True, False])
def test_prefill_seed_inherits_on_snapshot_host_before_native_import(tmp_path, live):
    (tmp_path / "torch.py").write_text("raise RuntimeError('stop at native import')\n")
    log = tmp_path / "D_dp0_port7920.log"
    log.write_text("(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout: current\n")
    if live:
        worker(
            tmp_path,
            20,
            dict(
                HCCL_INTRA_ROCE_ENABLE="1",
                ASCEND_BUFFER_POOL="4:8",
                HCCL_IF_IP="7.150.3.60",
                HCCL_SOCKET_IFNAME="enp67s0f5",
                ASCEND_RT_VISIBLE_DEVICES="0,1,2,3",
                LMCACHE_PREFILL_STORE_DEBUG="1",
            ),
        )
    plan = dict(seed_transport_logs=[str(log)], seed_device=0)
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(TOOLS))
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    env["HCCL_IF_IP"] = "7.150.2.72"  # A client's address must not survive.
    env.pop("LMCACHE_PREFILL_STORE_DEBUG", None)
    script = (
        "import json,sys,functools; from pathlib import Path; import kv_replay_transport as t; "
        "t.read_transport_environment=functools.partial(t.read_transport_environment,proc_root=Path(sys.argv[2])); "
        "from prefill_kv_seed import seed; seed(json.loads(sys.argv[1]))"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, json.dumps(plan), str(tmp_path / "proc")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert '"status": "validating"' not in child.stdout
    if not live:
        assert "Cannot locate live" in child.stderr
        assert "stop at native import" not in child.stderr
        return
    assert "stop at native import" in child.stderr
    line = next(line for line in child.stdout.splitlines() if '"status": "transport_config"' in line)
    actual = json.loads(line.split("] ", 1)[1])
    assert actual["source_pids"] == [1020]
    assert actual["worker_pids"] == [20]
    assert actual["environment"] == dict(
        HCCL_INTRA_ROCE_ENABLE="1",
        ASCEND_BUFFER_POOL="4:8",
        HCCL_IF_IP="7.150.3.60",
        HCCL_SOCKET_IFNAME="enp67s0f5",
    )
    assert actual["seed_physical_device"] == 0
    assert actual["seed_visibility"] is None


def test_follows_engine_parent_and_does_not_choose_unrelated_service(tmp_path):
    log = tmp_path / "D.log"
    log.write_text("(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout:\n")
    worker(tmp_path, 20, dict(HCCL_INTRA_ROCE_ENABLE="1"))
    worker(tmp_path, 21, dict(HCCL_INTRA_ROCE_ENABLE="0"))  # unrelated service
    engine = tmp_path / "proc/30"
    engine.mkdir()
    (engine / "cmdline").write_bytes(b"VLLM::EngineCore\0")
    (engine / "status").write_text("PPid:\t1020\n")
    (tmp_path / "proc/20/status").write_text("PPid:\t30\n")
    result = T.read_transport_environment([log], tmp_path / "proc")
    assert result["source_pids"] == [1020]
    assert result["environment"]["HCCL_INTRA_ROCE_ENABLE"] == "1"


@pytest.mark.parametrize("problem", ["renamed", "no_environment", "missing_parent", "cycle"])
def test_never_falls_back_to_partial_worker_environment(tmp_path, problem):
    log = tmp_path / "D.log"
    log.write_text("(Worker_DP0_TP0_EP0 pid=20) Mooncake payload layout:\n")
    worker(tmp_path, 20, dict(HCCL_INTRA_ROCE_ENABLE="1"))
    launcher = tmp_path / "proc/1020"
    (launcher / "status").write_text("PPid:\t0\n")
    if problem in ("renamed", "cycle"):
        (launcher / "cmdline").write_bytes(b"VLLM::APIServer\0")
        if problem == "cycle":
            (launcher / "status").write_text("PPid:\t20\n")
    elif problem == "no_environment":
        (launcher / "environ").write_bytes(b"\0" * 500)
    else:
        (launcher / "cmdline").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        T.read_transport_environment([log], tmp_path / "proc", skip_stopped=True)


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/python3", "/usr/bin/vllm", "serve", "/model"],
        ["python3", "-m", "vllm", "serve", "/model"],
        ["python3", "-m", "vllm.entrypoints.openai.api_server"],
    ],
)
def test_supported_unrenamed_launchers(argv):
    assert T._is_service_launcher(argv)
