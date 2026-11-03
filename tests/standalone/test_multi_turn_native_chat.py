# SPDX-License-Identifier: Apache-2.0
"""Exercise native tokenization, exact prefix retention and both real clients."""

import json
import re
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import prefill_kv_replay  # noqa: E402
from multi_turn_chat import NativeDialogue, restore_native_tokens  # noqa: E402


def render(messages):
    # Model-template behavior: role tokens, trimmed assistant messages, a
    # closed reasoning header in non-thinking mode. No manual stop strings.
    text = "[gMASK]<sop>"
    for message in messages:
        role, content = message["role"], message["content"]
        text += f"<|{role}|>"
        text += "</think>" + content.strip() if role == "assistant" else content
    return [ord(c) for c in text + "<|assistant|></think>"]


@contextmanager
def server(concurrency=1, mutate=False, token_count_delta=0):
    state = dict(requests=[], tokenizations=[], completed=set())
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency, timeout=10)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                assert body["chat_template_kwargs"] == dict(enable_thinking=False, clear_thinking=False)
                assert body["add_special_tokens"] is False
                assert body["add_generation_prompt"] is True
                messages = body["messages"]
                tokens = render(messages)
                if mutate and len(messages) > 1:
                    tokens[0] += 1
                with lock:
                    state["tokenizations"].append(messages)
                data = json.dumps(dict(tokens=tokens, count=len(tokens))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            assert self.path == "/v1/completions"
            assert isinstance(body["prompt"], list)
            assert body["add_special_tokens"] is False
            assert body["temperature"] == 0
            assert "stop" not in body and "chat_template" not in body
            match = re.search(r"-s(\d+)-", body["request_id"])
            session = int(match[1]) if match else 0
            turn = int(body["request_id"].rsplit("-", 1)[1])
            with lock:
                state["requests"].append((session, turn, body))
                if turn:
                    assert len({s for s, t in state["completed"] if t == 0}) == concurrency
                    assert (session, turn - 1) in state["completed"]
            if not turn:
                barrier.wait()
            answer = f" \r\n真实回答 {session}/{turn}。\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            identifier = "cmpl-" + str(uuid.uuid4())
            for text in (answer[:3], answer[3:]):
                event = dict(id=identifier, choices=[dict(index=0, text=text, finish_reason=None)])
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            event = dict(
                id=identifier,
                choices=[dict(index=0, text="", finish_reason="stop")],
                usage=dict(
                    prompt_tokens=len(body["prompt"]) + token_count_delta,
                    completion_tokens=8,
                    prompt_tokens_details=dict(cached_tokens=4 if turn else 0),
                ),
            )
            with lock:
                state["completed"].add((session, turn))
            self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}", state
        finally:
            http.shutdown()
            worker.join(timeout=5)


def invoke(tmp_path, url, concurrent, *extra):
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("固定文章。\n助手：\n", encoding="utf-8")
    questions = tmp_path / "questions.txt"
    questions.write_text("依据？\n结论？\n下一步？\n", encoding="utf-8")
    output = tmp_path / "run"
    script = "multi_turn_kv_reuse_concurrent.py" if concurrent else "multi_turn_kv_reuse.py"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(REPO / "tools" / script),
        "--base-url",
        url,
        "--tokenizer-url",
        url + "/v1",
        "--model-name",
        "glm",
        "--prefix-file",
        str(prefix),
        "--questions-file",
        str(questions),
        "--rounds",
        "3",
        "--output-dir",
        str(output),
    ]
    if concurrent:
        command += ["--concurrency", "3"]
    return subprocess.run(command + list(extra), capture_output=True, text=True, encoding="utf-8", timeout=30), output


@pytest.mark.parametrize("concurrent", [False, True])
def test_native_inputs_keep_prior_tokens_and_archive_exact_replay(tmp_path, concurrent, monkeypatch):
    concurrency = 3 if concurrent else 1
    with server(concurrency) as (url, state):
        result, root = invoke(tmp_path, url, concurrent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(state["requests"]) == 4 * concurrency
    assert (root / "initial_prompt.txt").read_text(encoding="utf-8") == "固定文章。"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["prompt_format"] == "native"
    for session in range(concurrency):
        folder = root / "sessions" / f"{session:03d}" if concurrent else root
        previous = []
        for _, turn, payload in sorted(row for row in state["requests"] if row[0] == session):
            tokens = payload["prompt"]
            assert tokens[: len(previous)] == previous
            assert restore_native_tokens(folder, turn) == tokens
            turn_dir = folder / "turns" / f"{turn:03d}"
            check = json.loads((turn_dir / "prefix_check.json").read_text())
            assert check["unchanged"] and check["common_prefix_tokens"] == len(previous)
            delta = json.loads((turn_dir / "prompt_token_delta.json").read_text())
            assert len(delta["token_ids"]) == len(tokens) - len(previous)
            assert (turn_dir / "output.txt").read_bytes().decode() == f" \r\n真实回答 {session}/{turn}。\n\n"
            previous = tokens
        if concurrent:
            replay = prefill_kv_replay.restore_turn(root, session, 3)
            assert replay["payload"]["prompt"] == previous
            assert render(replay["native_messages"]) == previous
            assert render(replay["previous_messages"]) == replay["previous_tokens"]
    # All real history replies arrive at the tokenizer verbatim; trimming is
    # the model template's operation, not mutation by the benchmark client.
    messages = max(state["tokenizations"], key=len)
    assert messages[1]["content"].startswith(" \r\n真实回答")
    assert messages[1]["content"].endswith("\n\n")
    if concurrent:
        # Exercise preparation from the actual benchmark archives, including
        # rejection when a restarted service renders a different template.
        startup = tmp_path / "P.log"
        startup.write_text("[KV_FILE_REPLAY] enabled role=P\n", encoding="utf-8")
        monkeypatch.setattr(prefill_kv_replay, "live_contract", lambda paths: dict(tp_size=4))

        def http(url, path, body=None):
            if path == "/v1/models":
                return dict(data=[dict(id="glm", root="/models/glm")])
            assert path == "/tokenize"
            return dict(tokens=render(body["messages"]))

        monkeypatch.setattr(prefill_kv_replay, "http_json", http)
        args = prefill_kv_replay.parser().parse_args(
            [
                "--run-dir",
                str(root),
                "--session",
                "0",
                "--round",
                "3",
                "--p-logs",
                str(startup),
                "--local-hostname",
                "127.0.0.1",
            ]
        )
        out = tmp_path / "replay"
        out.mkdir()
        restored, plan = prefill_kv_replay.prepare(args, out)
        assert restored["payload"]["prompt"] == render(restored["native_messages"])
        assert plan["previous_prompt_digest"] == prefill_kv_replay.token_digest(restored["previous_tokens"])

        def changed_template(url, path, body=None):
            response = http(url, path, body)
            if path == "/tokenize":
                response["tokens"][0] += 1
            return response

        monkeypatch.setattr(prefill_kv_replay, "http_json", changed_template)
        with pytest.raises(ValueError, match="differs from the archived input token IDs"):
            prefill_kv_replay.prepare(args, out)


def test_incompatible_template_fails_before_model_warmup(tmp_path):
    with server(mutate=True) as (url, state):
        result, _ = invoke(tmp_path, url, False)
    assert result.returncode == 1
    assert "preflight failed" in result.stderr
    assert state["requests"] == []


def test_unavailable_tokenizer_endpoint_has_no_silent_raw_fallback(tmp_path):
    with server() as (url, state):
        result, _ = invoke(tmp_path, url, False, "--tokenizer-url", "http://127.0.0.1:1")
    assert result.returncode == 1
    assert state["requests"] == []


def test_server_token_count_mismatch_stops_after_first_response(tmp_path):
    with server(token_count_delta=1) as (url, state):
        result, root = invoke(tmp_path, url, False)
    assert result.returncode == 1
    assert "Server prompt_tokens=" in result.stderr
    assert len(state["requests"]) == 1
    assert (root / "turns/000/response.json").exists()


def test_runtime_prefix_rewrite_preserves_evidence_and_refuses_send(tmp_path):
    class Renderer:
        def tokens(self, messages):
            return [1, 2] if len(messages) == 1 else [1, 3, 4]

    chat = NativeDialogue(Renderer(), "article")
    for turn in range(2):
        folder = tmp_path / "turns" / f"{turn:03d}"
        folder.mkdir(parents=True)
        (folder / "request.json").write_text("{}")
        if turn:
            chat.append("answer", "question")
            with pytest.raises(RuntimeError, match="token 1/2"):
                chat.prepare(dict(request_id="test"), folder)
            assert not (folder / "prompt_token_delta.json").exists()
            assert json.loads((folder / "prefix_check.json").read_text())["unchanged"] is False
        else:
            chat.prepare(dict(request_id="test"), folder)


def test_corrupt_token_archive_is_rejected(tmp_path):
    folder = tmp_path / "turns/000"
    folder.mkdir(parents=True)
    (folder / "prompt_token_delta.json").write_text('{"prefix_tokens":1,"token_count":2,"token_ids":[2]}')
    (folder / "request.json").write_text('{"prompt_token_count":2}')
    with pytest.raises(ValueError, match="Invalid native token delta"):
        restore_native_tokens(tmp_path, 0)
