#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render native chat on the serving tokenizer and archive append-only inputs.

Only tokenization uses the direct tokenizer endpoint. Generation keeps the
existing completions/PD route, with token IDs as its prompt.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

NATIVE_FORMAT = "native-chat-v1"
TEMPLATE_KWARGS = (("enable_thinking", False), ("clear_thinking", False))


def add_chat_arguments(parser) -> None:
    parser.add_argument(
        "--prompt-format",
        choices=("native", "legacy"),
        default="native",
        help="Native model chat template (default), or the previous raw-text benchmark",
    )
    parser.add_argument(
        "--tokenizer-url",
        help="vLLM /tokenize server, e.g. the direct P URL; defaults to --base-url",
    )


def initial_user_text(text: str) -> str:
    """Remove only the benchmark's old trailing assistant cue, not article text."""
    for suffix in ("\r\n助手：\r\n", "\r\n助手：\n", "\n助手：\r\n", "\n助手：\n"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def common_prefix(before: list[int], after: list[int]) -> int:
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return index
    return min(len(before), len(after))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


class ChatRenderer:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.timeout = timeout

    def tokens(self, messages: list[dict[str, str]]) -> list[int]:
        payload = dict(
            model=self.model,
            messages=messages,
            add_generation_prompt=True,
            add_special_tokens=False,
            chat_template_kwargs=dict(TEMPLATE_KWARGS),
        )
        request = urllib.request.Request(
            self.base_url + "/tokenize",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            # Each rendering runs in its own session worker; no shared opener.
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                request,
                timeout=self.timeout,
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"Native chat /tokenize failed at {self.base_url}: HTTP {error.code}: {detail}. "
                "For a PD proxy without /tokenize, set --tokenizer-url to a direct P endpoint. "
                "No raw-text fallback was sent."
            ) from error
        tokens = result.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or result.get("count") != len(tokens)
        ):
            raise RuntimeError("Native /tokenize returned invalid token IDs/count")
        return tokens

    def preflight(self, root: Path) -> None:
        """Check template extension before paying for any long model request."""
        messages = [{"role": "user", "content": "请简短回答。"}]
        previous = self.tokens(messages)
        for answer, question in (("测试回答。", "依据呢？"), ("第二次回答。", "结论呢？")):
            messages.extend((dict(role="assistant", content=answer), dict(role="user", content=question)))
            current = self.tokens(messages)
            if common_prefix(previous, current) != len(previous):
                raise RuntimeError(
                    "Serving chat template rewrites the preceding prompt; native prefix preflight failed. "
                    "No model request was sent."
                )
            previous = current
        write_json(
            root / "native_chat.json",
            dict(
                format=NATIVE_FORMAT,
                tokenizer_url=self.base_url,
                model=self.model,
                chat_template_kwargs=dict(TEMPLATE_KWARGS),
                add_generation_prompt=True,
                add_special_tokens=False,
                preflight="passed",
                generation_endpoint="/v1/completions",
            ),
        )


class NativeDialogue:
    def __init__(self, renderer: ChatRenderer, initial: str):
        self.renderer = renderer
        self.messages = [dict(role="user", content=initial)]
        self.previous_tokens: list[int] = []

    def append(self, answer: str, question: str) -> None:
        # Preserve response text. Any whitespace handling is done exclusively
        # by the actual model template, identically on every subsequent turn.
        self.messages.extend((dict(role="assistant", content=answer), dict(role="user", content=question)))

    def prepare(self, payload: dict, directory: Path) -> dict:
        tokens = self.renderer.tokens(self.messages)
        preserved = common_prefix(self.previous_tokens, tokens)
        previous_count = len(self.previous_tokens)
        check = dict(
            previous_prompt_tokens=previous_count,
            common_prefix_tokens=preserved,
            current_prompt_tokens=len(tokens),
            unchanged=preserved == previous_count,
        )
        write_json(directory / "prefix_check.json", check)
        if not check["unchanged"]:
            raise RuntimeError(
                f"Native template changed the old prompt at token {preserved}/{previous_count}; "
                f"request not sent; see {directory / 'prefix_check.json'}"
            )
        write_json(
            directory / "prompt_token_delta.json",
            dict(
                prefix_tokens=previous_count,
                token_count=len(tokens),
                token_ids=tokens[previous_count:],
            ),
        )
        metadata_path = directory / "request.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            prompt_format=NATIVE_FORMAT,
            prompt_token_count=len(tokens),
            prompt_source="prompt_token_delta.json chain",
            add_special_tokens=False,
        )
        write_json(metadata_path, metadata)
        self.previous_tokens = tokens
        print(
            f"[MULTI_TURN_PREFIX] request_id={payload['request_id']} "
            f"previous={previous_count} preserved={preserved} prompt={len(tokens)} unchanged=true",
            flush=True,
        )
        return dict(payload, prompt=tokens, add_special_tokens=False)


def complete_native(client, dialogue: NativeDialogue, payload: dict, directory: Path):
    # Rendering/validation is outside request TTFT and off the coordinator
    # thread, so one session's tokenization does not serialize other users.
    prepared = dialogue.prepare(payload, directory)
    text, stats = client.complete(prepared, directory)
    actual = stats.get("prompt_tokens")
    if actual is not None and actual != len(prepared["prompt"]):
        raise RuntimeError(f"Server prompt_tokens={actual}, native input has {len(prepared['prompt'])} tokens")
    return text, stats


def restore_native_tokens(directory: Path, turn: int) -> list[int]:
    tokens: list[int] = []
    for current in range(turn + 1):
        folder = directory / "turns" / f"{current:03d}"
        delta = json.loads((folder / "prompt_token_delta.json").read_text(encoding="utf-8"))
        metadata = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        added = delta.get("token_ids")
        if (
            delta.get("prefix_tokens") != len(tokens)
            or not isinstance(added, list)
            or any(type(value) is not int or value < 0 for value in added)
        ):
            raise ValueError(f"Invalid native token delta at round {current}")
        tokens.extend(added)
        if len(tokens) != delta.get("token_count") or len(tokens) != metadata.get("prompt_token_count"):
            raise ValueError(f"Native prompt token count differs at round {current}")
    return tokens
