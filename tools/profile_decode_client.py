#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the entire decode request on existing vLLM Ascend D instances.

Send requests through the existing P/D proxy with --base-url and pass each
possible decoder's direct API address with --d-url. Profiling starts on all
listed decoders before the request and stops after its full response. This
includes D-side KV loading and every generation step. The default request has
exactly 50,000 input tokens and generates 128 output tokens.

Run inside the D container and pass --profile-dir to export newly created
captures into MindStudio trace_view.json files. Without a locally accessible
profile directory, captures remain in each server's configured profiler path.
The D servers need --profiler-config with profiler="torch" enabled.
The updated P/D proxy returns actual per-request routes for this client. Each
selected decoder's address and configured server-side profile directory are
printed and saved in result.json, including any recompute route changes.

Example command
  python tools/profile_decode_client.py --base-url http://proxy:9000 \
      --d-url http://127.0.0.1:7920 --profile-dir /path/to/D_profiles

The HTTP client and trace exporter use the neighboring profile_prefill_client
module. Trace export runs after all profilers have stopped.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from profile_prefill_client import Client, analyse_runs, profile_runs, repeat_tokens, write_json

PREFIX = "[DECODE_PROFILE]"


def preflight(decoders, tokenizer, model, required_tokens):
    paths = tokenizer.request("/openapi.json").get("paths", {})
    if "/tokenize" not in paths:
        raise RuntimeError("The tokenizer API needs /tokenize; pass a vLLM instance with --tokenizer-url")
    models = tokenizer.request("/v1/models").get("data", [])
    if model is None:
        if len(models) != 1:
            raise ValueError("Use --model to select a served model")
        model = models[0]["id"]
    instances = []
    for client in decoders:
        instance = {
            "url": client.base_url,
            "profile_dir": None,
            "dp_rank": None,
        }
        paths = client.request("/openapi.json").get("paths", {})
        missing = {"/start_profile", "/stop_profile"} - paths.keys()
        if missing:
            raise RuntimeError(
                f"{client.base_url} is missing {sorted(missing)}; enable --profiler-config on this D instance"
            )
        models = client.request("/v1/models").get("data", [])
        selected = next((item for item in models if item["id"] == model), None)
        if selected is None:
            raise ValueError(f"Model {model!r} is absent from {client.base_url}")
        maximum = selected.get("max_model_len")
        if maximum is not None and maximum < required_tokens:
            raise ValueError(f"{client.base_url} max_model_len={maximum}, required at least {required_tokens}")
        if "/server_info" in paths:
            config = client.request("/server_info?config_format=json").get("vllm_config", {})
            profiler = config.get("profiler_config") or {}
            if profiler.get("profiler") != "torch":
                raise ValueError(f"{client.base_url} requires profiler='torch' for MindStudio traces")
            for field in ("delay_iterations", "max_iterations", "warmup_iterations", "wait_iterations"):
                if profiler.get(field, 0):
                    raise ValueError(f"Set {field}=0 on {client.base_url} to capture the whole decode request")
            role = (config.get("kv_transfer_config") or {}).get("kv_role")
            if role == "kv_producer":
                raise ValueError(f"{client.base_url} is a P instance; --d-url needs a decoder")
            instance["profile_dir"] = profiler.get("torch_profiler_dir")
            instance["dp_rank"] = (config.get("parallel_config") or {}).get("data_parallel_rank")
        instances.append(instance)
    return model, instances


def make_prompt(client, model, count):
    # A fixed seed text makes token IDs reproducible with the same tokenizer.
    result = client.request(
        "/tokenize",
        {
            "model": model,
            "prompt": (
                "The archive records measurements of rivers, mountains, cities and forests. "
                "Read the records carefully and summarize their patterns.\n"
            ),
            "add_special_tokens": False,
        },
    )
    return repeat_tokens(result["tokens"], count)


def measured_request(client, model, prompt, output_tokens, root):
    payload = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 1024,
        "stream": False,
    }
    write_json(root / "request.json", payload)
    started = time.perf_counter()
    response = client.request("/v1/completions", payload, headers={"X-VLLM-Profile-Routes": "1"})
    elapsed = time.perf_counter() - started
    write_json(root / "response.json", response)
    choices = response.get("choices", [])
    usage = response.get("usage") or {}
    if response.get("error"):
        raise RuntimeError(f"Completion failed: {response['error']}")
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"length", "stop"}:
        raise RuntimeError("Expected one completed response; see response.json")
    if usage.get("prompt_tokens") != len(prompt) or usage.get("completion_tokens") != output_tokens:
        raise RuntimeError(f"Expected {len(prompt)} prompt and {output_tokens} output tokens, received {usage}")
    root.joinpath("output.txt").write_text(choices[0].get("text") or "", encoding="utf-8")
    return {
        "response_id": response.get("id"),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "request_seconds": elapsed,
        "finish_reason": choices[0]["finish_reason"],
        "routes": response.get("vllm_profile_routes", []),
    }


def report_routes(measurement, instances, report):
    routes = measurement["routes"]
    if not routes:
        report["routing_status"] = "proxy_metadata_unavailable"
        print(
            f"{PREFIX} proxy route metadata unavailable; update the P/D proxy to report the executed instance",
            flush=True,
        )
        return
    instance_by_url = {instance["url"]: instance for instance in instances}
    resolved = []
    for route in routes:
        decoder_url = str(route["decoder_url"]).rstrip("/").removesuffix("/v1")
        instance = instance_by_url.get(decoder_url)
        profile_dir = instance["profile_dir"] if instance else None
        dp_rank = route.get("decoder_dp_rank")
        if dp_rank is None and instance:
            dp_rank = instance["dp_rank"]
        trace_pattern = (
            profile_dir.rstrip("/") + "/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json" if profile_dir else None
        )
        resolved.append(
            {
                **route,
                "decoder_dp_rank": dp_rank,
                "profile_dir": profile_dir,
                "trace_path_pattern": trace_pattern,
                "listed_for_profiling": instance is not None,
            }
        )
        print(
            f"{PREFIX} executed decoder={decoder_url}, dp_rank={dp_rank}, "
            f"request={route.get('request_id')}, server_profile_dir={profile_dir or 'unavailable'}",
            flush=True,
        )
        if trace_pattern:
            print(f"{PREFIX} MindStudio path on {urlsplit(decoder_url).netloc}: {trace_pattern}", flush=True)
    report["routing_status"] = "reported_by_proxy"
    report["executed_routes"] = resolved


def profile_request(decoders, client, model, prompt, output_tokens, root, report):
    attempted = []
    request_error = None
    try:
        for decoder in decoders:
            # Include an attempted start in cleanup if its HTTP reply times out.
            attempted.append(decoder)
            decoder.request("/start_profile", {})
            print(f"{PREFIX} started {decoder.base_url}", flush=True)
        report["profile_started_at_ns"] = time.time_ns()
        return measured_request(client, model, prompt, output_tokens, root)
    except BaseException as error:
        request_error = error
        raise
    finally:
        stop_errors = []
        for decoder in reversed(attempted):
            try:
                decoder.request("/stop_profile", {})
                print(f"{PREFIX} stopped {decoder.base_url}", flush=True)
            except Exception as error:
                stop_errors.append(f"{decoder.base_url}: {error}")
        report["profile_stopped_at_ns"] = time.time_ns()
        if stop_errors:
            report["profiler_stop_errors"] = stop_errors
            print(f"{PREFIX} profiler stop errors: {' | '.join(stop_errors)}", file=sys.stderr, flush=True)
            if request_error is None:
                raise RuntimeError("Profiler stop failed; see result.json for the affected instances")


def capture(args):
    if args.prefix_tokens <= 0 or args.output_tokens <= 0 or args.timeout <= 0:
        raise ValueError("Token counts and --timeout must be positive")
    key = os.environ.get("OPENAI_API_KEY")
    client = Client(args.base_url, args.timeout, key)
    decoders = [Client(url, args.timeout, key) for url in args.d_url]
    if len({decoder.base_url for decoder in decoders}) != len(decoders):
        raise ValueError("Pass each --d-url once")
    tokenizer = Client(args.tokenizer_url or decoders[0].base_url, args.timeout, key)
    run_id = uuid.uuid4().hex
    root = (args.output or Path(f"decode-profile-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preflight",
        "run_id": run_id,
        "base_url": client.base_url,
        "d_urls": [decoder.base_url for decoder in decoders],
        "tokenizer_url": tokenizer.base_url,
        "prefix_tokens": args.prefix_tokens,
        "output_tokens": args.output_tokens,
        "profile_dir": str(args.profile_dir) if args.profile_dir else None,
        "capture_scope": "D-side request including initial KV loading and all generation steps",
    }
    try:
        model, instances = preflight(decoders, tokenizer, args.model, args.prefix_tokens + args.output_tokens)
        report["model"] = model
        report["decoder_instances"] = instances
        for instance in instances:
            print(
                f"{PREFIX} decoder={instance['url']}, server_profile_dir={instance['profile_dir'] or 'unavailable'}",
                flush=True,
            )
        prompt = make_prompt(tokenizer, model, args.prefix_tokens)
        report["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        before = profile_runs(args.profile_dir) if args.profile_dir else set()
        report["status"] = "capturing"
        write_json(root / "result.json", report)
        print(f"{PREFIX} input={len(prompt):,}, output={args.output_tokens:,}, request={client.base_url}", flush=True)
        report["measurement"] = profile_request(decoders, client, model, prompt, args.output_tokens, root, report)
        report_routes(report["measurement"], instances, report)
        report["status"] = "captured"
        if args.profile_dir:
            runs = profile_runs(args.profile_dir) - before
            report["raw_profiles"] = sorted(map(str, runs))
            write_json(root / "raw_profiles.json", report["raw_profiles"])
            write_json(root / "result.json", report)
            report["traces"] = analyse_runs(runs, root, prefix=PREFIX)
            report["status"] = "exported"
        else:
            print(f"{PREFIX} raw captures saved under each D server's torch_profiler_dir", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        write_json(root / "result.json", report)
        print(f"{PREFIX} result directory: {root}", flush=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--base-url", default="http://127.0.0.1:9000", help="Request URL, normally the existing P/D proxy")
    cli.add_argument("--d-url", action="append", help="Direct D API URL; repeat for every decoder the proxy may select")
    cli.add_argument("--tokenizer-url", help="vLLM tokenization URL; defaults to the first --d-url")
    cli.add_argument("--model", help="Served model ID; auto-selected when exactly one is listed")
    cli.add_argument("--prefix-tokens", type=int, default=50000, help="Exact input token count")
    cli.add_argument("--output-tokens", type=int, default=128, help="Exact generated token count")
    cli.add_argument("--timeout", type=float, default=7200, help="HTTP timeout in seconds")
    cli.add_argument("--output", type=Path, help="New client result directory")
    cli.add_argument("--profile-dir", type=Path, help="Locally visible D profiler root; exports only new captures")
    cli.add_argument("--analyse-only", type=Path, help="Export existing captures inside the Ascend environment")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.analyse_only:
        directory = args.analyse_only.resolve(strict=True)
        runs = {directory} if directory.name.endswith("_ascend_pt") else profile_runs(directory)
        analyse_runs(runs, directory, prefix=PREFIX)
        return
    if not args.d_url:
        cli.error("--d-url is required for capture")
    if args.profile_dir:
        args.profile_dir = args.profile_dir.resolve(strict=True)
    capture(args)


if __name__ == "__main__":
    main()
