# SPDX-License-Identifier: Apache-2.0
"""Attach actual P/D routes to non-streaming profiling responses on request."""

import json


def profile_route(instance_info):
    reservation = instance_info.reservation
    return {
        "request_id": instance_info.request_id,
        "prefiller_url": instance_info.prefiller.url,
        "decoder_url": instance_info.decoder.url,
        "decoder_dp_rank": reservation.dp_rank if reservation is not None else None,
    }


def profile_response(body, routes):
    response = json.loads(body)
    if isinstance(response, dict):
        response["vllm_profile_routes"] = routes
    return json.dumps(response, ensure_ascii=False).encode("utf-8")


async def profile_response_chunks(chunks, routes):
    body = b"".join([chunk async for chunk in chunks])
    try:
        yield profile_response(body, routes)
    except (ValueError, UnicodeError):
        yield body
