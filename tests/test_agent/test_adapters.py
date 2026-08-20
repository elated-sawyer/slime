"""Unit tests for the agent HTTP adapters (Anthropic + OpenAI) and parsing.

Every test drives a REAL adapter over a real aiohttp loopback
(``TestServer``/``TestClient``) and a real ``/generate`` upstream
(:class:`tests.test_agent._fakes.FakeSGLangServer`) -- so the whole
translate -> sglang -> parse -> record_turn -> finish_session path runs
unmocked; only the model server and tokenizer are faked. Covers both wire
protocols plus the standalone parsing helpers in ``slime.agent.parsing``.

Replaces the pre-refactor ``tests/test_agent_adapters.py`` (which imported now-
removed symbols and a dropped ``/v1/responses`` endpoint).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSGLangServer, FakeTokenizer, ScriptedTokenizer  # noqa: E402

from slime.agent.adapters import anthropic, openai, openai_responses  # noqa: E402
from slime.agent.parsing import ParsedModelOutput, parse_model_output, parse_xml_tool_uses  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Headers:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _parse_sse(raw: str) -> list[tuple[str, object]]:
    """Parse an SSE byte-stream body into ``(event_name, payload)`` pairs;
    ``payload`` is the decoded JSON, or the literal ``"[DONE]"``."""
    events: list[tuple[str, object]] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            data = "\n".join(data_lines)
            events.append((event_name, data if data == "[DONE]" else json.loads(data)))
        event_name = "message"
        data_lines = []

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    flush()
    return events


async def _drain(adapter, sid) -> list[Sample]:
    return await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)


# ===========================================================================
# §1 session-id resolution
# ===========================================================================


def test_anthropic_session_id_prefers_bearer_then_api_key():
    assert anthropic._request_session_id(_Headers({"X-Api-Key": "key"})) == "key"
    assert anthropic._request_session_id(_Headers({"Authorization": "Bearer bsid", "X-Api-Key": "key"})) == "bsid"
    assert anthropic._request_session_id(_Headers({})) == "default"


def test_openai_session_id_prefers_bearer_then_body():
    req = _Headers({})
    assert openai._request_session_id(req, {"metadata": {"session_id": "meta"}, "user": "u"}) == "meta"
    assert openai._request_session_id(req, {"user": "u"}) == "u"
    assert openai._request_session_id(_Headers({"Authorization": "Bearer bsid"}), {"user": "u"}) == "bsid"
    assert openai._request_session_id(req, {}) == "default"


# ===========================================================================
# §2 translation (wire -> chat-template messages)
# ===========================================================================


def test_anthropic_translation_keeps_tool_results_thinking_and_tools():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "name": "lookup", "input": {"q": "slime"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "result"}]},
    ]
    translated = anthropic._translate_messages(messages, system="sys")
    assert translated == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "plan",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "slime"}}}],
        },
        {"role": "tool", "content": "result"},
    ]
    tools = anthropic._tools_to_chat_tools(
        [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}]
    )
    assert tools == [
        {"type": "function", "function": {"name": "lookup", "description": "search", "parameters": {"type": "object"}}}
    ]


def test_openai_translation_developer_to_system_and_tool_calls_to_dict():
    translated = openai._translate_messages(
        [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "slime"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
        ]
    )
    assert translated == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
        # wire-only id dropped; arguments coerced JSON-string -> dict.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "slime"}}}],
        },
        # tool_call_id dropped.
        {"role": "tool", "content": "found"},
    ]


def test_responses_items_roundtrip_mixed_text_reasoning_and_function_call():
    parsed = ParsedModelOutput(
        reasoning="inspect first",
        text="I will check. ",
        tool_uses=[{"name": "exec_command", "input": {"cmd": "ls"}}],
    )
    output = openai_responses._output_items(parsed, "stop")
    messages = openai_responses._input_items_to_messages(
        [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "solve"}]},
            *output,
            {"type": "function_call_output", "call_id": output[-1]["call_id"], "output": "file.txt"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "solve"},
        openai_responses._manager_message(parsed),
        {"role": "tool", "content": "file.txt"},
    ]
    assistant = messages[2]
    assert assistant["content"] == "I will check. "
    assert assistant["reasoning_content"] == "inspect first"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}


def test_responses_translation_normalizes_system_messages_without_reordering_history():
    adapter = openai_responses.OpenAIResponsesAdapter(
        tokenizer=object(),
        sglang_url="http://127.0.0.1:1",
    )
    translated, tools = adapter._translate(
        {
            "instructions": "top-level instructions",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "leading developer"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "solve"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "late developer"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
        }
    )

    assert tools is None
    assert translated == [
        {
            "role": "system",
            "content": "top-level instructions\n\nleading developer",
        },
        {"role": "user", "content": "solve"},
        {
            "role": "user",
            "content": "<system-reminder>\nlate developer\n</system-reminder>",
        },
        {"role": "user", "content": "continue"},
    ]


# ===========================================================================
# §3 non-stream JSON + token capture (real HTTP, real /generate)
# ===========================================================================


def test_anthropic_messages_nonstream_records_token_segments():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 101), (-0.2, 102)]]) as sglang:
            tok = FakeTokenizer(outputs={(101, 102): "done now"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-a")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-a"},
                    json={"model": "m", "max_tokens": 7, "messages": [{"role": "user", "content": "hi"}]},
                )
                data = await resp.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-a")

        assert resp.status == 200
        assert data["type"] == "message" and data["stop_reason"] == "end_turn"
        assert data["content"] == [{"type": "text", "text": "done now"}]
        # adapter posted the rendered prompt ids and capped max_new_tokens at max_tokens.
        assert sglang.requests[0]["sampling_params"]["max_new_tokens"] == 7
        assert sglang.routing_keys == ["sid-a"]
        # one trained turn: the two response ids carry loss=1 + real logprobs.
        assert len(samples) == 1
        s = samples[0]
        assert s.tokens[-2:] == [101, 102]
        assert s.loss_mask[-2:] == [1, 1]
        assert s.rollout_log_probs[-2:] == [-0.1, -0.2]
        assert s.response == "done now"

    asyncio.run(run_case())


def test_openai_chat_completions_nonstream_records_token_segments():
    async def run_case():
        async with FakeSGLangServer([[(-0.3, 201)]]) as sglang:
            tok = FakeTokenizer(outputs={(201,): "hello"})
            adapter = openai.OpenAIAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-o")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sid-o"},
                    json={"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi?"}]},
                )
                data = await resp.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-o")

        assert resp.status == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
        assert data["choices"][0]["finish_reason"] == "stop"
        assert sglang.requests[0]["sampling_params"]["max_new_tokens"] == 4
        assert len(samples) == 1 and samples[0].tokens[-1] == 201 and samples[0].loss_mask[-1] == 1

    asyncio.run(run_case())


def test_openai_responses_two_turn_sse_keeps_clean_token_attribution():
    async def run_case():
        wire_records = []
        first_output = (
            "I will inspect. "
            "<tool_call><function=exec_command><parameter=cmd>ls</parameter></function></tool_call>"
        )
        tokenizer = ScriptedTokenizer(
            prompts=[[1, 2], [1, 2, 10, 11, 20]],
            outputs={(10, 11): first_output, (30,): "done"},
        )
        async with FakeSGLangServer([[(-0.2, 10), (-0.3, 11)], [(-0.4, 30)]]) as sglang:
            adapter = openai_responses.OpenAIResponsesAdapter(
                tokenizer=tokenizer,
                sglang_url=sglang.url,
                wire_capture_callback=wire_records.append,
            )
            adapter.open_session("sid-r")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            base_input = [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "solve"}]},
            ]
            tools = [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "run a command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
                {"type": "web_search"},
            ]
            try:
                first = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-r"},
                    json={"model": "m", "stream": True, "input": base_input, "tools": tools},
                )
                first_events = _parse_sse(await first.text())
                first_completed = next(
                    payload for name, payload in first_events if name == "response.completed"
                )["response"]
                function_call = next(item for item in first_completed["output"] if item["type"] == "function_call")

                second = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-r"},
                    json={
                        "model": "m",
                        "stream": True,
                        "input": [
                            *base_input,
                            *first_completed["output"],
                            {
                                "type": "function_call_output",
                                "call_id": function_call["call_id"],
                                "output": "a.txt",
                            },
                        ],
                        "tools": tools,
                    },
                )
                second_events = _parse_sse(await second.text())
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-r")

        assert first.status == 200 and second.status == 200
        assert "text/event-stream" in first.headers["Content-Type"]
        assert first_completed["status"] == "completed"
        assert any(name == "response.function_call_arguments.done" for name, _ in first_events)
        assert any(name == "response.completed" for name, _ in second_events)
        assert [event["sequence_number"] for _, event in first_events] == list(range(len(first_events)))
        request_records = [record for record in wire_records if record.direction == "request"]
        assert len(request_records) == 2
        assert all(record.final and record.sequence == 0 for record in request_records)
        assert all(json.loads(record.payload)["model"] == "m" for record in request_records)
        response_groups = {
            exchange_id: [
                record
                for record in wire_records
                if record.direction == "response" and record.exchange_id == exchange_id
            ]
            for exchange_id in {record.exchange_id for record in request_records}
        }
        assert all(group for group in response_groups.values())
        assert all(
            [record.sequence for record in group] == list(range(len(group)))
            and sum(record.final for record in group) == 1
            and group[-1].final
            and b"response.completed" in group[-1].payload
            for group in response_groups.values()
        )
        added_items = [payload["item"] for name, payload in first_events if name == "response.output_item.added"]
        assert all(item["status"] == "in_progress" for item in added_items)
        assert all(not item.get("content") for item in added_items if item["type"] in {"reasoning", "message"})
        assert all(not item.get("arguments") for item in added_items if item["type"] == "function_call")
        assert sglang.routing_keys == ["sid-r", "sid-r"]
        assert tokenizer.rendered[0][1] == [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "run a command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            }
        ]
        replayed_assistant = next(
            message for message in tokenizer.rendered[1][0] if message.get("role") == "assistant"
        )
        assert replayed_assistant["content"] == "I will inspect."
        assert replayed_assistant["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
        assert len(samples) == 1
        assert sum(samples[0].loss_mask) == 3
        assert [
            value
            for value, mask in zip(samples[0].rollout_log_probs, samples[0].loss_mask, strict=True)
            if mask
        ] == [
            -0.2,
            -0.3,
            -0.4,
        ]

    asyncio.run(run_case())


# ===========================================================================
# §4 streaming SSE
# ===========================================================================


def test_anthropic_messages_streams_blocks():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 301)]]) as sglang:
            tok = FakeTokenizer(outputs={(301,): "streamed"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-as")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-as", "Accept": "text/event-stream"},
                    json={
                        "model": "m",
                        "stream": True,
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "x"}],
                    },
                )
                raw = await resp.text()
            finally:
                await client.close()
            await _drain(adapter, "sid-as")

        names = [name for name, _ in _parse_sse(raw)]
        assert names[0] == "message_start"
        assert "content_block_delta" in names
        assert names[-1] == "message_stop"
        deltas = [p for n, p in _parse_sse(raw) if n == "content_block_delta"]
        assert any(d["delta"].get("text") == "streamed" for d in deltas)

    asyncio.run(run_case())


def test_openai_chat_completions_streams_chunks_until_done():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 401)]]) as sglang:
            tok = FakeTokenizer(outputs={(401,): "streamed text"})
            adapter = openai.OpenAIAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-os")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sid-os"},
                    json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
                )
                raw = await resp.text()
            finally:
                await client.close()
            await _drain(adapter, "sid-os")

        events = _parse_sse(raw)
        chunks = [p for _, p in events if isinstance(p, dict)]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert events[-1] == ("message", "[DONE]")

    asyncio.run(run_case())


# ===========================================================================
# §5 multi-turn token alignment (tool call -> tool result -> answer)
# ===========================================================================


def test_anthropic_multiturn_wire_roundtrip_and_token_capture():
    """Two-turn round-trip: a tool-call turn, then a tool-result + answer turn.

    Asserts the adapter-level behaviour the branching test can't see: the wire
    tool_use block round-trips, both turns route to the same sid, and
    finish_session yields aligned training samples. (The fine-grained
    clean/drift linearization is owned by test_trajectory_manager_branching.)"""

    async def run_case():
        r1 = "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        async with FakeSGLangServer([[(-0.5, 700), (-0.5, 701)], [(-0.4, 800)]]) as sglang:
            tok = FakeTokenizer(outputs={(700, 701): r1, (800,): "the answer"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-mt")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            tools = [
                {"name": "lookup", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}
            ]
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-mt"},
                    json={
                        "model": "m",
                        "max_tokens": 5,
                        "tools": tools,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "find slime"}]}],
                    },
                )
                fdata = await first.json()
                tool_use = next(b for b in fdata["content"] if b["type"] == "tool_use")
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-mt"},
                    json={
                        "model": "m",
                        "max_tokens": 7,
                        "tools": tools,
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "find slime"}]},
                            {"role": "assistant", "content": fdata["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": tool_use["id"], "content": "found"}
                                ],
                            },
                        ],
                    },
                )
                await second.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-mt")

        assert first.status == 200 and second.status == 200
        assert fdata["stop_reason"] == "tool_use"
        assert tool_use["name"] == "lookup" and tool_use["input"] == {"query": "slime"}
        # both turns routed to the same sid; the adapter posted the growing prompt.
        assert sglang.routing_keys == ["sid-mt", "sid-mt"]
        assert (
            sglang.requests[1]["input_ids"][: len(sglang.requests[0]["input_ids"])] == sglang.requests[0]["input_ids"]
        )
        # finish_session produces at least one aligned, partly-trained sample.
        assert samples
        for s in samples:
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0

    asyncio.run(run_case())


# ===========================================================================
# §6 adapter behaviour: turn cap, mid-list system fold
# ===========================================================================


def test_max_turns_per_sid_returns_429():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 501)], [(-0.1, 502)]]) as sglang:
            tok = FakeTokenizer()
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url, max_turns_per_sid=1)
            adapter.open_session("sid-cap")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
                h = {"Authorization": "Bearer sid-cap"}
                first = await client.post("/v1/messages", headers=h, json=body)
                second = await client.post("/v1/messages", headers=h, json=body)
            finally:
                await client.close()
            await _drain(adapter, "sid-cap")
        assert first.status == 200
        assert second.status == 429

    asyncio.run(run_case())


def test_session_turn_cap_overrides_adapter_fallback():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 501)], [(-0.1, 502)], [(-0.1, 503)]]) as sglang:
            tok = FakeTokenizer()
            adapter = anthropic.AnthropicAdapter(
                tokenizer=tok,
                sglang_url=sglang.url,
                max_turns_per_sid=2,
                require_registered_session=True,
            )
            adapter.open_session("sid-one", max_turns=1)
            adapter.open_session("sid-fallback")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
            try:
                one_first = await client.post(
                    "/v1/messages", headers={"Authorization": "Bearer sid-one"}, json=body
                )
                one_second = await client.post(
                    "/v1/messages", headers={"Authorization": "Bearer sid-one"}, json=body
                )
                fallback_first = await client.post(
                    "/v1/messages", headers={"Authorization": "Bearer sid-fallback"}, json=body
                )
                fallback_second = await client.post(
                    "/v1/messages", headers={"Authorization": "Bearer sid-fallback"}, json=body
                )
                fallback_third = await client.post(
                    "/v1/messages", headers={"Authorization": "Bearer sid-fallback"}, json=body
                )
            finally:
                await client.close()
            await _drain(adapter, "sid-one")
            await _drain(adapter, "sid-fallback")

        assert [one_first.status, one_second.status] == [200, 429]
        assert [fallback_first.status, fallback_second.status, fallback_third.status] == [200, 200, 429]
        assert len(sglang.requests) == 3

    asyncio.run(run_case())


def test_locked_session_sampling_defaults_cannot_be_overridden_by_wire_body():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 501)], [(-0.1, 502)]]) as sglang:
            adapter = openai_responses.OpenAIResponsesAdapter(
                tokenizer=FakeTokenizer(),
                sglang_url=sglang.url,
                require_registered_session=True,
                lock_sampling_defaults=True,
            )
            adapter.open_session(
                "sid-locked",
                sampling_defaults={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_new_tokens": 128,
                    "seed": 17,
                },
            )
            adapter.open_session("sid-unset", sampling_defaults={"seed": 18})
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {
                "model": "m",
                "input": "x",
                "temperature": 0.9,
                "top_p": 0.1,
                "top_k": 5,
                "max_output_tokens": 64,
            }
            try:
                locked = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-locked"},
                    json=body,
                )
                unset = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-unset"},
                    json=body,
                )
            finally:
                await client.close()
            await _drain(adapter, "sid-locked")
            await _drain(adapter, "sid-unset")

        assert locked.status == unset.status == 200
        locked_sampling, unset_sampling = [item["sampling_params"] for item in sglang.requests]
        assert locked_sampling["temperature"] == 0.2
        assert locked_sampling["top_p"] == 0.8
        assert locked_sampling["top_k"] == 5
        assert locked_sampling["max_new_tokens"] == 128
        assert locked_sampling["sampling_seed"] == 17
        assert "seed" not in locked_sampling
        assert unset_sampling["temperature"] == 0.9
        assert unset_sampling["top_p"] == 0.1
        assert unset_sampling["top_k"] == 5
        assert unset_sampling["max_new_tokens"] == 64
        assert unset_sampling["sampling_seed"] == 18
        assert "seed" not in unset_sampling

    asyncio.run(run_case())


def test_strict_session_mode_rejects_unknown_bearer_before_generation():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 501)]]) as sglang:
            tok = FakeTokenizer(outputs={(501,): "done"})
            adapter = openai_responses.OpenAIResponsesAdapter(
                tokenizer=tok,
                sglang_url=sglang.url,
                require_registered_session=True,
            )
            adapter.open_session("sid-known")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            body = {"model": "m", "input": "x"}
            try:
                unknown = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-unknown"},
                    json=body,
                )
                unknown_payload = await unknown.json()
                known = await client.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer sid-known"},
                    json=body,
                )
            finally:
                await client.close()
            await _drain(adapter, "sid-known")

        assert unknown.status == 401
        assert unknown_payload["error"]["type"] == "invalid_session"
        assert known.status == 200
        assert len(sglang.requests) == 1

    asyncio.run(run_case())


def test_mid_list_system_folds_into_user():
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "skills list"},
            {"role": "user", "content": "next"},
        ]
    }
    changed = anthropic._fold_mid_list_system_into_user(body)
    assert changed
    # the mid-list system message is gone; its text is wrapped into the prior user.
    assert [m["role"] for m in body["messages"]] == ["user", "user"]
    folded = body["messages"][0]["content"]
    assert any(b.get("text", "").startswith("<system-reminder>") for b in folded)


# ===========================================================================
# §7 parsing helpers (slime.agent.parsing)
# ===========================================================================


def test_parse_model_output_plain_text_no_parsers():
    parsed = parse_model_output("just text", tools_schema=None, tool_parser_name=None, reasoning_parser_name=None)
    assert parsed.text == "just text"
    assert parsed.tool_uses == []
    assert parsed.reasoning == ""


def test_parse_model_output_think_split_fallback():
    # The qwen3 reasoning parser lives in sglang (lazy import); skip where the
    # lean CPU CI env has no sglang installed.
    pytest.importorskip("sglang")
    parsed = parse_model_output(
        "<think>reason here</think>visible",
        tools_schema=None,
        tool_parser_name=None,
        reasoning_parser_name="qwen3",
    )
    # the qwen3 reasoning parser (or the </think> fallback) splits reasoning out.
    assert "visible" in parsed.text
    assert "reason here" in parsed.reasoning


def test_parse_xml_tool_uses_fallback():
    raw = "lead <tool_call><function=lookup><parameter=q>slime</parameter></function></tool_call> tail"
    schema = [{"function": {"name": "lookup"}}]
    cleaned, uses = parse_xml_tool_uses(raw, schema)
    assert uses == [{"name": "lookup", "input": {"q": "slime"}}]
    assert "<tool_call>" not in cleaned
    assert "lead" in cleaned and "tail" in cleaned


def test_parse_xml_tool_uses_ignores_unknown_tool():
    raw = "<tool_call><function=unknown><parameter=q>x</parameter></function></tool_call>"
    cleaned, uses = parse_xml_tool_uses(raw, [{"function": {"name": "lookup"}}])
    assert uses == []
    assert "<tool_call>" in cleaned  # left untouched


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
