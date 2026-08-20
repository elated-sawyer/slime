"""OpenAI Responses API adapter for token-capturing agent rollouts.

The adapter translates the client-managed Responses history into the same
chat-template message shape used by the other Slime agent adapters, sends one
turn to SGLang, and returns either a completed JSON Response or a Responses SSE
stream.  It deliberately owns no task or sandbox policy.

Responses represents one assistant turn as adjacent reasoning, message, and
function-call items.  They must be merged back into one assistant message before
rendering the next prompt; otherwise TrajectoryManager sees rewritten history
and sampled-token attribution is lost.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import BaseAdapter, Reply, flatten_content, manager_finish_reason
from slime.agent.adapters.openai import _arguments_as_dict, _request_session_id, _tools_to_chat_tools

logger = logging.getLogger(__name__)

_REASONING_PREFIX = "slime:"
_CONTEXT_CUTOFF_TEXT = (
    "I have reached the context budget for this task and must stop. "
    "The files already written in the workspace are my final answer."
)
_SYSTEM_REMINDER_PREFIX = "<system-reminder>\n"
_SYSTEM_REMINDER_SUFFIX = "\n</system-reminder>"


@dataclass(frozen=True, slots=True)
class ResponsesWireRecord:
    """One application-wire chunk emitted for a completed Responses exchange.

    Request JSON is captured in canonical UTF-8 form. Streaming response payloads
    are the exact SSE chunks passed to aiohttp; non-streaming responses are the
    exact JSON bytes returned to the client. The callback is a production audit
    hook: exceptions propagate so a configured capture cannot fail silently.
    """

    session_id: str
    exchange_id: str
    direction: str
    sequence: int
    content_type: str
    payload: bytes
    final: bool


class OpenAIResponsesAdapter(BaseAdapter):
    """Serve ``/v1/responses`` while retaining Slime's sampled token spans."""

    logger = logger
    log_prefix = "openai_responses_adapter"
    max_token_keys = ("max_output_tokens", "max_tokens")
    stop_keys = ()

    def __init__(
        self,
        *args,
        wire_capture_callback: Callable[[ResponsesWireRecord], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wire_capture_callback = wire_capture_callback

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/responses", self._run_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request, body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        input_data = body.get("input", "")
        if not isinstance(input_data, (str, list)):
            raise web.HTTPBadRequest(text="input must be a string or a list")

        messages: list[dict[str, Any]] = []
        instructions = body.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": flatten_content(instructions)})

        if isinstance(input_data, str):
            if input_data:
                messages.append({"role": "user", "content": input_data})
        else:
            messages.extend(_input_items_to_messages(input_data))
        messages = _normalize_system_messages(messages)

        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise web.HTTPBadRequest(text="tools must be a list")
        return messages, _tools_to_chat_tools(tools)

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        items = _output_items(parsed, raw_finish)
        return Reply(
            manager_message=_manager_message(parsed),
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=items,
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        envelope = _response_envelope(body, reply.wire, in_tok, out_tok)
        session_id = self._session_id(request, body)
        exchange_id = envelope["id"]
        self._capture_wire(
            ResponsesWireRecord(
                session_id=session_id,
                exchange_id=exchange_id,
                direction="request",
                sequence=0,
                content_type="application/json",
                payload=json.dumps(
                    body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                final=True,
            )
        )
        if stream:
            sequence = 0

            def capture_chunk(payload: bytes, final: bool) -> None:
                nonlocal sequence
                self._capture_wire(
                    ResponsesWireRecord(
                        session_id=session_id,
                        exchange_id=exchange_id,
                        direction="response",
                        sequence=sequence,
                        content_type="text/event-stream",
                        payload=payload,
                        final=final,
                    )
                )
                sequence += 1

            return await _render_stream(request, envelope, on_chunk=capture_chunk)
        payload = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        self._capture_wire(
            ResponsesWireRecord(
                session_id=session_id,
                exchange_id=exchange_id,
                direction="response",
                sequence=0,
                content_type="application/json",
                payload=payload,
                final=True,
            )
        )
        return web.Response(body=payload, content_type="application/json")

    def _capture_wire(self, record: ResponsesWireRecord) -> None:
        callback = self.wire_capture_callback
        if callback is not None:
            callback(record)


def _pack_reasoning(text: str) -> str:
    """Create an opaque token that can restore local-model reasoning on replay.

    This is transport encoding, not a security boundary.  The local adapter is
    both producer and consumer; the prefix prevents it from decoding tokens
    produced by another Responses provider.
    """

    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    return f"{_REASONING_PREFIX}{encoded}"


def _unpack_reasoning(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith(_REASONING_PREFIX):
        return ""
    try:
        return base64.urlsafe_b64decode(value[len(_REASONING_PREFIX) :].encode("ascii")).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""


def _reasoning_text(item: dict[str, Any]) -> str:
    for key in ("content", "summary"):
        value = item.get(key)
        if not isinstance(value, list):
            continue
        chunks = [
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("text")
        ]
        if chunks:
            return "\n".join(chunks)
    return _unpack_reasoning(item.get("encrypted_content"))


def _content_text(value: Any) -> str:
    if isinstance(value, dict):
        return flatten_content([value])
    return flatten_content(value)


def _normalize_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one leading system message without reordering later history.

    Responses clients may send both top-level ``instructions`` and developer
    items, including developer items in the middle of replayed history. Many
    model chat templates accept only one leading system message. Consecutive
    leading system messages are therefore merged, while later system messages
    stay at their original position as explicit user-visible reminders.
    """

    leading_system: list[str] = []
    first_non_system = 0
    while first_non_system < len(messages):
        message = messages[first_non_system]
        if message.get("role") != "system":
            break
        content = flatten_content(message.get("content"))
        if content:
            leading_system.append(content)
        first_non_system += 1

    normalized: list[dict[str, Any]] = []
    if leading_system:
        normalized.append(
            {"role": "system", "content": "\n\n".join(leading_system)}
        )
    for message in messages[first_non_system:]:
        if message.get("role") != "system":
            normalized.append(message)
            continue
        normalized.append(
            {
                "role": "user",
                "content": (
                    _SYSTEM_REMINDER_PREFIX
                    + flatten_content(message.get("content"))
                    + _SYSTEM_REMINDER_SUFFIX
                ),
            }
        )
    return normalized


def _tool_output_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("output", "body", "content"):
            if key in value:
                return _tool_output_text(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        parts = [_content_text(part) for part in value]
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value)


def _local_shell_arguments(action: Any) -> dict[str, Any]:
    if isinstance(action, str):
        return {"cmd": action}
    if not isinstance(action, dict):
        return {}
    if isinstance(action.get("command"), str):
        return {"cmd": action["command"]}
    commands = action.get("commands")
    if isinstance(commands, list):
        values = [command for command in commands if isinstance(command, str)]
        return {"cmd": values[0]} if len(values) == 1 else {"commands": values}
    return {key: value for key, value in action.items() if key != "type"}


def _append_assistant(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Merge adjacent Responses output items back into one assistant turn."""

    if not messages or messages[-1].get("role") != "assistant":
        messages.append(message)
        return

    previous = messages[-1]
    content = message.get("content")
    if content:
        previous["content"] = f"{previous.get('content') or ''}{content}"

    reasoning = message.get("reasoning_content")
    if reasoning:
        prior = previous.get("reasoning_content")
        previous["reasoning_content"] = f"{prior}\n{reasoning}" if prior else reasoning

    calls = message.get("tool_calls") or []
    if calls:
        previous["tool_calls"] = [*(previous.get("tool_calls") or []), *calls]


def _flush_reasoning(messages: list[dict[str, Any]], pending: list[str]) -> None:
    if pending:
        _append_assistant(
            messages,
            {"role": "assistant", "content": "", "reasoning_content": "\n".join(pending)},
        )
        pending.clear()


def _input_items_to_messages(items: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "reasoning":
            reasoning = _reasoning_text(item)
            if reasoning:
                pending_reasoning.append(reasoning)
            continue

        if item_type == "message" or (not item_type and "role" in item):
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            message: dict[str, Any] = {"role": role, "content": _content_text(item.get("content"))}
            if role == "assistant":
                if pending_reasoning:
                    message["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning.clear()
                _append_assistant(messages, message)
            else:
                _flush_reasoning(messages, pending_reasoning)
                messages.append(message)
            continue

        if item_type == "function_call":
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "tool",
                            "arguments": _arguments_as_dict(item.get("arguments")),
                        },
                    }
                ],
            }
            if pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            _append_assistant(messages, message)
            continue

        if item_type in {"local_shell_call", "shell_call"}:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "shell", "arguments": _local_shell_arguments(item.get("action"))},
                    }
                ],
            }
            if pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            _append_assistant(messages, message)
            continue

        if item_type in {"function_call_output", "local_shell_call_output", "shell_call_output"}:
            _flush_reasoning(messages, pending_reasoning)
            messages.append({"role": "tool", "content": _tool_output_text(item.get("output"))})
            continue

        if item_type in {"input_text", "input_image"}:
            _flush_reasoning(messages, pending_reasoning)
            messages.append({"role": "user", "content": _content_text(item)})

    _flush_reasoning(messages, pending_reasoning)
    return messages


def _manager_message(parsed) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        message["reasoning_content"] = parsed.reasoning

    calls: list[dict[str, Any]] = []
    for call in parsed.tool_uses:
        arguments = call.get("input")
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            {
                "type": "function",
                "function": {"name": call.get("name") or "tool", "arguments": arguments},
            }
        )
    if calls:
        message["tool_calls"] = calls
    return message


def _output_items(parsed, raw_finish: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if parsed.reasoning:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{secrets.token_hex(12)}",
                "summary": [{"type": "summary_text", "text": parsed.reasoning}],
                "content": [{"type": "reasoning_text", "text": parsed.reasoning}],
                "encrypted_content": _pack_reasoning(parsed.reasoning),
                "status": "completed",
            }
        )

    text = parsed.text or ""
    if raw_finish == "length" and not text.strip() and not parsed.tool_uses:
        text = _CONTEXT_CUTOFF_TEXT
    if text:
        items.append(
            {
                "type": "message",
                "id": f"msg_{secrets.token_hex(12)}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    for call in parsed.tool_uses:
        arguments = call.get("input")
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{secrets.token_hex(12)}",
                "call_id": f"call_{secrets.token_hex(12)}",
                "name": call.get("name") or "tool",
                "arguments": json.dumps(
                    arguments if isinstance(arguments, dict) else {}, ensure_ascii=False, sort_keys=True
                ),
                "status": "completed",
            }
        )
    return items


def _response_envelope(body: dict, items: list[dict], in_tok: int, out_tok: int) -> dict[str, Any]:
    return {
        "id": f"resp_{secrets.token_hex(16)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model", "slime-actor"),
        "output": items,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


async def _render_stream(
    request: web.Request,
    envelope: dict[str, Any],
    *,
    on_chunk: Callable[[bytes, bool], None] | None = None,
) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)
    sequence_number = 0

    async def send(event: dict[str, Any], *, final: bool = False) -> None:
        nonlocal sequence_number
        event["sequence_number"] = sequence_number
        sequence_number += 1
        payload = json.dumps(event, ensure_ascii=False)
        encoded = f"event: {event['type']}\ndata: {payload}\n\n".encode()
        if on_chunk is not None:
            on_chunk(encoded, final)
        await response.write(encoded)

    started = {**envelope, "status": "in_progress", "output": []}
    await send({"type": "response.created", "response": started})

    for output_index, item in enumerate(envelope["output"]):
        item_id = item["id"]
        if item["type"] == "reasoning":
            added_item = {
                "type": "reasoning",
                "id": item_id,
                "summary": [],
                "content": [],
                "status": "in_progress",
            }
        elif item["type"] == "message":
            added_item = {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
        else:
            added_item = {**item, "arguments": "", "status": "in_progress"}
        await send({"type": "response.output_item.added", "output_index": output_index, "item": added_item})

        if item["type"] == "reasoning":
            text = item["summary"][0]["text"] if item.get("summary") else ""
            base = {"item_id": item_id, "output_index": output_index, "summary_index": 0}
            await send(
                {
                    "type": "response.reasoning_summary_part.added",
                    **base,
                    "part": {"type": "summary_text", "text": ""},
                }
            )
            await send({"type": "response.reasoning_summary_text.delta", **base, "delta": text})
            await send({"type": "response.reasoning_summary_text.done", **base, "text": text})
            await send(
                {
                    "type": "response.reasoning_summary_part.done",
                    **base,
                    "part": {"type": "summary_text", "text": text},
                }
            )
        elif item["type"] == "message":
            text = item["content"][0]["text"] if item.get("content") else ""
            base = {"item_id": item_id, "output_index": output_index, "content_index": 0}
            await send(
                {
                    "type": "response.content_part.added",
                    **base,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            )
            await send({"type": "response.output_text.delta", **base, "delta": text})
            await send({"type": "response.output_text.done", **base, "text": text})
            await send(
                {
                    "type": "response.content_part.done",
                    **base,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                }
            )
        elif item["type"] == "function_call":
            arguments = item.get("arguments", "{}")
            base = {"item_id": item_id, "output_index": output_index}
            await send({"type": "response.function_call_arguments.delta", **base, "delta": arguments})
            await send(
                {
                    "type": "response.function_call_arguments.done",
                    **base,
                    "name": item.get("name") or "tool",
                    "arguments": arguments,
                }
            )

        await send({"type": "response.output_item.done", "output_index": output_index, "item": item})

    await send({"type": "response.completed", "response": envelope}, final=True)
    await response.write_eof()
    return response
