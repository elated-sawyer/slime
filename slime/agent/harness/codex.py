"""Codex harness.

Two non-obvious bits: the provider base_url must be inline in the TOML (Codex
only honours env vars for the default OpenAI provider), and the config is written
via a base64 round-trip to dodge shell-quoting traps.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class CodexHarness(BaseHarness):
    name = "codex"

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_CODEX_TARBALL"
    extra_args_env = "SLIME_AGENT_CODEX_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CODEX_EXTRA_ENVS"
    wire_api_env = "SLIME_AGENT_CODEX_WIRE_API"

    # static flags after ``codex exec``; --skip-git-repo-check lets it run in
    # workdirs whose git check is brittle (e.g. shallow clones)
    exec_flags = "--skip-git-repo-check"

    # config.toml written into the sandbox. base_url MUST be inline here (Codex
    # only honours env vars for the default OpenAI provider). {model} / {base_url}
    # are filled per run in write_config; the rest is fixed wiring.
    config_toml = (
        'model = "{model}"\n'
        'model_provider = "slime"\n'
        "{runtime_config}"
        "\n"
        "[model_providers.slime]\n"
        'name = "slime"\n'
        'base_url = "{base_url}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "{wire_api}"\n'
    )

    @classmethod
    def wire_api(cls) -> str:
        value = os.environ.get(cls.wire_api_env, "chat").strip().lower()
        if value not in {"chat", "responses"}:
            raise ValueError(f"{cls.wire_api_env} must be 'chat' or 'responses', got {value!r}")
        return value

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="codex --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        wire_api = self.wire_api()
        runtime_config = ""
        if wire_api == "responses":
            runtime_config = (
                'approval_policy = "never"\n'
                'sandbox_mode = "danger-full-access"\n'
            )
        if ctx.model_context_window is not None:
            if ctx.model_context_window <= 0:
                raise ValueError("model_context_window must be positive")
            runtime_config = (
                f"model_context_window = {ctx.model_context_window}\n"
                f"model_auto_compact_token_limit = {ctx.model_context_window}\n"
                f"{runtime_config}"
            )
        toml = self.config_toml.format(
            model=ctx.model_label,
            base_url=f"{ctx.adapter_url}/v1",
            wire_api=wire_api,
            runtime_config=runtime_config,
        )
        if wire_api == "responses":
            toml += '\n[history]\npersistence = "none"\n'
        toml_b64 = base64.b64encode(toml.encode("utf-8")).decode("ascii")
        config_dir = f"{ctx.home_dir}/.codex"
        config_path = f"{config_dir}/config.toml"
        await sb.exec(
            f"mkdir -p {shlex.quote(config_dir)} && "
            # base64 round-trip avoids any single-quote / heredoc shell-quoting trap
            f"echo {shlex.quote(toml_b64)} | base64 -d > {shlex.quote(config_path)} && "
            f"chown -R {shlex.quote(f'{ctx.execution_user}:{ctx.execution_user}')} {shlex.quote(config_dir)}",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        # ``codex exec`` is the non-interactive entrypoint
        cmd = f"codex exec {self.exec_flags}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        cmd = f"{cmd} -- {shlex.quote(prompt)} </dev/null"
        env: dict[str, str] = {}
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            configured = json.loads(extra_envs)
            if not isinstance(configured, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in configured.items()
            ):
                raise ValueError(f"{self.extra_envs_env} must be a JSON object of string values")
            env.update(configured)
        env.update(ctx.extra_env)
        env.update(
            {
                "HOME": ctx.home_dir,
                "CODEX_HOME": f"{ctx.home_dir}/.codex",
                # Codex propagates OPENAI_API_KEY into Authorization: Bearer;
                # the Slime adapter resolves the sid from that header.
                "OPENAI_API_KEY": ctx.session_id,
                "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
            }
        )
        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env=env,
            time_budget_sec=time_budget_sec,
            user=ctx.execution_user,
            home_dir=ctx.home_dir,
        )
