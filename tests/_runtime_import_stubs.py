"""Minimal import-only stubs for CPU contract tests.

The tested rollout conversion code does not instantiate SGLang or Megatron.
These stubs let a lightweight CPU environment import the owning modules while
keeping all runtime behavior fail-fast if a test accidentally tries to use it.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace


class _UnavailableRuntime:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("runtime-only dependency was replaced by an import stub")


def install_megatron_import_stubs() -> None:
    if "megatron" in sys.modules or importlib.util.find_spec("megatron") is not None:
        return
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.__path__ = []
    core.mpu = SimpleNamespace()
    packed = ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = _UnavailableRuntime
    megatron.core = core
    sys.modules.update(
        {
            "megatron": megatron,
            "megatron.core": core,
            "megatron.core.packed_seq_params": packed,
        }
    )


def install_sglang_import_stubs() -> None:
    if "sglang" in sys.modules or importlib.util.find_spec("sglang") is not None:
        return
    sglang = ModuleType("sglang")
    sglang.__path__ = []
    srt = ModuleType("sglang.srt")
    srt.__path__ = []
    constants = ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang.srt = srt
    sys.modules.update(
        {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.constants": constants,
        }
    )

    external = ModuleType("slime.backends.sglang_utils.external")
    external.start_external_rollout_servers = _UnavailableRuntime
    config = ModuleType("slime.backends.sglang_utils.sglang_config")
    config.ModelConfig = _UnavailableRuntime
    config.ServerGroupConfig = _UnavailableRuntime
    config.SglangConfig = _UnavailableRuntime
    engine = ModuleType("slime.backends.sglang_utils.sglang_engine")
    engine.SGLangEngine = _UnavailableRuntime
    sys.modules.update(
        {
            "slime.backends.sglang_utils.external": external,
            "slime.backends.sglang_utils.sglang_config": config,
            "slime.backends.sglang_utils.sglang_engine": engine,
        }
    )
