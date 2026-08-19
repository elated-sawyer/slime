from __future__ import annotations

import ast
from pathlib import Path

import pytest


NUM_GPUS = 0
SLIME_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("driver", ("train.py", "train_async.py"))
def test_sync_and_async_drivers_expose_the_same_lifecycle_events(driver):
    tree = ast.parse((SLIME_ROOT / driver).read_text(encoding="utf-8"))
    lifecycle_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "lifecycle"
    }
    assert {
        "run_restored",
        "batch_skipped",
        "optimizer_outcome",
        "batch_failed",
        "run_finished",
    } <= lifecycle_calls
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "consolidate_train_batch_outcomes" in called_names
