from __future__ import annotations

from types import SimpleNamespace

import torch

from _runtime_import_stubs import install_megatron_import_stubs, install_sglang_import_stubs


install_megatron_import_stubs()
install_sglang_import_stubs()

from slime.backends.megatron_utils import data as megatron_data
from slime.backends.megatron_utils.data import DataIterator
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample


NUM_GPUS = 0


def _manager():
    real_class = getattr(RolloutManager, "__ray_actor_class__", RolloutManager)
    manager = real_class.__new__(real_class)
    manager.args = SimpleNamespace(
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        rollout_top_p=1.0,
        use_rollout_routing_replay=False,
        global_batch_size=2,
        rollout_data_transport="object-store",
    )
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.train_parallel_config = {"dp_size": 2}
    return manager


def _sample(index, rollout_id, metadata):
    return Sample(
        group_index=0,
        index=index,
        rollout_id=rollout_id,
        tokens=[10, 11, 12],
        response_length=2,
        reward=float(index),
        loss_mask=[1, 1],
        status=Sample.Status.COMPLETED,
        train_metadata=metadata,
    )


def test_train_metadata_survives_conversion_dp_split_and_data_iterator(monkeypatch):
    manager = _manager()
    samples = [
        _sample(0, 100, {"attempt_id": 100, "segment_id": "a"}),
        _sample(1, 101, None),
        _sample(2, 100, {"attempt_id": 100, "segment_id": "b"}),
    ]
    train_data = manager._convert_samples_to_train_data(samples)
    assert train_data["metadata"] == [samples[0].train_metadata, None, samples[2].train_metadata]

    monkeypatch.setattr(
        "slime.ray.rollout.build_dp_schedule",
        lambda *args, **kwargs: (
            [[0, 2], [1]],
            [[[0, 1]], [[0]]],
            [1],
            [2],
        ),
    )
    monkeypatch.setattr("slime.ray.rollout.ray.put", lambda value, **kwargs: value)
    refs = manager._split_train_data_by_dp(train_data)

    rank0 = refs[0].inner
    rank1 = refs[1].inner
    assert rank0["metadata"] == [samples[0].train_metadata, samples[2].train_metadata]
    assert rank1["metadata"] == [None]
    iterator = DataIterator(rank0, rank0["micro_batch_indices"])
    assert iterator.get_next(["metadata"])["metadata"] == [
        samples[0].train_metadata,
        samples[2].train_metadata,
    ]


def test_no_train_metadata_keeps_legacy_batch_shape():
    manager = _manager()
    train_data = manager._convert_samples_to_train_data([_sample(0, 100, None), _sample(1, 101, None)])
    assert "metadata" not in train_data


def test_rollout_logger_ignores_structured_train_metadata(monkeypatch):
    monkeypatch.setattr(megatron_data.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(megatron_data.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(megatron_data.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        megatron_data.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 1,
        raising=False,
    )

    captured = {}

    def capture_log_data(metric_name, args, rollout_id, log_dict):
        del args
        captured.update(
            metric_name=metric_name,
            rollout_id=rollout_id,
            log_dict=log_dict,
        )
        return None

    monkeypatch.setattr(megatron_data, "gather_log_data", capture_log_data)
    megatron_data.log_rollout_data(
        7,
        SimpleNamespace(
            ci_test=False,
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        {
            "response_lengths": [2],
            "loss_masks": [torch.tensor([1, 1])],
            "total_lengths": [3],
            "global_batch_sizes": [1],
            "metadata": [{"attempt_id": 100, "segment_id": "a"}],
        },
    )

    assert captured["metric_name"] == "rollout"
    assert captured["rollout_id"] == 7
    assert "metadata" not in captured["log_dict"]
