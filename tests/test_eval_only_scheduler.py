from types import SimpleNamespace

from slime.backends.megatron_utils import model as model_module


def test_eval_only_uses_nonzero_placeholder_scheduler(monkeypatch):
    captured = {}

    class FakeOptimizerParamScheduler:
        def __init__(self, optimizer, **kwargs):
            captured["optimizer"] = optimizer
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "OptimizerParamScheduler", FakeOptimizerParamScheduler)

    args = SimpleNamespace(
        num_rollout=0,
        eval_interval=1,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    optimizer = object()

    scheduler = model_module.get_optimizer_param_scheduler(args, optimizer)

    assert isinstance(scheduler, FakeOptimizerParamScheduler)
    assert captured["optimizer"] is optimizer
    assert args.train_iters == 0
    assert args.lr_decay_iters == 1
    assert captured["lr_decay_steps"] == 1
    assert captured["wd_incr_steps"] == 1
