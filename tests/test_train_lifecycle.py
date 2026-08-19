from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime.utils.train_lifecycle import (
    OptimizerStepOutcome,
    OptimizerStepStatus,
    RolloutSkipReason,
    SkippedTrainBatch,
    TrainBatchOutcome,
    TrainerLifecycleController,
    consolidate_train_batch_outcomes,
)


NUM_GPUS = 0


def _outcome(*, rollout_id=3, committed=True):
    return TrainBatchOutcome(
        rollout_id=rollout_id,
        optimizer_steps=(
            OptimizerStepOutcome(
                rollout_id=rollout_id,
                step_id=0,
                status=OptimizerStepStatus.COMMITTED if committed else OptimizerStepStatus.SKIPPED,
                global_batch_size=8,
                reason=None if committed else "non_finite_gradient",
            ),
        ),
    )


def test_worker_outcome_consensus_preserves_actual_step_records():
    outcome = _outcome()
    assert consolidate_train_batch_outcomes([outcome, outcome]) == outcome
    assert outcome.committed_steps[0].step_key == (3, 0)


def test_worker_outcome_disagreement_fails_closed():
    with pytest.raises(RuntimeError, match="disagree"):
        consolidate_train_batch_outcomes([_outcome(rollout_id=3), _outcome(rollout_id=4)])


def test_controller_invokes_one_driver_side_event_per_batch():
    calls = []

    class Participant:
        def on_run_restored(self, context):
            calls.append(("restored", context.start_rollout_id))

        def on_batch_skipped(self, outcome):
            calls.append(("skipped", outcome.rollout_id))

        def on_optimizer_committed(self, outcome):
            calls.append(("committed", outcome.rollout_id))

        def on_batch_failed(self, failure):
            calls.append(("failed", failure.rollout_id))

        def prepare_checkpoint(self, context):
            return None

        def on_checkpoint_committed(self, context):
            pass

        def on_checkpoint_aborted(self, context):
            pass

        def on_run_finished(self):
            calls.append(("finished", None))

    controller = TrainerLifecycleController(Participant())
    controller.run_restored(2)
    controller.optimizer_outcome(_outcome(rollout_id=3))
    controller.optimizer_outcome(_outcome(rollout_id=4, committed=False))
    controller.batch_skipped(
        SkippedTrainBatch(rollout_id=5, reason=RolloutSkipReason(reason_code="empty"))
    )
    controller.run_finished()

    assert calls == [
        ("restored", 2),
        ("committed", 3),
        ("skipped", 4),
        ("skipped", 5),
        ("finished", None),
    ]


def test_lifecycle_factory_must_return_complete_synchronous_participant(monkeypatch):
    monkeypatch.setattr("slime.utils.train_lifecycle.load_function", lambda path: lambda args: object())
    with pytest.raises(TypeError, match="missing methods"):
        TrainerLifecycleController.from_args(
            SimpleNamespace(custom_train_lifecycle_factory_path="package.factory")
        )
