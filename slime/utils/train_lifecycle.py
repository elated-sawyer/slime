"""Typed driver-side training lifecycle contracts.

The optimizer executes on every trainer worker, but lifecycle participants run
exactly once in the control process after worker outcomes have been checked for
consensus.  This module is task-neutral and contains no registry semantics.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from slime.utils.misc import load_function


class OptimizerStepStatus(str, Enum):  # noqa: UP042 - Slime still targets Python 3.10
    COMMITTED = "committed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class OptimizerStepOutcome:
    rollout_id: int
    step_id: int
    status: OptimizerStepStatus
    global_batch_size: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.rollout_id < 0 or self.step_id < 0:
            raise ValueError("rollout_id and step_id must be non-negative")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.status is OptimizerStepStatus.COMMITTED and self.reason is not None:
            raise ValueError("committed optimizer step cannot have a skip reason")
        if self.status is OptimizerStepStatus.SKIPPED and not self.reason:
            raise ValueError("skipped optimizer step requires a reason")

    @property
    def step_key(self) -> tuple[int, int]:
        """Stable identity of the actual optimizer invocation within a run epoch."""

        return self.rollout_id, self.step_id


@dataclass(frozen=True)
class TrainBatchOutcome:
    rollout_id: int
    optimizer_steps: tuple[OptimizerStepOutcome, ...]

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("rollout_id must be non-negative")
        if not self.optimizer_steps:
            raise ValueError("TrainBatchOutcome requires at least one optimizer-step outcome")
        if any(item.rollout_id != self.rollout_id for item in self.optimizer_steps):
            raise ValueError("optimizer-step outcome belongs to another rollout")
        step_ids = tuple(item.step_id for item in self.optimizer_steps)
        if step_ids != tuple(range(len(step_ids))):
            raise ValueError("optimizer step_ids must be contiguous from zero")

    @property
    def committed_steps(self) -> tuple[OptimizerStepOutcome, ...]:
        return tuple(item for item in self.optimizer_steps if item.status is OptimizerStepStatus.COMMITTED)

    @property
    def skipped_steps(self) -> tuple[OptimizerStepOutcome, ...]:
        return tuple(item for item in self.optimizer_steps if item.status is OptimizerStepStatus.SKIPPED)

    @property
    def actor_updated(self) -> bool:
        return bool(self.committed_steps)


@dataclass(frozen=True)
class RolloutSkipReason:
    reason_code: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason_code:
            raise ValueError("reason_code must be non-empty")


@dataclass(frozen=True)
class SkippedTrainBatch:
    rollout_id: int
    reason: RolloutSkipReason

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("rollout_id must be non-negative")


@dataclass(frozen=True)
class TrainBatchFailure:
    rollout_id: int
    error_type: str
    message: str


@dataclass(frozen=True)
class RunRestoreContext:
    start_rollout_id: int


@dataclass(frozen=True)
class CheckpointContext:
    rollout_id: int
    force_sync: bool


@dataclass(frozen=True)
class CheckpointParticipantState:
    participant_id: str
    state_ref: str
    state_digest: str


class TrainerLifecycleParticipant(Protocol):
    """Single driver-level participant; checkpoint methods are wired in M5."""

    def on_run_restored(self, context: RunRestoreContext) -> None: ...

    def on_batch_skipped(self, outcome: SkippedTrainBatch) -> None: ...

    def on_optimizer_committed(self, outcome: TrainBatchOutcome) -> None: ...

    def on_batch_failed(self, failure: TrainBatchFailure) -> None: ...

    def prepare_checkpoint(self, context: CheckpointContext) -> CheckpointParticipantState | None: ...

    def on_checkpoint_committed(self, context: CheckpointContext) -> None: ...

    def on_checkpoint_aborted(self, context: CheckpointContext) -> None: ...

    def on_run_finished(self) -> None: ...


class _NoopTrainerLifecycle:
    def on_run_restored(self, context: RunRestoreContext) -> None:
        del context

    def on_batch_skipped(self, outcome: SkippedTrainBatch) -> None:
        del outcome

    def on_optimizer_committed(self, outcome: TrainBatchOutcome) -> None:
        del outcome

    def on_batch_failed(self, failure: TrainBatchFailure) -> None:
        del failure

    def prepare_checkpoint(self, context: CheckpointContext) -> CheckpointParticipantState | None:
        del context
        return None

    def on_checkpoint_committed(self, context: CheckpointContext) -> None:
        del context

    def on_checkpoint_aborted(self, context: CheckpointContext) -> None:
        del context

    def on_run_finished(self) -> None:
        return None


class TrainerLifecycleController:
    """Own synchronous invocation and driver-level exactly-once placement."""

    _REQUIRED_METHODS = (
        "on_run_restored",
        "on_batch_skipped",
        "on_optimizer_committed",
        "on_batch_failed",
        "prepare_checkpoint",
        "on_checkpoint_committed",
        "on_checkpoint_aborted",
        "on_run_finished",
    )

    def __init__(self, participant: TrainerLifecycleParticipant | None = None) -> None:
        self.participant = participant or _NoopTrainerLifecycle()
        missing = [name for name in self._REQUIRED_METHODS if not callable(getattr(self.participant, name, None))]
        if missing:
            raise TypeError(f"trainer lifecycle participant is missing methods: {missing}")

    @classmethod
    def from_args(cls, args) -> TrainerLifecycleController:
        path = getattr(args, "custom_train_lifecycle_factory_path", None)
        if path is None:
            return cls()
        factory = load_function(path)
        participant = factory(args)
        if inspect.isawaitable(participant):
            raise TypeError("trainer lifecycle factory must be synchronous")
        return cls(participant)

    def run_restored(self, start_rollout_id: int) -> None:
        self._call("on_run_restored", RunRestoreContext(start_rollout_id=start_rollout_id))

    def batch_skipped(self, outcome: SkippedTrainBatch) -> None:
        self._call("on_batch_skipped", outcome)

    def optimizer_outcome(self, outcome: TrainBatchOutcome) -> None:
        if outcome.actor_updated:
            self._call("on_optimizer_committed", outcome)
            return
        reasons = sorted({item.reason for item in outcome.skipped_steps if item.reason})
        self.batch_skipped(
            SkippedTrainBatch(
                rollout_id=outcome.rollout_id,
                reason=RolloutSkipReason(
                    reason_code="optimizer_steps_skipped",
                    message=", ".join(reasons),
                    metadata={"optimizer_step_count": len(outcome.optimizer_steps)},
                ),
            )
        )

    def batch_failed(self, rollout_id: int, exc: BaseException) -> None:
        self._call(
            "on_batch_failed",
            TrainBatchFailure(
                rollout_id=rollout_id,
                error_type=type(exc).__name__,
                message=str(exc),
            ),
        )

    def run_finished(self) -> None:
        self._call("on_run_finished")

    def _call(self, method_name: str, *args) -> Any:
        result = getattr(self.participant, method_name)(*args)
        if inspect.isawaitable(result):
            raise TypeError(f"trainer lifecycle method {method_name} must be synchronous")
        return result


def consolidate_train_batch_outcomes(outcomes: list[TrainBatchOutcome]) -> TrainBatchOutcome:
    """Require every distributed trainer worker to report the same outcome."""

    if not outcomes:
        raise ValueError("trainer returned no worker outcomes")
    if not all(isinstance(item, TrainBatchOutcome) for item in outcomes):
        raise TypeError("actor workers must return TrainBatchOutcome")
    first = outcomes[0]
    mismatched = [rank for rank, item in enumerate(outcomes[1:], start=1) if item != first]
    if mismatched:
        raise RuntimeError(f"trainer workers disagree on optimizer outcome; mismatched ranks={mismatched}")
    return first
