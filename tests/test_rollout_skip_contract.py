from __future__ import annotations

import pytest

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.train_lifecycle import RolloutSkipReason
from slime.utils.types import Sample


NUM_GPUS = 0


def test_explicit_skip_requires_an_empty_sample_set():
    skipped = RolloutFnTrainOutput(
        samples=[],
        skip_reason=RolloutSkipReason(reason_code="no_trainable_attempts"),
    )
    assert skipped.skip_reason.reason_code == "no_trainable_attempts"

    with pytest.raises(ValueError, match="cannot also contain samples"):
        RolloutFnTrainOutput(
            samples=[[Sample(index=0)]],
            skip_reason=RolloutSkipReason(reason_code="invalid"),
        )
