"""Temporal window sizes, gaps and rejected empty training windows."""


import pytest

from ml_toolkit.config import HoldoutConfig, TimeSeriesCVConfig, TrainingConfig
from ml_toolkit.training.splits import make_validation_splits, split_holdout


from tests.support.training import make_identified_data as identified_data


@pytest.mark.parametrize("size, count", [(3, 3), (0.2, 3), (0.01, 1), (1, 1)])
def test_tail_sizes_use_rows_or_ceiling_fraction(size, count):
    data = identified_data().take(range(11))
    split = split_holdout(data, HoldoutConfig(val_size=size))
    assert type(split.train) is type(data) is type(split.val)
    assert split.train.object_id.to_list() == list(range(11 - count))
    assert split.val.object_id.to_list() == list(range(11 - count, 11))


def test_holdout_gap():
    data = identified_data().take(range(11))
    split = split_holdout(data, HoldoutConfig(val_size=0.25, gap=2))
    assert split.train.object_id.to_list() == list(range(6))
    assert split.val.object_id.to_list() == [8, 9, 10]


def test_test_and_validation_fractions_use_their_current_input_and_round_up():
    data = identified_data().take(range(11))
    outer = split_holdout(data, HoldoutConfig(val_size=0.2))
    split = split_holdout(outer.train, HoldoutConfig(val_size=0.25, gap=1))
    assert (len(outer.train), len(outer.val)) == (8, 3)
    assert (len(split.train), len(split.val)) == (5, 2)


@pytest.mark.parametrize("size", [0, -1, 0.0, 1.0, float("nan"), float("inf")])
def test_invalid_sizes_fail_at_configuration(size):
    with pytest.raises(ValueError):
        TrainingConfig(validation=HoldoutConfig(), test_size=size)
    with pytest.raises(ValueError):
        HoldoutConfig(val_size=size)


def test_cv_fraction_window_gap_and_original_indices():
    splits = make_validation_splits(
        identified_data().take(range(20)),
        TimeSeriesCVConfig(
            n_splits=3,
            val_size=0.2,
            gap=2,
            max_train_size=5,
            skip_splits=1,
        ),
    )
    assert [split.index for split in splits] == [2, 3]
    for split in splits:
        assert len(split.train) == 5 and len(split.val) == 4
        assert split.val.object_id[0] - split.train.object_id[-1] == 3


@pytest.mark.parametrize(
    "config", [HoldoutConfig(val_size=5), HoldoutConfig(val_size=0.8, gap=2)]
)
def test_empty_train_is_rejected(config):
    with pytest.raises(ValueError, match="Not enough rows"):
        make_validation_splits(identified_data().take(range(5)), config)
