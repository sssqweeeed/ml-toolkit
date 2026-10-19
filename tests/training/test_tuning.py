"""Scalar optimization, parameter isolation and objective direction."""


import pytest
from tests.support.training import setup_mlflow

from ml_toolkit.config import TuningConfig
from ml_toolkit.training import tune_optuna


def test_optimizer_only_requires_a_scalar_callback(tmp_path):
    setup_mlflow(tmp_path)
    seen = []
    base = {"base": 7}

    def score(params):
        seen.append(params.copy())
        params["base"] = -1
        return 3.0

    result = tune_optuna(
        base_params=base,
        param_space=lambda trial: {"fixed": 42},
        score=score,
        config=TuningConfig(
            metric="callback-owned-name", direction="minimize", n_trials=2
        ),
    )
    assert seen == [{"base": 7, "fixed": 42}] * 2
    assert result.best_params == {"base": 7, "fixed": 42}
    assert result.best_score == 3.0 and result.best_trial == 0
    assert base == {"base": 7}


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_optimizer_rejects_nonfinite_objectives(tmp_path, score):
    setup_mlflow(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        tune_optuna(
            base_params={},
            param_space=lambda trial: {},
            score=lambda params: score,
            config=TuningConfig(metric="mae", direction="minimize", n_trials=1),
        )


@pytest.mark.parametrize(
    "direction, expected_score, expected_trial",
    [("minimize", 1.0, 1), ("maximize", 3.0, 0)],
)
def test_optimizer_honors_direction_and_returns_complete_winning_parameters(
    tmp_path, direction, expected_score, expected_trial
):
    setup_mlflow(tmp_path)
    result = tune_optuna(
        base_params={"retained": 7, "candidate": -1.0},
        param_space=lambda trial: {"candidate": (3.0, 1.0, 2.0)[trial.number]},
        score=lambda params: params["candidate"],
        config=TuningConfig(metric="score", direction=direction, n_trials=3),
    )
    assert result.best_score == expected_score
    assert result.best_trial == expected_trial
    assert result.best_params == {"retained": 7, "candidate": expected_score}
