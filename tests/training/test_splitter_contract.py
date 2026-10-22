"""Custom splitter ownership and lazy workflow execution."""

from collections.abc import Iterator
from weakref import ref

import mlflow
import pytest

from ml_toolkit.config import TrainingConfig, TrackingConfig
from ml_toolkit.data import Pool
from ml_toolkit.training import Split, Splitter, TimeSeriesCV, train_model
from tests.support.training import MeanModel, mae, make_identified_data, setup_mlflow
from tests.support.evaluators import CallbackEvaluator


def test_custom_splitter_controls_final_rows_and_execution_order(tmp_path):
    setup_mlflow(tmp_path)
    data = make_identified_data()
    events, fits, reports = [], [], []
    pools = []

    def make_split(train, heldout, index):
        part = Split(data.take(train), data.take(heldout), index=index)
        pools.extend((ref(part.train), ref(part.heldout)))
        return part

    class Model(MeanModel):
        def fit(self, train, validation=None):
            ids = train.object_id.to_list()
            assert max(ids) < 20
            fits.append(
                (ids, None if validation is None else validation.object_id.to_list())
            )
            self._iteration = len(train) - 1
            return super().fit(train, validation)

        @property
        def best_iteration(self):
            return self._iteration

    class ProjectSplitter(Splitter[Pool]):
        def get_validation(self) -> Iterator[Split[Pool]]:
            events.append("first")
            yield make_split(range(8), range(8, 12), 7)
            # The workflow must finish the first fit/report before requesting more data.
            assert len(fits) == len(reports) == 1
            assert all(pool() is None for pool in pools)
            events.append("second")
            yield make_split(range(12), range(12, 16), 2)
            assert len(fits) == len(reports) == 2
            assert all(pool() is None for pool in pools)
            events.append("exhausted")

        def get_final(self) -> Split[Pool]:
            assert events == ["first", "second", "exhausted"]
            events.append("final")
            # Neither the last fold's train nor all non-test rows.
            return Split(self.data.take(range(4, 18)), self.data.take(range(20, 24)))

        def parameters(self) -> dict[str, object]:
            return {"policy": "project"}

    def evaluate(**inputs):
        parts = inputs["split"].unique(maintain_order=True).to_list()
        reports.append(parts)
        if parts[-1] == "val":
            assert inputs["data"].object_id.max() < 20
        return {}

    result = train_model(
        splitter=ProjectSplitter(data),
        params={"iterations": 99},
        model_factory=Model,
        metric=mae,
        run_name="custom-splitter",
        config=TrainingConfig(tracking=TrackingConfig(log_final_model=False)),
        prediction_evaluators=CallbackEvaluator(name="rows", evaluate=evaluate),
    )
    assert events == ["first", "second", "exhausted", "final"]
    assert fits == [
        (list(range(8)), list(range(8, 12))),
        (list(range(12)), list(range(12, 16))),
        (list(range(4, 18)), None),
    ]
    assert [part.index for part in result.validation.splits] == [7, 2]
    assert result.params["iterations"] == 12
    assert reports == [["train", "val"], ["train", "val"], ["train", "test"]]
    assert result.test_metrics is not None
    parent = next(
        run
        for run in mlflow.search_runs(output_format="list")
        if run.data.tags["mlflow.runName"] == "custom-splitter"
    )
    settings = mlflow.artifacts.load_dict(
        f"runs:/{parent.info.run_id}/configuration/splitter.json"
    )
    assert settings["parameters"] == {"policy": "project"}
    assert settings["class"].endswith(".ProjectSplitter")
    config = mlflow.artifacts.load_dict(
        f"runs:/{parent.info.run_id}/configuration/requested.json"
    )
    assert set(config) == {"tuning", "reporting", "tracking"}


@pytest.mark.parametrize(
    "case,expected_fits",
    [
        ("empty", 0),
        ("missing", 0),
        ("none", 0),
        ("zero", 0),
        ("negative", 0),
        ("duplicate", 1),
    ],
)
def test_invalid_validation_stops_before_affected_fit(tmp_path, case, expected_fits):
    setup_mlflow(tmp_path)
    fits = []

    class InvalidSplitter(Splitter[Pool]):
        def get_validation(self):
            if case == "empty":
                return
            train = self.data.take(range(8))
            heldout = self.data.take(range(8, 12))
            index = {"none": None, "zero": 0, "negative": -1}.get(case, 1)
            yield Split(train, None if case == "missing" else heldout, index=index)
            if case == "duplicate":
                yield Split(train, heldout, index=1)

        def get_final(self):
            pytest.fail("Final data requested after invalid validation")

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(1)
            return super().fit(train, validation)

    with pytest.raises(ValueError, match="validation split|Validation split"):
        train_model(
            splitter=InvalidSplitter(make_identified_data()),
            params={},
            model_factory=Model,
            metric=mae,
            run_name="invalid-split",
            config=TrainingConfig(tracking=TrackingConfig(log_final_model=False)),
        )
    assert len(fits) == expected_fits


def test_final_schema_checked_before_final_fit(tmp_path):
    setup_mlflow(tmp_path)
    fits = []

    class Custom(Splitter[Pool]):
        def get_validation(self):
            yield Split(self.data.take(range(8)), self.data.take(range(8, 12)), index=1)

        def get_final(self):
            return Split(self.data.select_features(["x"]))

    class Model(MeanModel):
        def fit(self, train, validation=None):
            fits.append(1)
            return super().fit(train, validation)

    with pytest.raises(ValueError, match="missing"):
        train_model(
            splitter=Custom(make_identified_data()),
            params={},
            model_factory=Model,
            metric=mae,
            run_name="missing-final-feature",
            config=TrainingConfig(tracking=TrackingConfig(log_final_model=False)),
        )
    assert len(fits) == 1


def test_builtin_cv_constructs_only_requested_pools(monkeypatch):
    data = make_identified_data()
    calls = []
    take = type(data).take

    def capture(self, indices):
        calls.append(1)
        return take(self, indices)

    monkeypatch.setattr(type(data), "take", capture)
    splitter = TimeSeriesCV(data, n_splits=3, skip_splits=1, test_size=4)
    iterator = splitter.get_validation()
    assert calls == []
    assert next(iterator).index == 2
    assert len(calls) == 2
    assert next(iterator).index == 3
    assert len(calls) == 4
    with pytest.raises(StopIteration):
        next(iterator)
