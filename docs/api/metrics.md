# Metrics

Metrics turn aligned targets, predictions, and optional weights into named
scores. Use them directly to evaluate existing predictions or pass a metric
callable to the training workflow.

## Evaluate binary probabilities

```python
import numpy as np

from ml_toolkit.metrics import metrics_classification

observed = np.array([0.0, 0.0, 1.0, 1.0])
probabilities = np.array([0.1, 0.6, 0.7, 0.9])
scores = metrics_classification(observed, probabilities, threshold=0.5)
print(scores.precision)  # 0.6666666666666666
print(scores.recall)     # 1.0
print(scores.to_dict())
```

Three rows exceed the cutoff and two of them are positive, so precision is
`2 / 3`. Both positives are captured, so recall is `1.0`. Changing the cutoff
changes threshold-based scores; ROC AUC and log loss use probabilities directly.

For Poisson frequency targets use `metrics_frequency`. Severity metrics also
need a Tweedie `power`, which you can bind in a callback. PU metrics evaluate
positive capture with `PUMetric`; unlabeled observations are not confirmed
negative examples.

## Supply a custom training metric

```python
from ml_toolkit.metrics import MetricScores
from ml_toolkit.types import FloatArray


def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    error = abs(observed - prediction)
    value = float((error * weight).sum() / weight.sum())
    return MetricScores({"mae": value})


print(weighted_mae(observed, probabilities, np.ones(4)).to_dict())
# {'mae': 0.275}
```

Training passes three aligned arrays and validates weights at its data boundary.
Return an object with `to_dict()`, such as `MetricScores`. Use the same score key
(`"mae"` here) in selection or tuning configuration when optimizing it.

## API reference

Import metrics and result types from `ml_toolkit.metrics`.

::: ml_toolkit.metrics
    options:
      members:
        - Metric
        - MetricResult
        - MetricScores
        - metrics_classification
        - MetricsClassification
        - confusion_matrix
        - metrics_frequency
        - MetricsFrequency
        - metrics_severity
        - MetricsSeverity
        - PUMetric
        - positive_capture_curve
        - ranking_metrics
        - RankingMetrics
