# Metrics

Metrics evaluate aligned targets, predictions, and weights. Use them directly
for existing predictions or pass a compatible callable to training. Numerical
`FloatArray` parameters mean one-dimensional NumPy `float64` arrays; use
`np.array([...], dtype=float)` when preparing inputs.

## Choose a metric

| Data and purpose | Entry point | Result |
| --- | --- | --- |
| Confirmed binary labels and probabilities | `metrics_classification` | `MetricsClassification` |
| Binary labels after applying a cutoff | `confusion_matrix` | `(TN, FP, FN, TP)` weighted counts |
| Poisson frequency rates with exposure | `metrics_frequency` | `MetricsFrequency` |
| Severity predictions with a Tweedie power | `metrics_severity` | `MetricsSeverity` |
| Ranking outcomes by exposure capacity | `ranking_metrics` | `RankingMetrics` curves and selected points |
| Known positive and unlabeled rows | `PUMetric`, `positive_capture_curve` | Named scores or two curve arrays |

Classification, frequency, severity, and PU score objects expose `to_dict()`.
`RankingMetrics` instead contains curves and attributes for inspection and
[ranking plots](plots.md#gains-and-lift); it is not itself a training metric
result with `to_dict()`.

## Evaluate binary probabilities

<!-- example: metrics-classification -->
```python
from pprint import pprint

import numpy as np

from ml_toolkit.metrics import confusion_matrix, metrics_classification

observed = np.array([0.0, 0.0, 1.0, 1.0])
probabilities = np.array([0.1, 0.6, 0.7, 0.9])
weight = np.ones(4)
scores = metrics_classification(observed, probabilities, threshold=0.5)
print(f"precision={scores.precision:.3f}, recall={scores.recall:.3f}")
pprint({name: round(value, 4) for name, value in scores.to_dict().items()}, sort_dicts=False)
labels = (probabilities >= 0.5).astype(float)
print(confusion_matrix(observed, labels, sample_weight=weight))
```

<!-- output: metrics-classification -->
```text
precision=0.667, recall=1.000
{'accuracy': 0.75,
 'precision': 0.6667,
 'recall': 1.0,
 'f1': 0.8,
 'roc_auc': 1.0,
 'average_precision': 1.0,
 'log_loss': 0.3709}
(1.0, 1.0, 0.0, 2.0)
```
<!-- /output -->

Three rows reach the cutoff and two of them are positive, so precision is
`2 / 3`. Both positives are captured, so recall is `1.0`. Changing the cutoff
changes threshold-based scores; ROC AUC, average precision, and log loss use
probabilities directly. `confusion_matrix` requires predicted labels instead
of probabilities. Pass finite, nonnegative weights with positive total for
weighted scores. ROC AUC is NaN when only one target class is present.

## Evaluate frequency and severity

Continue with the NumPy import above. Frequency targets and predictions are
rates; exposure supplies row weights and ranking capacity. Do not multiply the
rates by exposure before calling the metric.

<!-- example: metrics-frequency-severity -->
```python
from ml_toolkit.metrics import metrics_frequency, metrics_severity

frequency = metrics_frequency(
    np.array([0.0, 1.0, 2.0]),
    np.array([0.2, 1.0, 1.8]),
    np.ones(3),
)
pprint({name: round(value, 4) for name, value in frequency.to_dict().items()}, sort_dicts=False)
severity = metrics_severity(
    np.array([100.0, 200.0, 300.0]),
    np.array([110.0, 190.0, 300.0]),
    np.ones(3),
    power=2.0,
)
pprint({name: round(value, 4) for name, value in severity.to_dict().items()}, sort_dicts=False)
```

<!-- output: metrics-frequency-severity -->
```text
{'mean_poisson_deviance': 0.1405,
 'd2_poisson_score': 0.848,
 'mean_absolute_error': 0.1333,
 'root_mean_squared_error': 0.1633,
 'prediction_over_observed': 1.0,
 'lift_at_10': 2.0,
 'gain_at_10': 0.2,
 'normalized_gini': 1.0}
{'mean_tweedie_deviance': 0.0038,
 'd2_tweedie_score': 0.98,
 'mean_absolute_error': 6.6667,
 'root_mean_squared_error': 8.165,
 'prediction_over_observed': 1.0,
 'lift_at_10': 1.5,
 'gain_at_10': 0.15,
 'normalized_gini': 1.0}
```
<!-- /output -->

Frequency predictions must be strictly positive and targets nonnegative.
Severity inputs must satisfy the chosen Tweedie power; `power=2.0` uses Gamma
deviance and requires positive targets and predictions. Both metrics also
calculate ranking scores, so the exposure-weighted observed total must be
positive. Their `gain_at_10` and `lift_at_10` describe the highest-scored 10% of
exposure; normalized Gini is NaN when perfect-ranking Gini is close to zero.

Bind the extra severity argument before passing it to training:

<!-- example: metrics-severity-callback -->
```python
from functools import partial

severity_metric = partial(metrics_severity, power=2.0)
bound_scores = severity_metric(
    np.array([100.0, 200.0, 300.0]),
    np.array([110.0, 190.0, 300.0]),
    np.ones(3),
)
print(round(bound_scores.mean_tweedie_deviance, 4))
```

<!-- output: metrics-severity-callback -->
```text
0.0038
```
<!-- /output -->

Use `metrics_frequency` and `MetricsFrequency` in new code. The legacy
`metrics_frequncy` and `MetricsFrequncy` spellings remain available as aliases.

## Inspect gains, lift, and Gini

<!-- example: metrics-ranking -->
```python
from ml_toolkit.metrics import ranking_metrics

ranking = ranking_metrics(
    observed, probabilities, weight, selected_percents=(25.0, 50.0)
)
print(ranking.normalized_gini)
print([(point.percent, point.gain, point.lift) for point in ranking.selected_points])
print(ranking.cumulative_exposure)
print(ranking.cumulative_observed)
```

<!-- output: metrics-ranking -->
```text
1.0
[(25.0, 0.5, 2.0), (50.0, 1.0, 2.0)]
[0.   0.25 0.5  0.75 1.  ]
[0.  0.5 1.  1.  1. ]
```
<!-- /output -->

Each selected point exposes `percent`, `gain`, and `lift`. Gain is the fraction
of total exposure-weighted observed outcome captured; lift divides gain by the
exposure fraction. Results interpolate at the requested capacity percentages.
Rows are sorted by descending prediction; tied scores retain input order and
zero-exposure rows are omitted. The cumulative arrays include an initial zero.
These are supervised outcomes; use PU metrics for positive/unlabeled labels.

## Evaluate positive-unlabeled predictions

<!-- example: metrics-pu -->
```python
from ml_toolkit.metrics import PUMetric, positive_capture_curve

pu_labels = np.array([1.0, 0.0, 1.0, 0.0])
pu_probabilities = np.array([0.9, 0.8, 0.4, 0.2])
pu_metric = PUMetric(prior=0.1, loss="nnPU")
pu_scores = pu_metric(pu_labels, pu_probabilities)
print(pu_scores.to_dict()["known_positive_capture_at_10pct"])
capacity, capture = positive_capture_curve(pu_labels, pu_probabilities)
print(capacity.tolist())
print(capture.tolist())
```

<!-- output: metrics-pu -->
```text
0.5
[0.0, 0.0, 0.5, 0.5, 1.0]
[0.0, 0.5, 0.5, 1.0, 1.0]
```
<!-- /output -->

Here `1` means known positive and `0` means unlabeled. The prior is population
positive prevalence, not the labeled fraction; match it and the loss to the
model's configuration. Both groups need positive total weight. `nnPU` clips
estimated negative risk at zero when calculating total risk; `uPU` retains it.

The returned curve arrays contain fractions, whereas the corresponding plot
uses percentages. All tied scores enter together. Capacity cutoffs can
therefore flag more unlabeled rows than the requested percentage. Capture
estimates recall only if labeled positives represent the positive population.

## Supply a custom training metric

<!-- example: metrics-custom -->
```python
from ml_toolkit.metrics import Metric, MetricScores
from ml_toolkit.types import FloatArray


def weighted_mae(
    observed: FloatArray, prediction: FloatArray, weight: FloatArray
) -> MetricScores:
    error = abs(observed - prediction)
    value = float((error * weight).sum() / weight.sum())
    return MetricScores({"mae": value})


metric: Metric = weighted_mae
print(metric(observed, probabilities, weight).to_dict())
```

<!-- output: metrics-custom -->
```text
{'mae': 0.275}
```
<!-- /output -->

Training passes three aligned arrays and validates weights at its data
boundary. Return an object satisfying `MetricResult`, with `to_dict()` mapping
score names to floats; `MetricScores` is a simple implementation. Use the same
key (`"mae"` here) in selection or tuning configuration when optimizing it,
and choose the appropriate optimization direction.

## API reference

Import metrics and result types from `ml_toolkit.metrics`. References are
grouped by task; call interfaces stay beside their callable classes. Selected
ranking points are attributes of `RankingMetrics`, with their class available
from `ml_toolkit.metrics.ranking`.

### Classification

::: ml_toolkit.metrics.metrics_classification
    options:
      heading_level: 4

::: ml_toolkit.metrics.MetricsClassification
    options:
      heading_level: 4

::: ml_toolkit.metrics.confusion_matrix
    options:
      heading_level: 4

### Frequency

<a id="ml_toolkit.metrics.metrics_frequency"></a>

#### metrics_frequency

```text
metrics_frequency(
    y_target: FloatArray, y_pred: FloatArray, exposure: FloatArray
) -> MetricsFrequency
```

::: ml_toolkit.metrics.frequency.metrics_frequncy
    options:
      heading_level: 4
      show_root_heading: false
      show_root_toc_entry: false
      show_signature: false

<a id="ml_toolkit.metrics.MetricsFrequency"></a>

::: ml_toolkit.metrics.frequency.MetricsFrequncy
    options:
      heading_level: 4
      heading: MetricsFrequency
      toc_label: MetricsFrequency
      show_signature: false

### Severity

::: ml_toolkit.metrics.metrics_severity
    options:
      heading_level: 4

::: ml_toolkit.metrics.MetricsSeverity
    options:
      heading_level: 4

### Ranking

<a id="ml_toolkit.metrics.ranking_metrics"></a>

#### ranking_metrics

```text
ranking_metrics(
    y_target: FloatArray,
    y_pred: FloatArray,
    exposure: FloatArray,
    selected_percents: tuple[float, ...] = (10.0,),
) -> RankingMetrics
```

Calculate gains, lift, and Gini after ranking by descending prediction. Supply
aligned finite vectors: observed rates in `y_target`, ordering scores in
`y_pred`, and nonnegative row exposures with positive total in `exposure`.
The exposure-weighted observed total must be positive. `selected_percents`
contains capacity percentages in `(0, 100]`; an empty tuple requests only the
full curves and Gini values.

Returns `RankingMetrics`, with cumulative arrays and selected gain/lift points
in request order. Zero-exposure rows are omitted, tied scores retain input
order, and selected points use linear interpolation. Normalized Gini is NaN
when perfect-ranking Gini is close to zero. Invalid shapes, non-finite values,
negative exposures, nonpositive totals, or out-of-range percentages raise
`ValueError`. See [the ranking recipe](#inspect-gains-lift-and-gini) for an
executable example and the following result attributes for their meanings.

::: ml_toolkit.metrics.RankingMetrics
    options:
      heading_level: 4

::: ml_toolkit.metrics.ranking.SelectedRankingPoint
    options:
      heading_level: 4

### Positive-unlabeled

::: ml_toolkit.metrics.PUMetric
    options:
      heading_level: 4

::: ml_toolkit.metrics.PUMetric.__call__
    options:
      heading_level: 5

::: ml_toolkit.metrics.positive_capture_curve
    options:
      heading_level: 4

### Custom training metrics

::: ml_toolkit.metrics.Metric
    options:
      heading_level: 4

::: ml_toolkit.metrics.Metric.__call__
    options:
      heading_level: 5

::: ml_toolkit.metrics.MetricResult
    options:
      heading_level: 4

::: ml_toolkit.metrics.MetricScores
    options:
      heading_level: 4
