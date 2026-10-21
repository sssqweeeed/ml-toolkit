# Metrics

Metrics evaluate aligned targets, predictions, and weights. The small arrays
below illustrate calculations, not model performance on a real dataset. Use them directly
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
import numpy as np

from ml_toolkit.metrics import confusion_matrix, metrics_classification

observed = np.array([0.0, 0.0, 1.0, 1.0])
probabilities = np.array([0.1, 0.6, 0.4, 0.9])
weight = np.array([1.0, 2.0, 1.0, 2.0])
for cutoff in (0.3, 0.7):
    scores = metrics_classification(
        observed, probabilities, threshold=cutoff, sample_weight=weight,
    )
    print(f"cutoff={cutoff}: precision={scores.precision:.3f}, recall={scores.recall:.3f}")
labels = (probabilities >= 0.7).astype(float)
print("Weighted (TN, FP, FN, TP):", confusion_matrix(observed, labels, sample_weight=weight))
```

<!-- output: metrics-classification -->
```text
cutoff=0.3: precision=0.600, recall=1.000
cutoff=0.7: precision=1.000, recall=0.667
Weighted (TN, FP, FN, TP): (3.0, 0.0, 1.0, 2.0)
```
<!-- /output -->

At cutoff 0.3, predicted positives have total weight 5 and positive weight 3,
so precision is `3 / 5`; all positive weight is captured. Raising the cutoff
to 0.7 removes the false positive but misses positive weight 1 out of 3.
Weights affect counts, precision, and recall. ROC AUC, average precision, and log loss use
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

# Annual claim rates and policy-year exposure, not claim counts.
frequency_observed = np.array([0.0, 1.0, 2.0])
frequency_prediction = np.array([0.2, 1.2, 1.5])
exposure = np.array([0.5, 1.0, 2.0])
frequency = metrics_frequency(frequency_observed, frequency_prediction, exposure)
print(f"Frequency MAE: {frequency.mean_absolute_error:.4f} claims per policy-year")
print(f"Poisson deviance: {frequency.mean_poisson_deviance:.4f}")
print(f"Predicted / observed claims: {frequency.prediction_over_observed:.4f}")

# Average claim amounts in dollars, weighted by claim counts.
severity_observed = np.array([100.0, 200.0, 300.0])
severity_prediction = np.array([120.0, 180.0, 240.0])
claim_count = np.array([1.0, 2.0, 3.0])
severity = metrics_severity(severity_observed, severity_prediction, claim_count, power=2.0)
print(f"Severity MAE: ${severity.mean_absolute_error:.2f} per claim")
print(f"Gamma deviance: {severity.mean_tweedie_deviance:.4f}")
print(f"Predicted / observed cost: {severity.prediction_over_observed:.4f}")
```

<!-- output: metrics-frequency-severity -->
```text
Frequency MAE: 0.3714 claims per policy-year
Poisson deviance: 0.1534
Predicted / observed claims: 0.8600
Severity MAE: $40.00 per claim
Gamma deviance: 0.0359
Predicted / observed cost: 0.8571
```
<!-- /output -->

The frequency totals are 4.3 predicted and 5.0 observed claims, hence 14%
underprediction. Weighting gives more influence to the final row's rate error.
Severity predicts $1,200 versus $1,400 observed, with $40 weighted absolute
error per claim. Gamma deviance is dimensionless. Export the complete named
score dictionary with `frequency.to_dict()` or `severity.to_dict()`.

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
bound_scores = severity_metric(severity_observed, severity_prediction, claim_count)
print(round(bound_scores.mean_tweedie_deviance, 4))
```

<!-- output: metrics-severity-callback -->
```text
0.0359
```
<!-- /output -->


## Inspect gains, lift, and Gini

<!-- example: metrics-ranking -->

```python
from ml_toolkit.metrics import ranking_metrics

ranking = ranking_metrics(
    np.array([1.0, 4.0, 2.0, 3.0]),
    np.array([0.1, 0.8, 0.9, 0.5]),
    np.array([1.0, 2.0, 1.0, 2.0]),
    selected_percents=(25.0, 50.0),
)
print(f"Normalized Gini: {ranking.normalized_gini:.4f}")
for point in ranking.selected_points:
    print(f"Exposure={point.percent:.0f}%: gain={point.gain:.4f}, lift={point.lift:.4f}")
print("Cumulative exposure:", ranking.cumulative_exposure.round(4).tolist())
print("Cumulative observed:", ranking.cumulative_observed.round(4).tolist())
```

<!-- output: metrics-ranking -->
```text
Normalized Gini: 0.4286
Exposure=25%: gain=0.2353, lift=0.9412
Exposure=50%: gain=0.5882, lift=1.1765
Cumulative exposure: [0.0, 0.1667, 0.5, 0.8333, 1.0]
Cumulative observed: [0.0, 0.1176, 0.5882, 0.9412, 1.0]
```
<!-- /output -->

The score puts the row with observed rate 2 above the row with rate 4, so
ranking is imperfect. Half of the exposure captures `10 / 17` of the observed
amount: 58.82%, or 1.1765 times uniform selection. At 25% exposure, capture is
slightly worse than uniform selection; the point interpolates within a row.

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
pu_values = pu_scores.to_dict()
print("Known-positive capture:", pu_values["known_positive_capture_at_10pct"])
print("Actual unlabeled fraction flagged:", pu_values["unlabeled_flagged_at_10pct"])
capacity, capture = positive_capture_curve(pu_labels, pu_probabilities)
print(capacity.tolist())
print(capture.tolist())
```

<!-- output: metrics-pu -->
```text
Known-positive capture: 0.5
Actual unlabeled fraction flagged: 0.5
[0.0, 0.0, 0.5, 0.5, 1.0]
[0.0, 0.5, 0.5, 1.0, 1.0]
```
<!-- /output -->

With only two unlabeled rows, reaching the requested 10% requires flagging
one whole row: actual capacity is 50%, and the same cutoff captures 50% of
known positives. Always inspect actual flagged capacity alongside capture.

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
print({name: round(value, 4) for name, value in metric(observed, probabilities, weight).to_dict().items()})
```

<!-- output: metrics-custom -->
```text
{'mae': 0.35}
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

::: ml_toolkit.metrics.metrics_frequency
    options:
      heading_level: 4

::: ml_toolkit.metrics.MetricsFrequency
    options:
      heading_level: 4

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
