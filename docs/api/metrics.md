# Metrics

Import metrics and result types from `ml_toolkit.metrics`. Training metrics
receive targets, predictions, and weights and return an object with
`to_dict() -> dict[str, float]`. Use `MetricScores` to wrap custom named scores.

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
