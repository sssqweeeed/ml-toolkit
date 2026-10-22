# Positive-unlabeled

Use the [original LINQS Cora archive](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
for an observed-link retrieval example: 2,708 real papers, their 1,433 binary
word indicators, and all 5,429 recorded directed citations. Recorded links are
known positives. The loader selects 20,000 distinct unrecorded directed pairs
as unverified candidates; their citation status is unknown.

Scores are cosine similarities between the source word vectors. They are
retrieval scores in `[0, 1]`, without a probability interpretation or fitted PU
classifier. Cora's published document classes are unused. This constructs an
open-world citation retrieval example; the sampled candidate ratio does not
estimate citation prevalence.

From an installed source checkout, prepare the original citation source first:

```bash
python -m pip install -r requirements-examples.txt
python -m examples.prepare_data --dataset cora
```

<!-- example: pu-setup -->

```python
from pathlib import Path
from ml_toolkit.config import FigureConfig

output = Path("figures")
figure = FigureConfig(dpi=150)
```

<!-- example: plots-pu-distribution -->
```python
from examples.citation_data import load_citation_gallery
from ml_toolkit.plots import pu

citations = load_citation_gallery(download_if_missing=False)
pu_labels, pu_scores = (citations.labels, citations.scores)
pu.score_distribution(
    pu_labels, pu_scores, output / "pu-distribution.png", figure=figure
)
print(
    f"Cora: {citations.paper_count} papers, {citations.known_citation_count} recorded citations, {citations.unlabeled_pair_count} unverified candidate pairs"
)
```

<!-- output: plots-pu-distribution -->
```text
Cora: 2708 papers, 5429 recorded citations, 20000 unverified candidate pairs
```
<!-- /output -->

![Word-similarity distributions for 5429 real recorded citations and 20000 unverified Cora candidate pairs.](../assets/plots/pu-distribution.png)

??? example "Inspect Cora records, candidate selection, and similarity scoring"

    ```python
    --8<-- "examples/citation_data.py"
    ```

<!-- example: plots-pu-capture -->
```python
pu.positive_capture(
    pu_labels, pu_scores, output / "pu-capture.png", figure=figure
)
```

![Capture of recorded Cora citations against the share of unverified candidate pairs reviewed.](../assets/plots/pu-capture.png)

Both calls return `None`. Histograms normalize the recorded-link and candidate
sets separately. Capture plots show recorded citations recovered as the
unverified review fraction increases. Missing records are not confirmed
negative citations, and capture on this graph does not establish recall for
all real citations. Both axes show percentages; tied scores enter together.

## Reports in a PU training workflow

`PUScoreDistribution(histogram_bins=30)` separates score distributions for known
positives and unlabeled rows, normalizing each group independently.
`PUPositiveCapture()` plots known-positive capture against the unlabeled
fraction reviewed. Both use row weights and only the current held-out split.
They require both groups to have positive total weight. The distribution's
`histogram_bins` controls score bins; capture processes tied scores together.

For an existing project PU workflow, add the following evaluators. The callable
below requires your actual chronologically ordered PU pool, model parameters,
and population prior; the Cora illustration above supplies no training dates
or population prior and is not used to fabricate a PU training run.

<!-- example: pu-workflow -->

```python
from ml_toolkit import train_model
from ml_toolkit.config import TrainingConfig
from ml_toolkit.data import Pool
from ml_toolkit.evaluation import PUScoreDistribution, PUPositiveCapture
from ml_toolkit.metrics import PUMetric
from ml_toolkit.models import CatBoostPUClassificationModel
from ml_toolkit.training import Holdout
from ml_toolkit.types import ModelParams


def train_pu_reports(data: Pool, *, prior: float, params: ModelParams):
    return train_model(
        splitter=Holdout(data, val_size=0.2, test_size=0.2),
        params={**params, "pu_prior": prior, "loss": "nnPU"},
        model_factory=CatBoostPUClassificationModel,
        metric=PUMetric(prior=prior, loss="nnPU"),
        run_name="pu-reports",
        config=TrainingConfig(),
        prediction_evaluators=[PUScoreDistribution(), PUPositiveCapture()],
    )
```

In a test run the files are `pu_score_distribution/test/pu_score_distribution.png`
and `pu_positive_capture/test/pu_positive_capture.png`; validation uses `val/`.
The displayed Cora figures illustrate the same direct plotting functions used
by those evaluators, using the stated retrieval scores rather than a fitted PU
model. A prior is a population prevalence assumption, not the labeled fraction.

## API reference

::: ml_toolkit.evaluation.PUScoreDistribution
    options:
      heading_level: 3
      members: []

::: ml_toolkit.evaluation.PUPositiveCapture
    options:
      heading_level: 3
      members: []

::: ml_toolkit.plots.pu
    options:
      heading_level: 3
      members: [score_distribution, positive_capture]
