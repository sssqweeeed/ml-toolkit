# Tests

Install the package and test dependencies from the repository root:

```bash
python -m pip install -e '.[test,all]'
```

Run the suite sequentially or with a bounded number of worker processes:

```bash
python -m pytest
python -m pytest -n 2 --dist loadscope
python -m pyright
```

`pytest-xdist` also supports `-n auto`. Each worker imports the ML dependencies
and may train models, so start with two workers on machines with limited memory.
`loadscope` keeps tests sharing module or class fixtures on the same worker.
Numerical backend threads default to one per process; explicit environment
settings override these defaults. Matplotlib uses the noninteractive Agg backend.

Folders follow package responsibilities: `data`, `config`, `training`, `models`,
`metrics`, `diagnostics`, `plots`, `psi`, `temporal`, and `tracking`.
`documentation` checks executable recipes and original gallery datasets;
`typing` checks accepted and rejected API usage using the inputs in `typecheck`.
Reusable model/data doubles belong in `support`, never in another test module.
Keep fixtures close to their consumers and use `tmp_path` for mutable artifacts.

Select a subsystem, exclude documentation recipes, or measure branch coverage:

```bash
python -m pytest tests/metrics
python -m pytest -m 'not documentation'
python -m pytest -n 2 --dist loadscope --cov --cov-report=term-missing
```

Optional backend tests skip when their dependency is absent. Real-data gallery
checks require locally cached UCI Bike Sharing, UCI Online Retail, and LINQS Cora
sources and never download them implicitly. CUDA simulation runs on CPU; tests
marked `gpu` require a real CUDA device and compatible toolkit. Skip reasons
appear in the summary.

## Documentation

Build the site with only the documentation dependencies; mkdocstrings reads
`src/` statically, so building HTML needs neither the ML stack nor datasets:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
```

To validate executable examples, captured output, and committed figures, install
the ML dependencies and explicitly prepare the original source datasets first:

```bash
python -m pip install -e '.[test,all]' -r requirements-examples.txt
python -m examples.prepare_data
python -m pytest tests/documentation -n 2 --dist worksteal --durations=15
```

This suite includes every page and figure checked by
`python -m examples.render_examples --check`; running both commands repeats model
training, tuning, SHAP calculations, and figure generation. Use
`python -m examples.render_examples` when intentionally updating example output
and figures. The documentation suite uses `worksteal` to distribute individual
recipes of different durations between workers, instead of keeping a whole
module on one worker. The duration report highlights the remaining slow tests.

During development, select and run the checks relevant to the change locally.
PRs and pushes to `dev` do not start automatic validation or documentation
publication. The `Tests` workflow runs the complete release gate for PRs into
`main`: package tests and strict Pyright on Python 3.11/3.12, documentation
recipes once on Python 3.12, a strict site build, and distribution installation
checks. New commits cancel obsolete checks for the same release PR.

The `Documentation` workflow publishes versioned release documentation and the
`stable` alias after release validation. Manual publication retries run on
`main`. Merging a release reuses its successful PR validation instead of
rerunning the ML documentation checks. Development documentation stays local.

## Gamma and PU loss checks

`tests/models/test_gamma.py` checks Gamma derivatives, deviance, input validation,
training equivalence, prediction modes and native persistence. The serving tests
also load Gamma and inflation-adjusted Gamma models without Numba or CUDA.
`tests/gamma_cuda_check.py` exercises the actual CUDA kernel under Numba's
simulator, while the type-inference test catches unsupported CUDA operations.
The `gpu` integration test requires a real compatible CUDA device; simulator
success does not replace that hardware check. PU tests cover the objectives
in `models.losses.pu`.
