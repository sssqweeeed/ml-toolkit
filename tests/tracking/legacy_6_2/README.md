# Frozen ml-toolkit 6.2.x artifacts

Written by ml-toolkit 6.2.x (6.2.0) on 2026-11-23. **Never regenerate these
files**: they pin backward compatibility, so newer code must keep loading them
and reproducing the stored predictions.

- `<model>/model/`: the logged MLflow pyfunc model directory of each adapter
  (CatBoost regression, log-link, classification, uPU, nnPU, inflation-adjusted,
  LightGBM serving). The bundled `code/` snapshot of the package was removed (and
  `code: code` dropped from `MLmodel`) so loading exercises the current code.
  Local user paths in `MLmodel` were scrubbed.
- `<model>/input.parquet`: serving input; `<model>/expected.parquet`: predictions
  returned by `mlflow.pyfunc.load_model(...).predict(input)` in 6.2.x.
- `psi/`: `PSIBaseline.save` output, a current sample and the expected
  `compare` result (`psi` and `level_distribution`).
- `levels/`: `LevelTransformer.to_dict()` JSON, an input frame and the expected
  `transform` output.

Consumed by `tests/tracking/test_legacy_artifacts.py`.
