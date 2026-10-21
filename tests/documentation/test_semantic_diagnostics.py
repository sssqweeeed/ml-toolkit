"""Documentation calculations expose drift, invalid levels, and threshold tradeoffs."""

import doctest
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from examples.render_examples import execute, recipes
from ml_toolkit.feature_levels import FeatureLevelConfig
from ml_toolkit.plots.classification import binary_curves
from ml_toolkit.psi import PSIAnalysis, psi

pytestmark = pytest.mark.documentation
ROOT = Path(__file__).resolve().parents[2]


def example_namespace(obj):
    example = doctest.DocTestParser().get_doctest(
        obj.__doc__, {}, obj.__name__, None, 0
    )
    result = doctest.DocTestRunner().run(example, clear_globs=False)
    assert result.failed == 0 and result.attempted > 0
    return example.globs


def test_psi_examples_show_changed_shares():
    direct = example_namespace(psi)
    np.testing.assert_allclose(psi(direct['reference'], direct['current']), np.log(3))
    analysis = example_namespace(PSIAnalysis)
    result = analysis['result']
    np.testing.assert_allclose(result.psi['psi'], [np.log(3)])
    north = result.level_distribution.filter(pl.col('level') == 'north')
    np.testing.assert_allclose(north['share'], [0.75, 0.25])


def test_feature_rule_example_distinguishes_invalid_missing_and_valid_values():
    namespace = example_namespace(FeatureLevelConfig)
    transformed = namespace['levels'].transform(namespace['current'])
    assert transformed['age'].cast(pl.String).to_list() == [
        '__ERROR__', '__NULL__', '[0, 35)', '[35, inf)'
    ]


def test_threshold_example_demonstrates_precision_recall_tradeoff():
    namespace = example_namespace(binary_curves)
    strict, permissive = namespace['curve'].iloc[[0, 2]].to_dict('records')
    assert strict['precision'] > permissive['precision']
    assert strict['recall'] < permissive['recall']
    assert strict['f1'] < permissive['f1']


def test_psi_guide_reuses_reference_bins_and_exposes_unseen_share():
    namespace = {}
    for recipe in recipes(ROOT / 'docs/api/psi.md'):
        execute(recipe, namespace)
    assert namespace['levels'].feature_levels[0].boundaries == (35.0,)
    assert namespace['level_comparison'].psi.equals(namespace['comparison'].psi)
    unseen = namespace['comparison'].level_distribution.filter(
        (pl.col('feature') == 'region') & (pl.col('level') == '__UNSEEN__')
    )
    np.testing.assert_allclose(unseen['share'], [0, 0.5])
