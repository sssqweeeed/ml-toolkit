from mlflow.models import set_model

from ml_toolkit.models import CatBoostLogLinkRegressionModel

set_model(CatBoostLogLinkRegressionModel())
