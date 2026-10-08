from mlflow.models import set_model

from ml_toolkit.models.lightgbm import LightGBMWrapper

set_model(LightGBMWrapper())
