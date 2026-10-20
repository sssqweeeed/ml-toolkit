from mlflow.models import set_model

from ml_toolkit.models.lightgbm import LightGBMServingModel

set_model(LightGBMServingModel())
