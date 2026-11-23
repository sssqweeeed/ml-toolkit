from mlflow.models import set_model

from ml_toolkit.models import CatBoostPUClassificationModel

set_model(CatBoostPUClassificationModel())
