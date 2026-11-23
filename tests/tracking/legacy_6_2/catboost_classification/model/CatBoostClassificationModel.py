from mlflow.models import set_model

from ml_toolkit.models import CatBoostClassificationModel

set_model(CatBoostClassificationModel())
