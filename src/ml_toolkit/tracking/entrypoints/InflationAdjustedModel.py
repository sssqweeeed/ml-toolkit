from mlflow.models import set_model
from ml_toolkit.models import InflationAdjustedModel

set_model(InflationAdjustedModel())
