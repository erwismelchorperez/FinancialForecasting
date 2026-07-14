from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Evaluator:

    def evaluate(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }