from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Evaluator:
    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.mean(np.abs((y_true-y_pred)/y_true))*100

    def evaluate(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mape(y_true, y_pred)
        }