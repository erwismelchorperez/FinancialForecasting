import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TrainerTS:

    def train_sarimax(self, series):

        split = int(len(series) * 0.80)

        train = series.iloc[:split]
        test = series.iloc[split:]

        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )

        fitted = model.fit(disp=False)

        preds = fitted.forecast(len(test))

        return {
            "model": fitted,
            "mae": mean_absolute_error(test, preds),
            "rmse": np.sqrt(mean_squared_error(test, preds))
        }

    def train_holtwinters(self, series):

        split = int(len(series) * 0.80)

        train = series.iloc[:split]
        test = series.iloc[split:]

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

        fitted = model.fit()

        preds = fitted.forecast(len(test))

        return {
            "model": fitted,
            "mae": mean_absolute_error(test, preds),
            "rmse": np.sqrt(mean_squared_error(test, preds))
        }