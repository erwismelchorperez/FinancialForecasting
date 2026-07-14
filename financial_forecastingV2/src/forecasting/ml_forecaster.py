from config.settings import (
    FORECAST_STEPS,
    LAGS
)

from src.forecasting.recursive import RecursiveForecast


class MLForecaster:

    def forecast(self, model, df_account):

        lag_cols = [
            f"lag_{lag}"
            for lag in LAGS
        ]

        history = []

        last_row = df_account.iloc[-1]

        for col in lag_cols:
            history.append(last_row[col])

        history.append(last_row["Valor"])

        feature_count = len(lag_cols)

        preds = RecursiveForecast.predict(
            model,
            history,
            FORECAST_STEPS,
            feature_count
        )

        return preds