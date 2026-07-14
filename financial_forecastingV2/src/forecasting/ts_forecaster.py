from config.settings import FORECAST_STEPS


class TSForecaster:

    def forecast(self, fitted_model):

        preds = fitted_model.forecast(
            steps=FORECAST_STEPS
        )

        return list(preds)