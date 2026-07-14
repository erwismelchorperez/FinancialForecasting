import pandas as pd
import numpy as np

from src.utils.date_utils import get_last_values
from src.forecasting.recursive import recursive_forecast
from src.models.model_selector import ModelSelector
from config.config import LAGS, FORECAST_STEPS

class ForecastEngine:

    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.selector = ModelSelector()
    
    def forecast_account(self, df_account, model_results):
        #print("df_account       ", df_account)
        #print("model_results    ", model_results)
        # seleccionar mejor modelo
        best_model = self.selector.select_best(model_results)

        print(best_model['model'])
        
        # 🔴 SARIMAX
        if best_model['model'] == "SARIMAX":
            preds = best_model.forecast(steps=FORECAST_STEPS)
            return preds.tolist()
        else:    
            # obtener últimos valores
            history = np.log1p(get_last_values(df_account, max(LAGS)))

            # forecast
            preds = recursive_forecast(
                best_model['model'],
                history,
                FORECAST_STEPS,
                len(LAGS)
            )

        return preds
    def forecast_by_account(self, df_account, model_results):
        forecasts = {}

        for model_name, res in model_results.items():

            if res is None:
                continue

            model = res["model"]

            try:
                # 🔴 CASO SARIMAX
                if model_name == "SARIMAX":
                    preds = model.forecast(steps=FORECAST_STEPS)
                    forecasts[model_name] = preds.tolist()

                # 🔵 MODELOS SUPERVISADOS (lags)
                else:
                    history = get_last_values(df_account, max(LAGS))

                    preds = recursive_forecast(
                        model,
                        history,
                        FORECAST_STEPS,
                        len(LAGS)
                    )

                    forecasts[model_name] = preds

            except Exception as e:
                print(f"❌ Error en forecast con {model_name}: {e}")
                forecasts[model_name] = None

        return forecasts
    
    def forecast_all(self, df_long, all_results):

        forecasts = {}
        #print("df_long              ",df_long)
        #print("all_results          ",all_results)
        for account, model_results in all_results.items():

            df_account = df_long[df_long["BALANCE GENERAL"] == account]

            preds = self.forecast_by_account(df_account, model_results)

            forecasts[account] = preds

        return forecasts