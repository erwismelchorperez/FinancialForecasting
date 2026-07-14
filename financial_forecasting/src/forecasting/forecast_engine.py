import pandas as pd
import numpy as np

from src.utils.date_utils import get_last_values
from src.forecasting.recursive import recursive_forecast
from src.models.model_selector import ModelSelector
from config.config import LAGS, FORECAST_STEPS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

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
                    """
                    # con esto estoy prediciedo sin ver el último año
                    preds = model.forecast(steps=FORECAST_STEPS)
                    forecasts[model_name] = preds.tolist()
                    """

                    y_series = (
                        df_account
                        .sort_values("Fecha")
                        ["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )
                    final_model = SARIMAX(
                        y_series,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)

                    # Forecast futuro
                    preds = final_model.forecast(
                        steps=FORECAST_STEPS
                    )

                    forecasts[model_name] = preds.tolist()
                elif model_name == "PROPHET":
                    # reconstruir serie completa
                    prophet_df = (df_account[["Fecha", "Valor"]].copy())
                    prophet_df.columns = ["ds","y"]

                    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

                    prophet_df["y"] = (prophet_df["y"].astype(float))

                    # reentrenar con TODO el histórico
                    final_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )

                    final_model.fit(prophet_df)

                    # generar fechas futuras
                    future = final_model.make_future_dataframe(periods=FORECAST_STEPS, freq="ME")
                    forecast_df = final_model.predict(future)
                    preds = (forecast_df["yhat"].tail(FORECAST_STEPS).tolist())
                    forecasts[model_name] = preds
                # ==========================================
                # ETS
                # ==========================================
                elif model_name == "ETS":
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    y_series = (
                        df_account
                        .sort_values("Fecha")
                        ["Valor"]
                        .astype(float)
                    )

                    final_model = ExponentialSmoothing(
                        y_series,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=12
                    ).fit()

                    preds = final_model.forecast(
                        FORECAST_STEPS
                    )

                    forecasts[model_name] = (
                        preds
                        .tolist()
                    )
                # ==========================================
                # AUTO ARIMA
                # ==========================================
                elif model_name == "AUTO_ARIMA":
                    from pmdarima import auto_arima

                    y_series = (
                        df_account
                        .sort_values("Fecha")
                        ["Valor"]
                        .astype(float)
                    )

                    final_model = auto_arima(
                        y_series,
                        seasonal=True,
                        m=12,
                        suppress_warnings=True,
                        error_action="ignore"
                    )

                    preds = final_model.predict(
                        n_periods=FORECAST_STEPS
                    )

                    forecasts[model_name] = (
                        preds
                        .tolist()
                    )
                # ==========================================
                # TBATS
                # ==========================================
                elif model_name == "TBATS":

                    from tbats import TBATS

                    y_series = (
                        df_account["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )

                    final_model = TBATS(
                        seasonal_periods=[12],
                        use_arma_errors=True,
                        use_box_cox=False
                    ).fit(y_series)

                    preds = final_model.forecast(
                        steps=FORECAST_STEPS
                    )

                    forecasts[model_name] = preds.tolist()
                # ==========================================
                # THETA
                # ==========================================
                elif model_name == "THETA":

                    from statsmodels.tsa.forecasting.theta import ThetaModel

                    y_series = (
                        df_account
                        .sort_values("Fecha")
                        ["Valor"]
                        .astype(float)
                    )

                    final_model = ThetaModel(
                        y_series,
                        period=12
                    ).fit()

                    preds = final_model.forecast(
                        FORECAST_STEPS
                    )

                    forecasts[model_name] = (
                        preds
                        .tolist()
                    )
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