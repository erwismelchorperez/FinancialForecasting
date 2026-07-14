from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tbats import TBATS
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
import numpy as np
import pandas as pd


class AccountModelTrainer:

    def __init__(self, models):
        self.models = models
        self.test_size = 12
    def mape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.mean(np.abs((y_true-y_pred)/y_true))*100
    def train_account(self, df):

        # Validaciones básicas
        if "Valor" not in df.columns:
            raise ValueError(f"No existe 'Valor'. Columnas actuales: {df.columns}")

        if "Fecha" not in df.columns:
            raise ValueError("Falta columna 'Fecha'")

        # Ordenar por tiempo (CRÍTICO)
        df = df.sort_values("Fecha").copy()

        # Definir features explícitas
        features = ["year", "month", "lag_1", "lag_2", "lag_3"]

        for f in features:
            if f not in df.columns:
                raise ValueError(f"Falta feature '{f}'. Revisa to_long()")

        # Variables
        X = df[features]
        y = df["Valor"]
        y = np.log1p(df["Valor"])

        # limpiar X e y correctamente
        X = X.replace([np.inf, -np.inf], np.nan)
        df_model = X.copy()
        df_model["Valor"] = y

        df_model = df_model.dropna()

        X = df_model[features]
        y = df_model["Valor"]

        """
        # Split temporal (80% train, 20% test)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        """

        X_train = X.iloc[:-self.test_size]
        X_test  = X.iloc[-self.test_size:]
        y_train = y.iloc[:-self.test_size]
        y_test  = y.iloc[-self.test_size:]

        if len(X_train) < 5 or len(X_test) < 2:
            raise ValueError(f"Datos insuficientes: train={len(X_train)}, test={len(X_test)}")
        # Validación crítica
        if X_train.shape[0] == 0 or X_train.shape[1] == 0:
            raise ValueError(f"X_train vacío: {X_train.shape}")

        results = {}

        # 🔹 Entrenamiento por modelo
        for name, model in self.models.items():
            print(f"Entrenando {name} con X_train shape: {X_train.shape}")
            try:
                if model== "SARIMAX":
                    y_series = df["Valor"].astype(float)
                    y_traindf = y_series.iloc[:-self.test_size]
                    y_testdf  = y_series.iloc[-self.test_size:]


                    model = SARIMAX(
                        y_traindf,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12)
                    ).fit(disp=False)

                    y_pred = model.forecast(steps=len(y_testdf))

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(y_testdf, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_testdf, y_pred)),
                            "mape": self.mape(y_testdf, y_pred),
                            "transition_error": abs(y_pred.iloc[0]-y_traindf.iloc[-1]),
                            "y_pred": y_pred,
                            "y_test": y_testdf.values}
                    }
                elif model == "PROPHET":
                    # ==========================
                    # Serie temporal
                    # ==========================
                    prophet_df = df[["Fecha", "Valor"]].copy()

                    prophet_df.columns = ["ds", "y"]

                    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
                    prophet_df["y"] = prophet_df["y"].astype(float)

                    # ==========================
                    # Split temporal
                    # ==========================
                    train_df = prophet_df.iloc[:-self.test_size]
                    test_df  = prophet_df.iloc[-self.test_size:]

                    # ==========================
                    # Modelo
                    # ==========================
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )

                    model.fit(train_df)

                    # ==========================
                    # Future dataframe
                    # ==========================
                    future = model.make_future_dataframe(
                        periods=self.test_size,
                        freq="ME"   # ← importante
                    )

                    forecast_df = model.predict(future)

                    # ==========================
                    # Predicciones SOLO test
                    # ==========================
                    y_pred = forecast_df["yhat"].tail(self.test_size).values

                    y_test = test_df["y"].values

                    # ==========================
                    # Guardar resultados
                    # ==========================
                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(y_test, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                            "mape": self.mape(y_test, y_pred),
                            "transition_error": abs(y_pred[0]-y_train.iloc[-1]),
                            "y_pred": y_pred,
                            "y_test": y_test
                        }
                    }
                elif name == "ETS":

                    from statsmodels.tsa.holtwinters import ExponentialSmoothing

                    y_series = (
                        df["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )

                    y_train = y_series.iloc[:-self.test_size]
                    y_test = y_series.iloc[-self.test_size:]

                    model = ExponentialSmoothing(
                        y_train,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=12
                    ).fit()

                    y_pred = model.forecast(
                        len(y_test)
                    )

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(
                                y_test,
                                y_pred
                            ),
                            "rmse": np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    y_pred
                                )
                            ),
                            "mape": self.mape(
                                y_test,
                                y_pred
                            ),
                            "transition_error": abs(
                                y_pred.iloc[0]
                                - y_train.iloc[-1]
                            ),
                            "y_pred": y_pred.values,
                            "y_test": y_test.values
                        }
                    }
                elif model == "AUTO_ARIMA":
                    y_series = (
                        df["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )

                    y_train = y_series.iloc[:-self.test_size]
                    y_test = y_series.iloc[-self.test_size:]

                    model = auto_arima(
                        y_train,
                        seasonal=True,
                        m=12,
                        suppress_warnings=True,
                        error_action="ignore"
                    )

                    y_pred = model.predict(
                        n_periods=len(y_test)
                    )

                    y_pred = np.asarray(
                        y_pred
                    )

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(
                                y_test,
                                y_pred
                            ),
                            "rmse": np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    y_pred
                                )
                            ),
                            "mape": self.mape(
                                y_test,
                                y_pred
                            ),
                            "transition_error": abs(
                                y_pred[0]
                                - y_train.iloc[-1]
                            ),
                            "y_pred": y_pred,
                            "y_test": y_test.values
                        }
                    }
                elif model == "TBATS":
                    # preparar serie
                    y_series = (
                        df["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )

                    y_train = y_series.iloc[:-self.test_size]
                    y_test = y_series.iloc[-self.test_size:]

                    # evitar errores
                    y_train = np.asarray(y_train)

                    model = TBATS(
                        seasonal_periods=[12],   # ← mensual
                        use_arma_errors=True,
                        use_box_cox=False,
                        use_trend=True,
                        use_damped_trend=True
                    )

                    fitted = model.fit(y_train)

                    y_pred = fitted.forecast(
                        steps=len(y_test)
                    )

                    results[name] = {
                        "model": fitted,
                        "metrics": {
                            "mae": mean_absolute_error(
                                y_test,
                                y_pred
                            ),
                            "rmse": np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    y_pred
                                )
                            ),
                            "mape": self.mape(
                                y_test.values,
                                y_pred
                            ),
                            "transition_error": abs(
                                y_pred[0]
                                - y_train.iloc[-1]
                            ),
                            "y_pred": y_pred,
                            "y_test": y_test.values
                        }
                    }
                elif model == "THETA":
                    y_series = (
                        df["Valor"]
                        .astype(float)
                        .reset_index(drop=True)
                    )

                    y_train = y_series.iloc[:-self.test_size]
                    y_test = y_series.iloc[-self.test_size:]

                    model = ThetaModel(
                        y_train,
                        period=12
                    ).fit()

                    y_pred = model.forecast(
                        len(y_test)
                    )

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(
                                y_test,
                                y_pred
                            ),
                            "rmse": np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    y_pred
                                )
                            ),
                            "mape": self.mape(
                                y_test,
                                y_pred
                            ),
                            "transition_error": abs(
                                y_pred.iloc[0]
                                - y_train.iloc[-1]
                            ),
                            "y_pred": y_pred.values,
                            "y_test": y_test.values
                        }
                    }
                else:
                    model.fit(X_train, y_train)

                    y_pred_log = model.predict(X_test)
                    y_pred = np.expm1(y_pred_log)
                    y_test_real = np.expm1(y_test)

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(y_test_real, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_test_real, y_pred)),
                            "mape": self.mape(y_test, y_pred),
                            "transition_error": abs(y_pred[0]-y_train.iloc[-1]),
                            "y_pred": y_pred,
                            "y_test": y_test_real.values}
                    }

            except Exception as e:
                print(f"❌ Error en modelo {name}: {e}")
                results[name] = None

        valid_models = {k: v for k, v in results.items() if v is not None}

        if len(valid_models) == 0:
            print("⚠️ Todos los modelos fallaron para esta cuenta")
            return {}

        return results