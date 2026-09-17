# src/models/independent_trainer.py

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tbats import TBATS
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
from src.models.base_trainer import BaseTrainer


class IndependentAccountTrainer(BaseTrainer):
    """
    Entrenador para cuentas que NO tienen dependencias.
    Solo usa la serie temporal de la cuenta objetivo.
    """
    
    def __init__(self, models, test_size=12):
        super().__init__(models, test_size)
    
    def train_account(self, df):
        """
        Entrena modelos para una cuenta sin dependencias.
        Solo usa la serie temporal 'target'.
        """
        df = self._validate_data(df)
        
        # Verificar que solo tenemos target y Fecha
        print(f"\n📊 Entrenando cuenta sin dependencias")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        
        # Obtener la serie temporal
        y_series = df["target"].astype(float)
        
        # Split temporal
        split_idx = len(df) - self.test_size
        if split_idx < 5:
            raise ValueError(f"Datos insuficientes: {len(df)} filas")
        
        y_train = y_series.iloc[:split_idx]
        y_test = y_series.iloc[split_idx:]
        
        results = {}
        
        # Entrenar cada modelo
        for name, model in self.models.items():
            print(f"  Entrenando {name}...")
            try:
                if name == "SARIMAX":
                    sarimax_model = SARIMAX(
                        y_train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12)
                    ).fit(disp=False)
                    y_pred = sarimax_model.forecast(steps=len(y_test))
                    results[name] = self._create_results(sarimax_model, y_test, y_pred, y_train)
                    
                elif name == "PROPHET":
                    prophet_df = df[["Fecha", "target"]].copy()
                    prophet_df.columns = ["ds", "y"]
                    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
                    
                    train_df = prophet_df.iloc[:split_idx]
                    test_df = prophet_df.iloc[split_idx:]
                    
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    prophet_model.fit(train_df)
                    
                    future = prophet_model.make_future_dataframe(periods=self.test_size, freq="ME")
                    forecast_df = prophet_model.predict(future)
                    
                    y_pred = forecast_df["yhat"].tail(self.test_size).values
                    results[name] = self._create_results(prophet_model, test_df["y"], y_pred, train_df["y"])
                    
                elif name == "ETS":
                    ets_model = ExponentialSmoothing(
                        y_train,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=12
                    ).fit()
                    y_pred = ets_model.forecast(len(y_test))
                    results[name] = self._create_results(ets_model, y_test, y_pred, y_train)
                    
                elif name == "AUTO_ARIMA":
                    auto_model = auto_arima(
                        y_train,
                        seasonal=True,
                        m=12,
                        suppress_warnings=True,
                        error_action="ignore"
                    )
                    y_pred = auto_model.predict(n_periods=len(y_test))
                    results[name] = self._create_results(auto_model, y_test, y_pred, y_train)
                    
                elif name == "TBATS":
                    tbats_model = TBATS(
                        seasonal_periods=[12],
                        use_arma_errors=True,
                        use_box_cox=False,
                        use_trend=True,
                        use_damped_trend=True
                    )
                    fitted = tbats_model.fit(np.asarray(y_train))
                    y_pred = fitted.forecast(steps=len(y_test))
                    results[name] = self._create_results(fitted, y_test, y_pred, y_train)
                    
                elif name == "THETA":
                    theta_model = ThetaModel(y_train, period=12).fit()
                    y_pred = theta_model.forecast(len(y_test))
                    results[name] = self._create_results(theta_model, y_test, y_pred, y_train)
                    
                else:
                    print(f"  ⚠️ Modelo {name} no soportado para cuentas independientes")
                    
            except Exception as e:
                print(f"  ❌ Error en modelo {name}: {e}")
                results[name] = None
        
        # Filtrar modelos válidos
        valid_models = {k: v for k, v in results.items() if v is not None}
        
        if valid_models:
            # Seleccionar mejor modelo basado en MAE
            best_model = min(valid_models.items(), key=lambda x: x[1]["metrics"]["mae"])
            print(f"  ✅ Mejor modelo: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f})")
        
        return valid_models