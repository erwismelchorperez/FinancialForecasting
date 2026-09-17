# src/models/dependent_trainer.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tbats import TBATS
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
from src.models.base_trainer import BaseTrainer


class DependentAccountTrainer(BaseTrainer):
    """
    Entrenador para cuentas que TIENEN dependencias.
    Usa tanto la serie temporal como las dependencias como features.
    """
    
    def __init__(self, models, test_size=12):
        super().__init__(models, test_size)
        self.ml_models = {
            "RANDOM_FOREST": RandomForestRegressor(n_estimators=100, random_state=42),
            "GRADIENT_BOOSTING": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBOOST": XGBRegressor(n_estimators=100, random_state=42),
            "LIGHTGBM": LGBMRegressor(n_estimators=100, random_state=42)
        }
    
    def _prepare_features(self, df):
        """
        Prepara las features para el modelo con dependencias.
        """
        # Identificar columnas de dependencias
        dep_cols = [col for col in df.columns if col.startswith('dep_') and not '_lag_' in col]
        
        # Identificar lags de dependencias
        dep_lags = [col for col in df.columns if col.startswith('dep_') and '_lag_' in col]
        
        # Features base (año, mes)
        feature_cols = ['year', 'month'] if 'year' in df.columns else []
        
        # Agregar lags del target
        target_lags = [col for col in df.columns if col.startswith('target_lag_')]
        feature_cols.extend(target_lags)
        
        # Agregar dependencias actuales y sus lags
        feature_cols.extend(dep_cols)
        feature_cols.extend(dep_lags)
        
        # Eliminar duplicados y columnas que no existen
        feature_cols = list(set([f for f in feature_cols if f in df.columns]))
        
        print(f"   Features seleccionadas: {len(feature_cols)}")
        for f in feature_cols[:5]:  # Mostrar primeras 5
            print(f"     - {f}")
        if len(feature_cols) > 5:
            print(f"     ... y {len(feature_cols) - 5} más")
        
        return feature_cols
    
    def train_account(self, df):
        """
        Entrena modelos para una cuenta con dependencias.
        """
        df = self._validate_data(df)
        
        print(f"\n📊 Entrenando cuenta con dependencias")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        
        # Preparar features
        features = self._prepare_features(df)
        
        if not features:
            print("  ⚠️ No se encontraron features. Usando solo target.")
            # Si no hay features, usar el trainer independiente
            from src.models.independent_trainer import IndependentAccountTrainer
            trainer = IndependentAccountTrainer(self.models, self.test_size)
            return trainer.train_account(df)
        
        # Preparar datos
        X = df[features].copy()
        y = df["target"].copy()
        
        # Transformación logarítmica
        y_log = np.log1p(y)
        
        # Limpiar datos
        X = X.replace([np.inf, -np.inf], np.nan)
        df_model = X.copy()
        df_model["target"] = y_log
        df_model = df_model.dropna()
        
        if len(df_model) < self.test_size + 5:
            raise ValueError(f"Datos insuficientes: {len(df_model)} filas después de limpieza")
        
        X = df_model[features]
        y = df_model["target"]
        
        # Split temporal
        split_idx = len(df_model) - self.test_size
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        results = {}
        
        # 1. Entrenar modelos tradicionales de series temporales (sin features)
        #    Solo usan la serie temporal 'target'
        print("  Entrenando modelos de series temporales...")
        y_series = df["target"].astype(float)
        y_train_ts = y_series.iloc[:split_idx]
        y_test_ts = y_series.iloc[split_idx:]
        
        # Usar los mismos modelos que en IndependentTrainer pero con los datos correctos
        ts_models = {k: v for k, v in self.models.items() 
                    if k in ["SARIMAX", "ETS", "AUTO_ARIMA", "TBATS", "THETA", "PROPHET"]}
        
        for name in ts_models:
            print(f"    Entrenando {name} (TS)...")
            try:
                if name == "SARIMAX":
                    model = SARIMAX(
                        y_train_ts,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12)
                    ).fit(disp=False)
                    y_pred = model.forecast(steps=len(y_test_ts))
                    results[name] = self._create_results(model, y_test_ts, y_pred, y_train_ts)
                    
                elif name == "PROPHET":
                    prophet_df = df[["Fecha", "target"]].copy()
                    prophet_df.columns = ["ds", "y"]
                    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
                    
                    train_df = prophet_df.iloc[:split_idx]
                    test_df = prophet_df.iloc[split_idx:]
                    
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    model.fit(train_df)
                    
                    future = model.make_future_dataframe(periods=self.test_size, freq="ME")
                    forecast_df = model.predict(future)
                    
                    y_pred = forecast_df["yhat"].tail(self.test_size).values
                    results[name] = self._create_results(model, test_df["y"], y_pred, train_df["y"])
                    
                elif name == "ETS":
                    model = ExponentialSmoothing(
                        y_train_ts,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=12
                    ).fit()
                    y_pred = model.forecast(len(y_test_ts))
                    results[name] = self._create_results(model, y_test_ts, y_pred, y_train_ts)
                    
                elif name == "AUTO_ARIMA":
                    model = auto_arima(
                        y_train_ts,
                        seasonal=True,
                        m=12,
                        suppress_warnings=True,
                        error_action="ignore"
                    )
                    y_pred = model.predict(n_periods=len(y_test_ts))
                    results[name] = self._create_results(model, y_test_ts, y_pred, y_train_ts)
                    
                elif name == "TBATS":
                    model = TBATS(
                        seasonal_periods=[12],
                        use_arma_errors=True,
                        use_box_cox=False,
                        use_trend=True,
                        use_damped_trend=True
                    )
                    fitted = model.fit(np.asarray(y_train_ts))
                    y_pred = fitted.forecast(steps=len(y_test_ts))
                    results[name] = self._create_results(fitted, y_test_ts, y_pred, y_train_ts)
                    
                elif name == "THETA":
                    model = ThetaModel(y_train_ts, period=12).fit()
                    y_pred = model.forecast(len(y_test_ts))
                    results[name] = self._create_results(model, y_test_ts, y_pred, y_train_ts)
                    
            except Exception as e:
                print(f"    ❌ Error en {name}: {e}")
                results[name] = None
        
        # 2. Entrenar modelos de Machine Learning (usan todas las features)
        print("  Entrenando modelos de Machine Learning...")
        
        for name, model in self.ml_models.items():
            print(f"    Entrenando {name} (ML)...")
            try:
                model.fit(X_train, y_train)
                y_pred_log = model.predict(X_test)
                
                # Revertir transformación logarítmica
                y_pred = np.expm1(y_pred_log)
                y_test_real = np.expm1(y_test)
                y_train_real = np.expm1(y_train)
                
                results[name] = self._create_results(
                    model, y_test_real, y_pred, y_train_real
                )
            except Exception as e:
                print(f"    ❌ Error en {name}: {e}")
                results[name] = None
        
        # Filtrar modelos válidos
        valid_models = {k: v for k, v in results.items() if v is not None}
        
        if valid_models:
            # Seleccionar mejor modelo basado en MAE
            best_model = min(valid_models.items(), key=lambda x: x[1]["metrics"]["mae"])
            print(f"  ✅ Mejor modelo: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f})")
        
        return valid_models