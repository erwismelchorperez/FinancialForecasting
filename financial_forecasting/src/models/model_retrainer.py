# src/models/model_retrainer.py

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from tbats import TBATS
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


class ModelRetrainer:
    """
    Reentrena modelos seleccionados con todos los datos disponibles.
    """
    
    def __init__(self, models_config=None):
        """
        Args:
            models_config: Diccionario con configuración de modelos
        """
        self.models_config = models_config or self._default_config()
    
    def _default_config(self):
        """Configuración por defecto de modelos."""
        return {
            'SARIMAX': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 12)
            },
            'ETS': {
                'trend': 'add',
                'seasonal': 'add',
                'seasonal_periods': 12
            },
            'AUTO_ARIMA': {
                'seasonal': True,
                'm': 12,
                'suppress_warnings': True
            },
            'TBATS': {
                'seasonal_periods': [12],
                'use_arma_errors': True,
                'use_box_cox': False,
                'use_trend': True,
                'use_damped_trend': True
            },
            'THETA': {
                'period': 12
            },
            'PROPHET': {
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False
            },
            'RANDOM_FOREST': {
                'n_estimators': 100,
                'random_state': 42
            },
            'GRADIENT_BOOSTING': {
                'n_estimators': 100,
                'random_state': 42
            },
            'XGBOOST': {
                'n_estimators': 100,
                'random_state': 42
            },
            'LIGHTGBM': {
                'n_estimators': 100,
                'random_state': 42
            }
        }
    
    def retrain_model(self, df, account_id, model_name, model_data, forecast_steps=12):
        """
        Reentrena un modelo específico con todos los datos.
        
        Args:
            df: DataFrame con los datos
            account_id: ID de la cuenta
            model_name: Nombre del modelo a reentrenar
            model_data: Datos del modelo original (para configuraciones)
            forecast_steps: Número de pasos a pronosticar
            
        Returns:
            Modelo reentrenado y predicciones
        """
        print(f"      🔄 Reentrenando {model_name} con todos los datos...")
        
        # Preparar datos
        y_series = df["target"].astype(float)
        
        # Preparar features para ML
        feature_cols = [col for col in df.columns 
                       if col not in ['Fecha', 'target', 'account_id', 'NIVEL', 'BALANCE GENERAL']]
        
        try:
            # ============================================================
            # Modelos de Machine Learning
            # ============================================================
            if model_name in ['RANDOM_FOREST', 'GRADIENT_BOOSTING', 'XGBOOST', 'LIGHTGBM']:
                X = df[feature_cols].copy()
                y = np.log1p(y_series)
                
                # Limpiar datos
                X = X.replace([np.inf, -np.inf], np.nan)
                df_model = X.copy()
                df_model['target'] = y
                df_model = df_model.dropna()
                
                if len(df_model) < 5:
                    print(f"         ⚠️ Datos insuficientes para {model_name}")
                    return None, None
                
                X_train = df_model[feature_cols]
                y_train = df_model['target']
                
                # Crear modelo
                config = self.models_config.get(model_name, {})
                if model_name == 'RANDOM_FOREST':
                    model = RandomForestRegressor(**config)
                elif model_name == 'GRADIENT_BOOSTING':
                    model = GradientBoostingRegressor(**config)
                elif model_name == 'XGBOOST':
                    model = XGBRegressor(**config)
                elif model_name == 'LIGHTGBM':
                    model = LGBMRegressor(**config)
                
                # Entrenar
                model.fit(X_train, y_train)
                
                # Generar predicciones futuras
                # Preparar features futuras
                X_future = self._prepare_future_features(df, feature_cols, forecast_steps)
                if X_future is not None:
                    y_pred_log = model.predict(X_future)
                    y_pred = np.expm1(y_pred_log)
                    return model, y_pred.tolist()
                else:
                    return model, None
            
            # ============================================================
            # Modelos de Series Temporales
            # ============================================================
            elif model_name == 'SARIMAX':
                config = self.models_config.get('SARIMAX', {})
                model = SARIMAX(
                    y_series,
                    order=config.get('order', (1, 1, 1)),
                    seasonal_order=config.get('seasonal_order', (1, 1, 1, 12))
                ).fit(disp=False)
                y_pred = model.forecast(steps=forecast_steps)
                return model, y_pred.tolist()
            
            elif model_name == 'ETS':
                config = self.models_config.get('ETS', {})
                model = ExponentialSmoothing(
                    y_series,
                    trend=config.get('trend', 'add'),
                    seasonal=config.get('seasonal', 'add'),
                    seasonal_periods=config.get('seasonal_periods', 12)
                ).fit()
                y_pred = model.forecast(forecast_steps)
                return model, y_pred.tolist()
            
            elif model_name == 'AUTO_ARIMA':
                config = self.models_config.get('AUTO_ARIMA', {})
                model = auto_arima(
                    y_series,
                    seasonal=config.get('seasonal', True),
                    m=config.get('m', 12),
                    suppress_warnings=config.get('suppress_warnings', True),
                    error_action='ignore'
                )
                y_pred = model.predict(n_periods=forecast_steps)
                return model, y_pred.tolist()
            
            elif model_name == 'TBATS':
                config = self.models_config.get('TBATS', {})
                model = TBATS(
                    seasonal_periods=config.get('seasonal_periods', [12]),
                    use_arma_errors=config.get('use_arma_errors', True),
                    use_box_cox=config.get('use_box_cox', False),
                    use_trend=config.get('use_trend', True),
                    use_damped_trend=config.get('use_damped_trend', True)
                )
                fitted = model.fit(np.asarray(y_series))
                y_pred = fitted.forecast(steps=forecast_steps)
                return fitted, y_pred.tolist()
            
            elif model_name == 'THETA':
                config = self.models_config.get('THETA', {})
                model = ThetaModel(y_series, period=config.get('period', 12)).fit()
                y_pred = model.forecast(forecast_steps)
                return model, y_pred.tolist()
            
            elif model_name == 'PROPHET':
                config = self.models_config.get('PROPHET', {})
                prophet_df = df[['Fecha', 'target']].copy()
                prophet_df.columns = ['ds', 'y']
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                
                model = Prophet(**config)
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=forecast_steps, freq='ME')
                forecast = model.predict(future)
                y_pred = forecast['yhat'].tail(forecast_steps).values
                return model, y_pred.tolist()
            
            else:
                print(f"         ⚠️ Modelo {model_name} no soportado para reentrenamiento")
                return None, None
                
        except Exception as e:
            print(f"         ❌ Error reentrenando {model_name}: {e}")
            return None, None
    
    def _prepare_future_features(self, df, feature_cols, forecast_steps):
        """
        Prepara features para predicciones futuras (modelos ML).
        """
        try:
            # Obtener último registro
            last_record = df.iloc[-1:].copy()
            
            # Crear fechas futuras
            last_date = df['Fecha'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq='ME'
            )
            
            X_future = pd.DataFrame()
            
            for col in feature_cols:
                if col == 'year':
                    X_future[col] = [d.year for d in future_dates]
                elif col == 'month':
                    X_future[col] = [d.month for d in future_dates]
                elif col.startswith('dep_') or '_lag_' in col:
                    # Usar último valor conocido
                    if col in last_record.columns:
                        last_value = last_record[col].values[0] if not pd.isna(last_record[col].values[0]) else 0
                        X_future[col] = [float(last_value)] * forecast_steps
                    else:
                        X_future[col] = [0.0] * forecast_steps
                else:
                    # Otras columnas: usar último valor
                    if col in last_record.columns:
                        last_value = last_record[col].values[0] if not pd.isna(last_record[col].values[0]) else 0
                        X_future[col] = [float(last_value)] * forecast_steps
                    else:
                        X_future[col] = [0.0] * forecast_steps
            
            # Asegurar orden de columnas
            X_future = X_future[feature_cols]
            return X_future
            
        except Exception as e:
            print(f"         ⚠️ Error preparando features futuras: {e}")
            return None
    
    def retrain_best_models(self, df, account_id, selected_models, forecast_steps=12):
        """
        Reentrena todos los modelos seleccionados para una cuenta.
        
        Returns:
            Dict con {model_name: (model, predictions)}
        """
        retrained = {}
        
        for model_name, model_data in selected_models:
            model, predictions = self.retrain_model(
                df, account_id, model_name, model_data, forecast_steps
            )
            
            if model is not None and predictions is not None:
                retrained[model_name] = {
                    'model': model,
                    'predictions': predictions
                }
                print(f"         ✅ {model_name} reentrenado exitosamente")
            else:
                print(f"         ❌ {model_name} falló en el reentrenamiento")
        
        return retrained