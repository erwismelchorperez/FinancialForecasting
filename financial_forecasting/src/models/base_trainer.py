# src/models/base_trainer.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseTrainer(ABC):
    """
    Clase base abstracta para entrenadores de modelos.
    """
    
    def __init__(self, models, test_size=12):
        self.models = models
        self.test_size = test_size
    
    def mape(self, y_true, y_pred):
        """Calcula el MAPE evitando división por cero."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _create_results(self, model, y_test, y_pred, y_train):
        """Crea el diccionario de resultados de manera consistente."""
        y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape_val = self.mape(y_test, y_pred)
        transition_error = abs(y_pred[0] - y_train[-1]) if len(y_pred) > 0 else float('inf')
        
        return {
            "model": model,
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "mape": mape_val,
                "transition_error": transition_error,
                "y_pred": y_pred,
                "y_test": y_test
            }
        }
    
    @abstractmethod
    def train_account(self, df):
        """Método abstracto para entrenar modelos."""
        pass
    
    def _validate_data(self, df):
        """Valida que el DataFrame tenga las columnas necesarias."""
        if "target" not in df.columns:
            raise ValueError(f"No existe 'target'. Columnas: {df.columns}")
        if "Fecha" not in df.columns:
            raise ValueError("Falta columna 'Fecha'")
        return df.sort_values("Fecha").copy()