from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
"""
class ModelFactory:

    @staticmethod
    def get_models():
        return {
            # 🔹 Modelos base
            #"Linear": LinearRegression(),
            # 🔹 Modelos con escalado
            #"Ridge": Pipeline([("scaler", StandardScaler()),("model", Ridge(alpha=1.0))]),
            #"Lasso": Pipeline([("scaler", StandardScaler()),("model", Lasso(alpha=0.1))]),
            #"SVR": Pipeline([("scaler", StandardScaler()),("model", SVR())]),
            #"KNN": Pipeline([("scaler", StandardScaler()),("model", KNeighborsRegressor(n_neighbors=5))]),
            # 🔹 Árboles (NO necesitan escalado)
            #"Tree": DecisionTreeRegressor(max_depth=5),
            "RF": RandomForestRegressor(n_estimators=100,random_state=42),
            "GB": GradientBoostingRegressor(),
            "XGB": XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42,n_jobs=-1,verbosity=0), 
            # 🔥 IMPORTANTE: aquí NO instanciamos modelo sklearn
            "SARIMAX": "SARIMAX",
            "PROPHET": "PROPHET",
            "ETS": "ETS",
            "AUTO_ARIMA": "AUTO_ARIMA",
            #"TBATS": "TBATS",
            "THETA": "THETA"            
        }
"""
# src/models/model_factory.py

class ModelFactory:
    
    @staticmethod
    def get_models():
        """
        Retorna todos los modelos disponibles.
        """
        return {
            "SARIMAX": "SARIMAX",
            "ETS": "ETS",
            "AUTO_ARIMA": "AUTO_ARIMA",
            #"TBATS": "TBATS",
            "THETA": "THETA",
            "PROPHET": "PROPHET",
        }
    
    @staticmethod
    def get_ml_models():
        """Retorna solo modelos de Machine Learning."""
        return {
            "RANDOM_FOREST": "RANDOM_FOREST",
            "GRADIENT_BOOSTING": "GRADIENT_BOOSTING",
            "XGBOOST": "XGBOOST",
            "LIGHTGBM": "LIGHTGBM"
        }
    
    @staticmethod
    def get_time_series_models():
        """Retorna solo modelos de series temporales."""
        return {
            "SARIMAX": "SARIMAX",
            "ETS": "ETS",
            "AUTO_ARIMA": "AUTO_ARIMA",
            #"TBATS": "TBATS",
            "THETA": "THETA",
            "PROPHET": "PROPHET"
        }