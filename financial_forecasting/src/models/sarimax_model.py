import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXModel:

    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, y):
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)

    def predict(self, X):
        # ⚠️ solo para compatibilidad (NO se usa realmente)
        return self.model_fit.fittedvalues

    def forecast(self, steps):
        return self.model_fit.forecast(steps=steps)