from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


class AccountModelTrainer:

    def __init__(self, models):
        self.models = models

    def train_account(self, df):

        # 🔴 Validaciones básicas
        if "Valor" not in df.columns:
            raise ValueError(f"No existe 'Valor'. Columnas actuales: {df.columns}")

        if "Fecha" not in df.columns:
            raise ValueError("Falta columna 'Fecha'")

        # 🔹 Ordenar por tiempo (CRÍTICO)
        df = df.sort_values("Fecha").copy()

        # 🔹 Definir features explícitas
        features = ["year", "month", "lag_1", "lag_2", "lag_3"]

        for f in features:
            if f not in df.columns:
                raise ValueError(f"Falta feature '{f}'. Revisa to_long()")

        # 🔹 Variables
        X = df[features]
        y = df["Valor"]
        y = np.log1p(df["Valor"])

        # 🔥 limpiar X e y correctamente
        X = X.replace([np.inf, -np.inf], np.nan)
        df_model = X.copy()
        df_model["Valor"] = y

        df_model = df_model.dropna()

        X = df_model[features]
        y = df_model["Valor"]

        # 🔹 Split temporal (80% train, 20% test)
        split = int(len(df) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if len(X_train) < 5 or len(X_test) < 2:
            raise ValueError(f"Datos insuficientes: train={len(X_train)}, test={len(X_test)}")
        # 🔴 Validación crítica
        if X_train.shape[0] == 0 or X_train.shape[1] == 0:
            raise ValueError(f"X_train vacío: {X_train.shape}")

        results = {}

        # 🔹 Entrenamiento por modelo
        for name, model in self.models.items():
            print(f"Entrenando {name} con X_train shape: {X_train.shape}")
            try:
                if model== "SARIMAX":
                    y_series = df["Valor"].astype(float)
                    # 🔴 MUY IMPORTANTE: también respetar orden temporal
                    #y_series = y_series.sort_index()

                    split = int(len(y_series) * 0.8)

                    y_train = y_series.iloc[:split]
                    y_test  = y_series.iloc[split:]

                    model = SARIMAX(
                        y_train,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12)
                    ).fit(disp=False)

                    y_pred = model.forecast(steps=len(y_test))

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(y_test, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                            "y_pred": y_pred,
                            "y_test": y_test.values}
                    }

                else:
                    model.fit(X_train, y_train)
                    #y_pred = model.predict(X_test)

                    y_pred_log = model.predict(X_test)
                    y_pred = np.expm1(y_pred_log)
                    y_test_real = np.expm1(y_test)

                    results[name] = {
                        "model": model,
                        "metrics": {
                            "mae": mean_absolute_error(y_test_real, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_test_real, y_pred)),
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