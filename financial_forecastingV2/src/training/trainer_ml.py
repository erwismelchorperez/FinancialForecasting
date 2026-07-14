import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


class TrainerML:

    def __init__(self, models):
        self.models = models

    def train(self, df):

        features = [
            "year",
            "month",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_6",
            "lag_12"
        ]

        df_model = df.dropna().copy()

        X = df_model[features]
        y = df_model["Valor"]

        split = int(len(df_model) * 0.80)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        results = {}

        for name, model in self.models.items():

            try:

                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                results[name] = {
                    "model": model,
                    "mae": mean_absolute_error(y_test, preds),
                    "rmse": np.sqrt(
                        mean_squared_error(y_test, preds)
                    )
                }

            except Exception as e:

                print(f"Error en {name}: {e}")

        return results