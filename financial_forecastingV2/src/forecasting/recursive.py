import numpy as np


class RecursiveForecast:

    @staticmethod
    def predict(model, history, steps, feature_count):

        history = list(history)

        forecasts = []

        for _ in range(steps):

            X = np.array(
                history[-feature_count:]
            ).reshape(1, -1)

            pred = model.predict(X)[0]

            forecasts.append(pred)

            history.append(pred)

        return forecasts