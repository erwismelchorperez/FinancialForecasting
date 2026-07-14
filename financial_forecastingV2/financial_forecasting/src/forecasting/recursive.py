import numpy as np

def recursive_forecast(model, history, steps, lags):

    history = list(history)
    preds = []

    for _ in range(steps):
        X = np.array(history[-lags:]).reshape(1, -1)
        #pred = model.predict(X)[0]

        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)

        preds.append(np.expm1(pred_log))
        history.append(pred_log)

    return preds