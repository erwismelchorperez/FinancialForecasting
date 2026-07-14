import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import pandas as pd
import numpy as np

class ForecastPlotter:

    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_name(self, name):
        return str(name).replace(" ", "_").replace("/", "_")
    def plot_test_vs_pred(self, model_results, df_account, account_name):

        TEST_SIZE = 12

        plt.figure(figsize=(12, 6))

        account_name_clean = self._sanitize_name(account_name)

        # ==========================================
        # Orden temporal
        # ==========================================
        df_account = df_account.dropna().sort_values("Fecha")

        # ==========================================
        # Fechas del conjunto TEST
        # ==========================================
        fechas_test = df_account["Fecha"].iloc[-TEST_SIZE:]

        # ==========================================
        # Obtener y_test
        # ==========================================
        y_test = None

        for res in model_results.values():

            if res is None:
                continue

            metrics = res.get("metrics", {})

            y_test = metrics.get("y_test")

            if y_test is not None:
                break

        # ==========================================
        # Graficar valores reales
        # ==========================================
        if y_test is not None:

            # seguridad extra
            fechas_real = fechas_test.iloc[:len(y_test)]

            plt.plot(
                fechas_real,
                y_test,
                label="Real",
                linewidth=2,
                color="black"
            )

        # ==========================================
        # Predicciones por modelo
        # ==========================================
        for model_name, res in model_results.items():

            if res is None:
                continue

            metrics = res.get("metrics", {})

            y_pred = metrics.get("y_pred")

            if y_pred is None:
                continue

            # seguridad por si algún modelo devuelve menos predicciones
            fechas_pred = fechas_test.iloc[:len(y_pred)]

            plt.plot(
                fechas_pred,
                y_pred,
                linestyle="--",
                label=f"Pred ({model_name})"
            )

        # ==========================================
        # Estética
        # ==========================================
        plt.title(f"Test vs Predicción - {account_name}")

        plt.xlabel("Fecha")

        plt.ylabel("Valor")

        plt.legend()

        plt.grid(True)

        # ==========================================
        # Guardar
        # ==========================================
        file_path = f"{self.output_dir}/test_vs_pred_{account_name_clean}.png"

        plt.savefig(file_path, bbox_inches="tight")

        plt.close()

        print(f"📊 Guardado: {file_path}")
    """
    def plot_test_vs_pred(self, model_results, df_account, account_name):
        plt.figure(figsize=(12, 6))

        account_name_clean = self._sanitize_name(account_name)

        # 🔴 1. reconstruir split (igual que en entrenamiento)
        df_account = df_account.dropna().sort_values("Fecha")

        split = int(len(df_account) * 0.8)

        df_test = df_account.iloc[split:]

        fechas_test = df_test["Fecha"]

        # 🔴 2. obtener y_test
        y_test = None
        for res in model_results.values():
            metrics = res.get("metrics", {})
            y_test = metrics.get("y_test")

            if y_test is not None:
                break

        # 🔴 3. graficar real
        if y_test is not None:
            plt.plot(fechas_test, y_test, label="Real", linewidth=2, color="black")

        # 🔵 4. predicciones por modelo
        for model_name, res in model_results.items():
            metrics = res.get("metrics", {})
            y_pred = metrics.get("y_pred")

            if y_pred is None:
                continue

            plt.plot(
                fechas_test,
                y_pred,
                linestyle="--",
                label=f"Pred ({model_name})"
            )

        plt.title(f"Test vs Predicción - {account_name}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        file_path = f"{self.output_dir}/test_vs_pred_{account_name_clean}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        print(f"📊 Guardado: {file_path}")
    """
    def plot_forecast(self, df_account, forecasts, future_dates, account_name):
        plt.figure(figsize=(12, 6))

        account_name_clean = self._sanitize_name(account_name)

        # ==============================
        # HISTÓRICO
        # ==============================
        plt.plot(
            df_account["Fecha"],
            df_account["Valor"],
            label="Histórico",
            linewidth=3,
            color="black"
        )

        # ==============================
        # FORECASTS POR MODELO
        # ==============================
        for model_name, preds in forecasts.items():

            # evitar modelos vacíos
            if preds is None:
                continue

            plt.plot(
                future_dates,
                preds,
                linestyle="--",
                label=model_name
            )

        # ==============================
        # ESTÉTICA
        # ==============================
        plt.title(f"Forecast - {account_name}")

        plt.xlabel("Fecha")
        plt.ylabel("Valor")

        plt.legend()

        plt.grid(True)

        # ==============================
        # GUARDAR
        # ==============================
        file_path = f"{self.output_dir}/forecast_{account_name_clean}.png"

        plt.savefig(file_path, bbox_inches="tight")

        plt.close()

        print(f"📈 Guardado: {file_path}")
    """
    def plot_forecast(self, df_account, forecast, future_dates, account_name):
        plt.figure(figsize=(10, 5))

        account_name_clean = self._sanitize_name(account_name)

        # histórico
        plt.plot(df_account["Fecha"], df_account["Valor"], label="Histórico")

        # forecast
        plt.plot(future_dates, forecast, linestyle="--", label="Forecast")

        plt.title(f"Forecast - {account_name}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        file_path = f"{self.output_dir}/forecast_{account_name_clean}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        print(f"📈 Guardado: {file_path}")
    """
    def plot_all_model_forecasts(self,df_account,model_results,future_dates,account_name):

        plt.figure(figsize=(12, 6))

        account_name_clean = self._sanitize_name(account_name)

        # Histórico
        plt.plot(
            df_account["Fecha"],
            df_account["Valor"],
            linewidth=3,
            label="Histórico"
        )

        for model_name, res in model_results.items():

            if res is None:
                continue

            model = res.get("model")

            if model is None:
                continue

            try:

                # ==========================
                # SARIMAX
                # ==========================
                if model_name == "SARIMAX":

                    y_series = (
                        df_account
                        .sort_values("Fecha")
                        ["Valor"]
                        .astype(float)
                    )

                    final_model = SARIMAX(
                        y_series,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)

                    preds = final_model.forecast(
                        steps=len(future_dates)
                    )

                    preds = preds.tolist()

                # ==========================
                # PROPHET
                # ==========================
                elif model_name == "PROPHET":

                    prophet_df = (
                        df_account
                        [["Fecha","Valor"]]
                        .rename(
                            columns={
                                "Fecha":"ds",
                                "Valor":"y"
                            }
                        )
                    )

                    prophet_df["ds"] = pd.to_datetime(
                        prophet_df["ds"]
                    )

                    final_model = model.__class__()

                    final_model.fit(prophet_df)

                    future = final_model.make_future_dataframe(
                        periods=len(future_dates),
                        freq="ME"
                    )

                    forecast = final_model.predict(future)

                    preds = (
                        forecast
                        ["yhat"]
                        .tail(len(future_dates))
                        .tolist()
                    )

                # ==========================
                # ML clásicos
                # ==========================
                else:

                    last_X = (
                        df_account
                        .select_dtypes(include=["number"])
                        .drop(columns=["Valor"])
                        .iloc[-1:]
                        .copy()
                    )

                    preds = []

                    for _ in range(len(future_dates)):

                        pred = model.predict(last_X)[0]

                        preds.append(pred)

                        if "lag_3" in last_X.columns:
                            last_X["lag_3"] = last_X["lag_2"]
                            last_X["lag_2"] = last_X["lag_1"]
                            last_X["lag_1"] = pred

                # Graficar
                plt.plot(
                    future_dates,
                    preds,
                    linestyle="--",
                    label=model_name
                )

            except Exception as e:
                print(f"Error con {model_name}: {e}")

        plt.title(f"Forecast por Modelo - {account_name}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")

        plt.legend()
        plt.grid(True)

        plt.xticks(rotation=45)

        file_path = (
            f"{self.output_dir}/"
            f"all_forecasts_{account_name_clean}.png"
        )

        plt.savefig(
            file_path,
            bbox_inches="tight"
        )

        plt.close()

        print(f"📈 Guardado: {file_path}")
    """
    def plot_all_model_forecasts(self,df_account,model_results,future_dates,account_name):
        plt.figure(figsize=(12, 6))

        account_name_clean = self._sanitize_name(account_name)

        # Histórico
        plt.plot(
            df_account["Fecha"],
            df_account["Valor"],
            linewidth=3,
            label="Histórico"
        )

        # Forecast de cada modelo
        for model_name, res in model_results.items():

            model = res.get("model")

            if model is None:
                continue

            try:
                # últimas features
                last_X = df_account.select_dtypes(include=["number"]).drop(
                    columns=["Valor"]
                ).iloc[-1:].copy()

                preds = []

                for _ in range(len(future_dates)):
                    pred = model.predict(last_X)[0]
                    preds.append(pred)

                    # actualizar lags
                    if "lag_3" in last_X.columns:
                        last_X["lag_3"] = last_X["lag_2"]
                        last_X["lag_2"] = last_X["lag_1"]
                        last_X["lag_1"] = pred

                plt.plot(
                    future_dates,
                    preds,
                    linestyle="--",
                    label=f"{model_name}"
                )

            except Exception as e:
                print(f"Error con {model_name}: {e}")

        plt.title(f"Forecast por Modelo - {account_name}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        file_path = f"{self.output_dir}/all_forecasts_{account_name_clean}.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        print(f"📈 Guardado: {file_path}")
    """