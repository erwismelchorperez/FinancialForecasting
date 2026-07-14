import pandas as pd
import os

class ResultsSaver:

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_metrics(self, all_results):
        rows = []

        for account, models in all_results.items():
            for model_name, res in models.items():
                metrics = res.get("metrics", {})

                rows.append({
                    "Cuenta": account,
                    "Modelo": model_name,
                    "MAE": metrics.get("mae"),
                    "RMSE": metrics.get("rmse")
                })

        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/metrics.csv", index=False)
        print("✅ Métricas guardadas en metrics.csv")
    """
    def save_forecasts(self, forecasts, future_dates):

        rows = []

        for account, preds in forecasts.items():
            for date, value in zip(future_dates, preds):
                print("value            ",forecasts[account][value],"\n")
                print("value            ",forecasts[account])
                rows.append({
                    "BALANCE GENERAL": account,
                    "Fecha": date,
                    "Forecast": forecasts[account][value]#float(value)
                })

        df = pd.DataFrame(rows)

        # ==============================
        # 🔥 PIVOT A FORMATO ANCHO
        # ==============================
        df_wide = df.pivot(
            index="BALANCE GENERAL",
            columns="Fecha",
            values="Forecast"
        )

        # ordenar columnas
        df_wide = df_wide.sort_index(axis=1)

        # formato tipo ene-25
        df_wide.columns = df_wide.columns.strftime("%b-%y")

        # reset index
        df_wide = df_wide.reset_index()

        # guardar
        df_wide.to_csv(f"{self.output_dir}/forecasts_wide.csv", index=False)

        print("✅ Forecasts guardados en formato ancho (wide)")
    """
    def save_forecasts(self, forecasts, future_dates):

        os.makedirs(self.output_dir, exist_ok=True)

        # 🔹 obtener lista de modelos (desde la primera cuenta)
        first_account = next(iter(forecasts))
        model_names = forecasts[first_account].keys()

        for model_name in model_names:

            rows = []

            for account, model_dict in forecasts.items():

                preds = model_dict.get(model_name)

                if preds is None:
                    continue

                for date, value in zip(future_dates, preds):
                    rows.append({
                        "BALANCE GENERAL": account,
                        "Fecha": date,
                        "Forecast": float(value)
                    })

            if len(rows) == 0:
                continue

            df = pd.DataFrame(rows)

            # 🔥 FORMATO ANCHO (como dataset original)
            df_wide = df.pivot(
                index="BALANCE GENERAL",
                columns="Fecha",
                values="Forecast"
            )

            # ordenar fechas
            df_wide = df_wide.sort_index(axis=1)

            # formato tipo ene-25
            df_wide.columns = df_wide.columns.strftime("%b-%y")

            df_wide = df_wide.reset_index()

            # 🔹 guardar archivo por modelo
            file_path = f"{self.output_dir}/forecast_{model_name}.csv"
            df_wide.to_csv(file_path, index=False)

            print(f"✅ Forecast guardado para {model_name}: {file_path}")