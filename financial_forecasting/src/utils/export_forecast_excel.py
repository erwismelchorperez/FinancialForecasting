import pandas as pd
import numpy as np
from src.utils.model_selector import ModelSelector
from datetime import datetime

class ForecastExcelExporter:

    def __init__(self, output_path="outputs/forecast_final.xlsx"):
        self.output_path = output_path

    def export(self,original_df,forecasts,results,future_dates):

        final_df = original_df.copy()

        summary = []

        # columnas futuras
        future_cols = [
            d.strftime("%b-%y")
            for d in future_dates
        ]

        # -------------------------
        # score compuesto
        # -------------------------
        def model_score(metrics):

            rmse = metrics.get(
                "rmse",
                float("inf")
            )

            mape = metrics.get(
                "mape",
                float("inf")
            )

            transition = metrics.get(
                "transition_error",
                float("inf")
            )

            return (
                0.40 * rmse
                +
                0.30 * mape
                +
                0.30 * transition
            )

        # ==========================
        # recorrer cuentas
        # ==========================
        for account, model_data in forecasts.items():

            if (
                account
                not in final_df["BALANCE GENERAL"].values
            ):
                continue

            # modelos válidos
            valid = {}
            #print("     account         ",account)
            #print("     results         ",results)
            for model_name, res in results[account].items():

                if res is None:
                    continue

                metrics = res.get(
                    "metrics"
                )

                if metrics is None:
                    continue

                valid[
                    model_name
                ] = res

            if len(valid) == 0:
                continue

            # -------------------------
            # elegir mejor
            # -------------------------
            best_model = min(
                valid,
                key=lambda m:
                model_score(
                    valid[m]["metrics"]
                )
            )

            print(
                f"\nCuenta: {account}"
            )

            for m, r in valid.items():

                metrics = r["metrics"]

                score = model_score(
                    metrics
                )

                print(
                    f"{m:<10}"
                    f" RMSE={metrics.get('rmse',0):,.2f}"
                    f" MAE={metrics.get('mae',0):,.2f}"
                    f" MAPE={metrics.get('mape',0):.2f}"
                    f" TRANS={metrics.get('transition_error',0):,.2f}"
                    f" SCORE={score:,.2f}"
                )

            print(
                f"🏆 Mejor → {best_model}"
            )

            preds = model_data.get(
                best_model
            )

            if (
                preds is None
                or len(preds) == 0
            ):
                continue

            # localizar fila
            idx = (
                final_df[
                    final_df[
                        "BALANCE GENERAL"
                    ]
                    ==
                    account
                ]
                .index[0]
            )

            # insertar forecast
            for col, value in zip(
                future_cols,
                preds
            ):

                final_df.loc[
                    idx,
                    col
                ] = float(value)

            metrics = valid[
                best_model
            ][
                "metrics"
            ]

            summary.append({

                "Cuenta":
                account,

                "Modelo":
                best_model,

                "MAE":
                metrics.get(
                    "mae"
                ),

                "RMSE":
                metrics.get(
                    "rmse"
                ),

                "MAPE":
                metrics.get(
                    "mape"
                ),

                "Transition_Error":
                metrics.get(
                    "transition_error"
                ),

                "Score":
                model_score(
                    metrics
                )

            })

        # ==========================
        # exportar
        # ==========================

        metrics_df = pd.DataFrame(summary)
        new_columns = []

        for col in final_df.columns:
            if type(col) == datetime:
                new_columns.append(col.strftime("%b-%y"))
            else:
                new_columns.append(col)

        final_df.columns = new_columns
        with pd.ExcelWriter(
            self.output_path,
            engine="openpyxl"
        ) as writer:

            final_df.to_excel(
                writer,
                sheet_name="forecast",
                index=False
            )

            metrics_df.to_excel(
                writer,
                sheet_name="best_models",
                index=False
            )

        print(
            f"\n✅ Excel generado: {self.output_path}"
        )