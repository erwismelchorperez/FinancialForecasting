import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR

from xgboost import XGBRegressor

from src.preprocessing.loader import DataLoader
from src.preprocessing.transformer import FinancialTransformer
from src.preprocessing.feature_engineering import FeatureEngineering

from src.training.trainer_ml import TrainerML
from src.training.trainer_ts import TrainerTS

from src.training.evaluator import ModelEvaluator

from src.forecasting.ml_forecaster import MLForecaster
from src.forecasting.ts_forecaster import TSForecaster

from src.utils.saver import ForecastSaver

from src.visualization.plotter import ForecastPlotter

from config.settings import (
    FORECAST_STEPS
)

FILEPATH = "data/raw/Tzaulan.xlsx"


def main():

    # =========================
    # LOAD
    # =========================

    loader = DataLoader(FILEPATH)

    df = loader.load()

    # =========================
    # TRANSFORM
    # =========================

    transformer = FinancialTransformer()

    df_long = transformer.transform(df)

    # =========================
    # FEATURES
    # =========================

    fe = FeatureEngineering()

    df_features = fe.create_features(df_long)

    # =========================
    # MODELOS ML
    # =========================

    ml_models = {

        "RF": RandomForestRegressor(),

        "GB": GradientBoostingRegressor(),

        "SVR": SVR(),

        "XGB": XGBRegressor()
    }

    trainer_ml = TrainerML(ml_models)

    ts_trainer = TrainerTS()

    evaluator = ModelEvaluator()

    ml_forecaster = MLForecaster()

    ts_forecaster = TSForecaster()

    plotter = ForecastPlotter()

    forecasts = {}

    accounts = df_features[
        "BALANCE GENERAL"
    ].unique()

    for account in accounts:

        print(f"\nProcesando: {account}")

        df_acc = df_features[
            df_features["BALANCE GENERAL"] == account
        ].copy()

        # =====================
        # ML
        # =====================

        ml_results = trainer_ml.train(df_acc)

        # =====================
        # TS
        # =====================

        series = df_acc["Valor"].dropna()

        ts_results = {}

        try:

            ts_results["SARIMAX"] = (
                ts_trainer.train_sarimax(series)
            )

        except Exception as e:
            print("SARIMAX error:", e)

        try:

            ts_results["HoltWinters"] = (
                ts_trainer.train_holtwinters(series)
            )

        except Exception as e:
            print("HW error:", e)

        # =====================
        # COMBINAR RESULTADOS
        # =====================

        all_results = {}

        all_results.update(ml_results)

        all_results.update(ts_results)

        # =====================
        # MEJOR MODELO
        # =====================

        best_model_name = evaluator.select_best(
            all_results
        )

        print(
            "Mejor modelo:",
            best_model_name
        )

        best_result = all_results[
            best_model_name
        ]

        # =====================
        # FORECAST
        # =====================

        if best_model_name in [
            "SARIMAX",
            "HoltWinters"
        ]:

            preds = ts_forecaster.forecast(
                best_result["model"]
            )

        else:

            preds = ml_forecaster.forecast(
                best_result["model"],
                df_acc
            )

        forecasts[account] = preds

        # =====================
        # FUTURE DATES
        # =====================

        last_date = df_acc["Fecha"].max()

        future_dates = pd.date_range(
            start=last_date,
            periods=FORECAST_STEPS + 1,
            freq="MS"
        )[1:]

        # =====================
        # PLOT
        # =====================

        plotter.plot(
            df_acc,
            preds,
            future_dates,
            account
        )

    # =========================
    # SAVE
    # =========================

    saver = ForecastSaver(
        "outputs/forecasts"
    )

    saver.save(
        forecasts,
        future_dates
    )


if __name__ == "__main__":
    main()