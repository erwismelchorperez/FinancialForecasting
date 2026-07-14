from src.data.preprocessor import FinancialPreprocessor
from src.data.splitter import AccountSplitter
from src.models.model_factory import ModelFactory
from src.models.trainer import AccountModelTrainer
from src.forecasting.forecast_engine import ForecastEngine
from src.forecasting.hierarchical import HierarchicalForecast
from src.utils.date_utils import generate_future_dates
from config.config import HIERARCHY, FORECAST_STEPS

class FinancialForecastSystem:

    def __init__(self, df):
        self.df = df

    def run(self):
        cuentas = ["Disponibilidades","Inversiones en valores"]
        # 🔹 1. Preprocesamiento
        prep = FinancialPreprocessor(self.df)
        df_clean = prep.clean_data()
        df_long = prep.to_long()

        df_long.to_csv("./data/processed/procesado.csv", index=False)

        #Filtrar para solo quedarme con uno 
        df_long = df_long[df_long["BALANCE GENERAL"].isin(cuentas)]
        #df_long = df_long[df_long["BALANCE GENERAL"] == "Inversiones en valores"]
        #df_long = df_long[df_long["BALANCE GENERAL"] == "Disponibilidades"]
        #print(df_long.info())
        splitter = AccountSplitter(df_long)
        accounts = splitter.get_accounts()

        # 🔹 2. Entrenamiento
        models = ModelFactory.get_models()
        trainer = AccountModelTrainer(models)

        all_results = {}

        for acc in accounts:
            df_acc = splitter.get_account_df(acc)
            all_results[acc] = trainer.train_account(df_acc)

        # 🔹 3. Forecast
        engine = ForecastEngine(HIERARCHY)
        forecasts = engine.forecast_all(df_long, all_results)
        print("forecasts        ", forecasts)
        # 🔹 4. Jerarquía (Bottom-Up)
        hierarchy_model = HierarchicalForecast(HIERARCHY)
        forecasts = hierarchy_model.bottom_up(forecasts)

        # 🔹 5. Fechas futuras
        last_date = df_long["Fecha"].max()
        future_dates = generate_future_dates(last_date, FORECAST_STEPS)

        # ✅ AGREGAR ESTO
        self.all_results = all_results
        self.df_long = df_long
        
        return forecasts, future_dates