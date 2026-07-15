from src.data.preprocessor import FinancialPreprocessor
from src.data.splitter import AccountSplitter
from src.models.model_factory import ModelFactory
from src.models.trainer import AccountModelTrainer
from src.forecasting.forecast_engine import ForecastEngine
from src.forecasting.hierarchical import HierarchicalForecast
from src.utils.date_utils import generate_future_dates
from config.config import HIERARCHY, FORECAST_STEPS, USE_ACCOUNT_DEPENDENCIES
from src.preprocessing.account_dependency_builder import AccountDependencyBuilder
from config.account_dependencies import ACCOUNT_DEPENDENCIES

class FinancialForecastSystem:

    def __init__(self, df):
        self.df = df

    def run(self):
        #cuentas = ["ACTIVO","Disponibilidades","Inversiones en valores","Deudores por reporto","Cartera de crédito vigente","Créditos comerciales","Créditos de consumo","Créditos a la vivienda"]
        cuentas = ["ACTIVO","Inversiones en valores"]


        # 🔹 1. Preprocesamiento
        prep = FinancialPreprocessor(self.df)
        df_clean = prep.clean_data()
        df_long = prep.to_long()

        if USE_ACCOUNT_DEPENDENCIES:
            dependency_builder = AccountDependencyBuilder(df_long)


        df_long.to_csv("./data/processed/procesado.csv", index=False)

        #Filtrar para solo quedarme con uno 
        df_long = df_long[df_long["BALANCE GENERAL"].isin(cuentas)]
        #df_long = df_long[df_long["BALANCE GENERAL"] == "Inversiones en valores"]

        splitter = AccountSplitter(df_long)
        accounts = splitter.get_accounts()

        # 🔹 2. Entrenamiento
        models = ModelFactory.get_models()
        trainer = AccountModelTrainer(models)

        all_results = {}

        """
        # primera versión sin las dependencias, podría manerala como un if else 
        for acc in accounts:
            df_acc = splitter.get_account_df(acc)
            all_results[acc] = trainer.train_account(df_acc)
        """
        for acc in accounts:
            print(USE_ACCOUNT_DEPENDENCIES,"  ",acc,"  ",ACCOUNT_DEPENDENCIES)
            if (USE_ACCOUNT_DEPENDENCIES and acc in ACCOUNT_DEPENDENCIES):
                print("🔗 Usando cuentas dependientes para:",acc)
                df_acc = dependency_builder.build(account_id)
            else:
                df_acc = splitter.get_account_df(acc)
                if acc == "Inversiones en valores":
                    print(df_acc.head())
                    print(df_acc.columns)                
                #all_results[acc] = trainer.train_account(df_acc)
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