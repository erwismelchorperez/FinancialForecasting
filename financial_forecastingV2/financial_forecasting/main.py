import pandas as pd

from src.pipeline.forecasting_system import FinancialForecastSystem
from src.utils.results_saver import ResultsSaver
from src.visualization.plotter import ForecastPlotter
from src.data.splitter import AccountSplitter


def main():

    # ===============================
    # 1. CARGA DE DATOS
    # ===============================
    print("📥 Cargando datos...")
    df = pd.read_excel("data/raw/Tzaulan.xlsx")

    # ===============================
    # 2. INICIALIZAR SISTEMA
    # ===============================
    print("⚙️ Inicializando sistema de forecasting...")
    system = FinancialForecastSystem(df)

    # ===============================
    # 3. EJECUTAR PIPELINE
    # ===============================
    print("🚀 Ejecutando forecasting...")
    forecasts, future_dates = system.run()

    # ⚠️ IMPORTANTE:
    # Asegúrate que tu system.run() también retorna:
    # all_results, df_long
    # Si no, ajustamos después 👇
    try:
        all_results = system.all_results
        df_long = system.df_long
    except AttributeError:
        raise ValueError(
            "❌ system.run() debe guardar all_results y df_long como atributos:\n"
            "self.all_results = all_results\n"
            "self.df_long = df_long"
        )

    # ===============================
    # 4. GUARDAR RESULTADOS
    # ===============================
    print("💾 Guardando resultados...")
    saver = ResultsSaver()
    saver.save_metrics(all_results)
    saver.save_forecasts(forecasts, future_dates)
    
    # ===============================
    # 5. IMPRIMIR FORECAST
    # ===============================
    """
    print("\n📊 RESULTADOS DE FORECAST:\n")

    for acc, values in forecasts.items():
        print(f"\nCuenta: {acc}")
        for date, val in zip(future_dates, values):
            print(date.strftime("%Y-%m"), round(val, 2))
    """
    # ===============================
    # 6. GRÁFICAS
    # ===============================
    print("\n📈 Generando gráficas...")
    plotter = ForecastPlotter()
    splitter = AccountSplitter(df_long)
    for acc in system.all_results.keys():

        # 🔹 métricas (test vs pred)
        #plotter.plot_test_vs_pred(system.all_results[acc], acc)
        df_acc = splitter.get_account_df(acc)
        plotter.plot_test_vs_pred(all_results[acc], df_acc, acc)

        # 🔹 forecast
        preds = forecasts[acc]
        df_acc = system.df_long[system.df_long["BALANCE GENERAL"] == acc]

        plotter.plot_forecast(df_acc, preds, future_dates, acc)

        # NUEVA gráfica: todos los modelos
        plotter.plot_all_model_forecasts(
            df_acc,
            all_results[acc],
            future_dates,
            acc
        )
    """
    for acc, model_results in all_results.items():

        print(f"📊 Graficando cuenta: {acc}")

        df_account = df_long[df_long["BALANCE GENERAL"] == acc]

        # 1. Comparación modelos (test vs pred)
        plotter.plot_test_vs_pred(model_results, acc)

        # 2. Forecast final
        if acc in forecasts:
            plotter.plot_forecast(
                df_account,
                forecasts[acc],
                future_dates,
                acc
            )

        # 👉 SOLO UNA CUENTA (para no saturarte)
        break
    """
    print("\n✅ Proceso completado exitosamente")
    

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()