# main_dependency.py

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import traceback

# Agregar el directorio src al path si es necesario
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.forecasting_dependency_system import FinancialForecastDependencySystem
from src.utils.results_saver import ResultsSaver
from src.visualization.plotter import ForecastPlotter
from src.data.splitter import AccountSplitter
from src.utils.export_forecast_excel import ForecastExcelExporter
from config.config import FORECAST_STEPS


def create_directories():
    """Crea todas las carpetas necesarias para el proyecto."""
    folders = [
        "data/raw",
        "data/processed",
        "data/forecasts",
        "results",
        "results/metrics",
        "results/forecasts",
        "results/plots",
        "results/excel",
        "logs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Carpeta asegurada: {folder}")


def load_data(file_path="data/raw/Historicos Tzaulan_v2.xlsx"):
    """Carga los datos desde un archivo Excel."""
    print(f"\n📥 Cargando datos desde: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            alternative_paths = [
                "Historicos Tzaulan_v2.xlsx",
                "../data/raw/Historicos Tzaulan_v2.xlsx",
                "./data/raw/Historicos Tzaulan_v2.xlsx"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Archivo no encontrado en ninguna ubicación esperada")
        
        df = pd.read_excel(file_path)
        print(f"   ✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"   📊 Columnas: {df.columns.tolist()[:5]}...")
        return df
        
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        print("   💡 Asegúrate de que el archivo exista en data/raw/")
        return None
    except Exception as e:
        print(f"   ❌ Error al cargar datos: {e}")
        traceback.print_exc()
        return None


def print_section(title, char="="):
    """Imprime una sección con formato."""
    print("\n" + char * 60)
    print(f" {title}")
    print(char * 60)


def summarize_results(all_results, forecasts, future_dates):
    """Genera un resumen de los resultados."""
    print_section("📊 RESUMEN DE RESULTADOS")
    
    print(f"\n📈 Cuentas procesadas: {len(all_results)}")
    
    # Modelos utilizados
    model_counts = {}
    model_mae = {}
    model_rmse = {}
    model_mape = {}
    
    for account_id, results in all_results.items():
        for model_name, model_data in results.items():
            if model_data and isinstance(model_data, dict):
                metrics = model_data.get("metrics", {})
                if metrics:
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1
                    
                    mae = metrics.get("mae", float('inf'))
                    rmse = metrics.get("rmse", float('inf'))
                    mape = metrics.get("mape", float('inf'))
                    
                    if model_name not in model_mae:
                        model_mae[model_name] = []
                        model_rmse[model_name] = []
                        model_mape[model_name] = []
                    
                    if mae != float('inf'):
                        model_mae[model_name].append(mae)
                    if rmse != float('inf'):
                        model_rmse[model_name].append(rmse)
                    if mape != float('inf'):
                        model_mape[model_name].append(mape)
    
    print("\n📊 Modelos utilizados:")
    for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        avg_mae = np.mean(model_mae[model_name]) if model_mae.get(model_name) else float('inf')
        avg_rmse = np.mean(model_rmse[model_name]) if model_rmse.get(model_name) else float('inf')
        avg_mape = np.mean(model_mape[model_name]) if model_mape.get(model_name) else float('inf')
        
        print(f"   - {model_name}:")
        print(f"       Cuentas: {count}")
        if avg_mae != float('inf'):
            print(f"       MAE promedio: {avg_mae:,.2f}")
        if avg_rmse != float('inf'):
            print(f"       RMSE promedio: {avg_rmse:,.2f}")
        if avg_mape != float('inf'):
            print(f"       MAPE promedio: {avg_mape:.2f}%")
    
    # Mejores modelos por cuenta
    print("\n🏆 Mejores modelos por cuenta (por MAPE):")
    best_models = {}
    for account_id, results in all_results.items():
        if results:
            valid_models = {k: v for k, v in results.items() if v is not None}
            if valid_models:
                best = min(valid_models.items(), 
                          key=lambda x: x[1]["metrics"].get("mape", float('inf')))
                best_models[best[0]] = best_models.get(best[0], 0) + 1
    
    for model_name, count in sorted(best_models.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {model_name}: {count} cuentas")
    
    # Forecasts
    if forecasts:
        print(f"\n🔮 Forecasts generados: {len(forecasts)} cuentas")
        if future_dates is not None:
            print(f"   📅 Período: {future_dates[0].strftime('%Y-%m-%d')} a {future_dates[-1].strftime('%Y-%m-%d')}")
            print(f"   📊 Pasos: {len(future_dates)} meses")
    
    print("\n📁 Archivos generados:")
    print("   - Métricas: results/metrics/")
    print("   - Forecasts: results/forecasts/")
    print("   - Gráficas: results/plots/")
    print("   - Excel: results/excel/")


def main():
    """Función principal del sistema de forecasting con dependencias."""
    
    # ============================================================
    # 1. Configuración inicial
    # ============================================================
    print_section("🔮 SISTEMA DE FORECASTING CON DEPENDENCIAS", "=")
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Crear carpetas necesarias
    print("\n📁 Creando estructura de carpetas...")
    create_directories()

    # ============================================================
    # 2. Cargar datos
    # ============================================================
    df = load_data()
    if df is None:
        print("\n❌ No se pudieron cargar los datos. Saliendo...")
        return

    # ============================================================
    # 3. Inicializar sistema
    # ============================================================
    print_section("⚙️ INICIALIZANDO SISTEMA")
    print("Creando instancia de FinancialForecastDependencySystem...")
    
    try:
        system = FinancialForecastDependencySystem(df)
        print("   ✅ Sistema inicializado correctamente")
    except Exception as e:
        print(f"   ❌ Error al inicializar sistema: {e}")
        traceback.print_exc()
        return

    # ============================================================
    # 4. Ejecutar forecasting
    # ============================================================
    print_section("🚀 EJECUTANDO FORECASTING")
    
    try:
        all_results, df_long = system.run()
        
        # Obtener forecasts del sistema
        forecasts = getattr(system, 'forecasts', {})
        future_dates = getattr(system, 'future_dates', None)
        retrained_models = getattr(system, 'retrained_models', {})
        
        print(f"\n   ✅ Forecasting completado exitosamente")
        print(f"      📊 Cuentas entrenadas: {len(all_results)}")
        print(f"      🔮 Forecasts generados: {len(forecasts)}")
        print(f"      🔄 Modelos reentrenados: {len(retrained_models)}")
        
        # Mostrar qué modelos se usaron para cada cuenta
        if retrained_models:
            print(f"\n   📊 Modelos utilizados para forecasts:")
            for account_id, models in retrained_models.items():
                if models:
                    model_names = list(models.keys())
                    print(f"      Cuenta {account_id}: {', '.join(model_names)}")
        
    except Exception as e:
        print(f"   ❌ Error en forecasting: {e}")
        traceback.print_exc()
        return

    # ============================================================
    # 5. Guardar resultados
    # ============================================================
    print_section("💾 GUARDANDO RESULTADOS")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saver = ResultsSaver()
        
        # 5.1 Guardar métricas
        print("\n📊 Guardando métricas...")
        saver.save_metrics(all_results)
        
        # 5.2 Guardar forecasts
        if forecasts and future_dates is not None:
            print("\n📊 Guardando forecasts...")
            saver.save_forecasts(forecasts, future_dates)
        else:
            print("\n   ⚠️ No hay forecasts para guardar")
        
        print("\n   ✅ Todos los resultados guardados correctamente")
        
    except Exception as e:
        print(f"   ❌ Error al guardar resultados: {e}")
        traceback.print_exc()

    # ============================================================
    # 6. Generar gráficas (SOLO PARA CUENTA 5)
    # ============================================================
    print_section("📈 GENERANDO GRÁFICAS (Cuenta 5)")

    try:
        plotter = ForecastPlotter()
        splitter = AccountSplitter(df_long)
        
        plots_dir = f"results/plots/{timestamp}"
        os.makedirs(plots_dir, exist_ok=True)
        print(f"📁 Carpeta de gráficas: {plots_dir}")
        
        total_plots = 0
        failed_plots = 0
        
        # SOLO procesar la cuenta 5
        accounts_to_plot = [5]
        
        for account_id in accounts_to_plot:
            if account_id not in all_results:
                print(f"   ⚠️ Cuenta {account_id} no tiene resultados")
                continue
                
            try:
                df_acc = splitter.get_account_df(account_id)
                if df_acc.empty:
                    print(f"   ⚠️ No hay datos para cuenta {account_id}")
                    continue
                
                account_name = df_long[df_long["account_id"] == account_id]["BALANCE GENERAL"].iloc[0] if not df_long[df_long["account_id"] == account_id].empty else f"Cuenta_{account_id}"
                title = f"{account_name} (ID: {account_id})"
                
                print(f"\n   📊 Generando gráficas para: {title}")
                
                # Gráfica 1: Test vs Predicción (Top 3 modelos)
                try:
                    plotter.plot_test_vs_pred(all_results[account_id], df_acc, title)
                    total_plots += 1
                    print(f"      ✅ Test vs Predicción (Top 3)")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    failed_plots += 1
                
                # Gráfica 2: Forecast (Top 3 modelos) - CORREGIDO
                if forecasts and account_id in forecasts:
                    try:
                        # PASAR results PARA QUE MUESTRE LOS 3 MODELOS
                        plotter.plot_forecast(
                            df_acc, 
                            forecasts[account_id], 
                            future_dates, 
                            title,
                            results=all_results[account_id]  # ← PASAR results
                        )
                        total_plots += 1
                        print(f"      ✅ Forecast (Top 3 modelos)")
                    except Exception as e:
                        print(f"      ❌ Error: {e}")
                        failed_plots += 1
                
                # Gráfica 3: Todos los modelos (destacando Top 3)
                try:
                    plotter.plot_all_model_forecasts(df_acc, all_results[account_id], future_dates, title)
                    total_plots += 1
                    print(f"      ✅ Todos los modelos (Top 3 destacados)")
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    failed_plots += 1
                    
            except Exception as e:
                print(f"   ❌ Error procesando cuenta {account_id}: {e}")
                failed_plots += 1
                traceback.print_exc()
        
        print(f"\n   ✅ Resumen de gráficas:")
        print(f"      Exitosas: {total_plots}")
        print(f"      Fallidas: {failed_plots}")
        
    except Exception as e:
        print(f"   ❌ Error general en generación de gráficas: {e}")
        traceback.print_exc()

    # main_dependency.py - Parte de exportación

    # ============================================================
    # 7. Exportar a Excel (Formato Original)
    # ============================================================
    print_section("📊 EXPORTANDO A EXCEL (FORMATO ORIGINAL)")
    
    try:
        from src.utils.export_forecast_excel import ForecastExcelExporter
        
        exporter = ForecastExcelExporter()
        
        if forecasts and future_dates is not None:
            excel_filename = f"results/excel/forecast_original_{timestamp}.xlsx"
            print(f"📁 Generando Excel: {excel_filename}")
            
            top3_forecasts = getattr(system, 'top3_forecasts', {})
            
            if top3_forecasts:
                print(f"   📊 Top 3 forecasts encontrados para {len(top3_forecasts)} cuentas")
            
            exporter.export(
                original_df=df,
                forecasts=forecasts,
                results=all_results,
                future_dates=future_dates,
                top3_forecasts=top3_forecasts,
                df_long=df_long,
                filename=excel_filename
            )
            print(f"   ✅ Exportación completada")
        else:
            print("   ⚠️ No hay forecasts para exportar")
            
    except Exception as e:
        print(f"   ❌ Error al exportar: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 8. Resumen final
    # ============================================================
    summarize_results(all_results, forecasts, future_dates)
    
    # ============================================================
    # 9. Finalizar
    # ============================================================
    print_section("✅ PROCESO COMPLETADO")
    print(f"🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📂 Resultados disponibles en:")
    print("   - results/metrics/     → Métricas de modelos")
    print("   - results/forecasts/   → Predicciones generadas")
    print("   - results/plots/       → Gráficas")
    print("   - results/excel/       → Reporte en Excel")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        traceback.print_exc()
        sys.exit(1)