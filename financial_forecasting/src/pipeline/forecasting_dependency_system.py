# src/pipeline/forecasting_dependency_system.py

from src.data.dependency_preprocessor import DependencyPreprocessor
from src.models.model_factory import ModelFactory
from src.models.trainer_factory import TrainerFactory
from src.utils.date_utils import generate_future_dates
from config.config import FORECAST_STEPS
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FinancialForecastDependencySystem:

    def __init__(self, df):
        self.df = df
        self.all_results = {}
        self.df_long = None
        self.forecasts = {}
        self.future_dates = None
        self.retrained_models = {}

    def run(self):
        # ============================================================
        # 1. Preprocesamiento
        # ============================================================
        print("\n🔧 Preprocesando datos...")
        prep = DependencyPreprocessor(self.df)
        df_clean = prep.clean_data()
        self.df_long = prep.to_long()

        self.df_long.to_csv("./data/processed/procesado.csv", index=False)
        print(f"   ✅ Datos procesados: {len(self.df_long)} registros")

        # ============================================================
        # 2. Configurar modelos
        # ============================================================
        print("\n🔧 Configurando modelos...")
        models = ModelFactory.get_models()
        print(f"   ✅ {len(models)} modelos disponibles")

        all_accounts = prep.get_accounts()
        print(f"   📊 {len(all_accounts)} cuentas encontradas")

        # ============================================================
        # 3. Entrenar modelos (con split train/test para evaluación)
        # ============================================================
        print("\n🎯 Entrenando modelos (con validación)...")
        
        for account_id in all_accounts:
            print(f"\n{'='*50}")
            account_info = prep.get_account_info(account_id)
            account_name = account_info['name'] if account_info else f"Cuenta_{account_id}"
            print(f"📊 Cuenta: {account_name} (ID: {account_id})")
            
            try:
                df_acc = prep.build(account_id)
                print(f"   📊 Dataset: {df_acc.shape[0]} filas, {df_acc.shape[1]} columnas")
                
                trainer = TrainerFactory.get_trainer(account_id, models)
                results = trainer.train_account(df_acc)
                
                if results:
                    self.all_results[account_id] = results
                    
                    valid_models = {k: v for k, v in results.items() if v is not None}
                    if valid_models:
                        sorted_models = sorted(
                            valid_models.items(),
                            key=lambda x: x[1]["metrics"]["mape"]
                        )
                        print(f"   ✅ Entrenamiento exitoso")
                        print(f"   📊 Top 3 modelos (por MAPE):")
                        for i, (name, data) in enumerate(sorted_models[:3], 1):
                            print(f"      {i}. {name}: MAPE = {data['metrics']['mape']:.2f}%")
                            print(f"         MAE = {data['metrics']['mae']:,.2f}")
                else:
                    print(f"   ⚠️ No se pudo entrenar ningún modelo")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
            #if account_id == 5:
                #break

        print(f"\n📊 Resumen: {len(self.all_results)} cuentas entrenadas")

        # ============================================================
        # 4. Seleccionar los 3 mejores modelos y USAR SUS PREDICCIONES
        # ============================================================
        print("\n🎯 Seleccionando los 3 mejores modelos y usando sus predicciones...")
        
        temp_forecasts = {}
        
        for account_id, results in self.all_results.items():
            if not results:
                continue
            
            print(f"\n📊 Cuenta {account_id}:")
            
            # Filtrar modelos válidos y ordenar por MAPE
            valid_models = {k: v for k, v in results.items() if v is not None}
            if not valid_models:
                continue
            
            sorted_models = sorted(
                valid_models.items(),
                key=lambda x: x[1]["metrics"]["mape"]
            )
            
            # Seleccionar top 3
            top_3 = sorted_models[:3]
            print(f"   📊 Top 3 modelos (por MAPE):")
            for i, (name, data) in enumerate(top_3, 1):
                print(f"      {i}. {name}: MAPE = {data['metrics']['mape']:.2f}%")
            
            # ============================================================
            # USAR LAS PREDICCIONES QUE YA EXISTEN EN results
            # ============================================================
            all_predictions = []
            
            for model_name, model_data in top_3:
                # Obtener las predicciones que ya se generaron durante el entrenamiento
                predictions = model_data.get('metrics', {}).get('y_pred')
                
                if predictions is not None and len(predictions) > 0:
                    # Convertir a lista si es necesario
                    if hasattr(predictions, 'tolist'):
                        predictions = predictions.tolist()
                    elif isinstance(predictions, np.ndarray):
                        predictions = predictions.tolist()
                    
                    print(f"      ✅ {model_name}: {len(predictions)} predicciones")
                    print(f"         Primeros 3 valores: {predictions[:3]}")
                    
                    # Verificar si son constantes
                    if len(set(predictions)) == 1:
                        print(f"         ⚠️ ADVERTENCIA: Predicciones constantes: {predictions[0]}")
                    else:
                        print(f"         ✅ Predicciones VARIABLES")
                    
                    all_predictions.append(predictions)
                else:
                    print(f"      ⚠️ {model_name}: No tiene predicciones disponibles")
            
            # ============================================================
            # Generar forecast como PROMEDIO de los 3 modelos
            # ============================================================
            if all_predictions:
                # Asegurar que todas las predicciones tengan la misma longitud
                min_len = min(len(p) for p in all_predictions)
                all_predictions = [p[:min_len] for p in all_predictions]
                
                # Calcular promedio
                avg_forecast = np.mean(all_predictions, axis=0).tolist()
                temp_forecasts[account_id] = avg_forecast
                
                print(f"   ✅ Forecast generado (promedio de {len(all_predictions)} modelos)")
                print(f"      Primeros 3 valores: {avg_forecast[:3]}")
                print(f"      Últimos 3 valores: {avg_forecast[-3:]}")
                
                if len(set(avg_forecast)) == 1:
                    print(f"   ⚠️ ADVERTENCIA: El forecast promedio es CONSTANTE: {avg_forecast[0]}")
                else:
                    print(f"   ✅ Forecast VARIABLE (correcto)")
            else:
                print(f"   ⚠️ No se pudo generar forecast para cuenta {account_id}")

            # ============================================================
            # FILTRO: Si account_id == 5, terminar el proceso
            # ============================================================
            if account_id == 5:
                print(f"\n{'='*60}")
                print(f"🎯 CUENTA 5 ENCONTRADA - Finalizando proceso...")
                print(f"{'='*60}")
                break

        # Asignar los forecasts generados
        self.forecasts = temp_forecasts

        # ============================================================
        # 5. Generar fechas futuras
        # ============================================================
        print("\n📅 Generando fechas futuras...")
        last_date = self.df_long["Fecha"].max()
        self.future_dates = generate_future_dates(last_date, FORECAST_STEPS)
        print(f"   ✅ Fechas: {self.future_dates[0]} a {self.future_dates[-1]}")

        # ============================================================
        # DIAGNÓSTICO FINAL
        # ============================================================
        print("\n" + "="*60)
        print("🔍 DIAGNÓSTICO DE FORECASTS")
        print("="*60)

        for account_id, forecast in self.forecasts.items():
            print(f"\n📊 Cuenta {account_id}:")
            print(f"   Forecast: {len(forecast)} valores")
            print(f"   Primeros 5 valores: {forecast[:5]}")
            print(f"   Últimos 5 valores: {forecast[-5:]}")
            
            if len(set(forecast)) == 1:
                print(f"   ⚠️ TODOS LOS VALORES SON IGUALES: {forecast[0]}")
            else:
                print(f"   ✅ Forecast VARIABLE (correcto)")

        print(f"\n✅ Proceso completado")
        print(f"   📊 {len(self.forecasts)} forecasts generados")

        # En forecasting_dependency_system.py - Agregar al final del método run()
        # ============================================================
        # GUARDAR TOP 3 FORECASTS PARA EXPORTAR
        # ============================================================
        self.top3_forecasts = {}
        
        for account_id, results in self.all_results.items():
            if not results:
                continue
            
            valid_models = {k: v for k, v in results.items() if v is not None and 'metrics' in v}
            if not valid_models:
                continue
            
            sorted_models = sorted(
                valid_models.items(),
                key=lambda x: x[1]["metrics"].get("mape", float('inf'))
            )
            
            top_3 = sorted_models[:3]
            
            if top_3:
                self.top3_forecasts[account_id] = {}
                for i, (model_name, model_data) in enumerate(top_3, 1):
                    metrics = model_data.get("metrics", {})
                    y_pred = metrics.get("y_pred")
                    
                    if y_pred is not None:
                        if hasattr(y_pred, 'tolist'):
                            y_pred = y_pred.tolist()
                        elif isinstance(y_pred, np.ndarray):
                            y_pred = y_pred.tolist()
                        
                        self.top3_forecasts[account_id][f"Top{i}_{model_name}"] = y_pred
                        print(f"   ✅ Top {i}: {model_name} - {len(y_pred)} predicciones guardadas")
        
        return self.all_results, self.df_long

    