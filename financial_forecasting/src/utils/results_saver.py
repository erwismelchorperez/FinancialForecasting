# src/utils/results_saver.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class ResultsSaver:
    """Guarda los resultados en formato JSON y CSV."""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear carpetas
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.forecasts_dir = os.path.join(output_dir, "forecasts")
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.forecasts_dir, exist_ok=True)
    
    def save_metrics(self, all_results):
        """Guarda las métricas de todos los modelos."""
        if not all_results:
            print("⚠️ No hay resultados para guardar")
            return
        
        # Guardar en JSON
        json_file = os.path.join(self.metrics_dir, f"metrics_{self.timestamp}.json")
        
        # Convertir arrays numpy a listas para JSON
        serializable_results = {}
        for account_id, models in all_results.items():
            serializable_results[str(account_id)] = {}
            for model_name, model_data in models.items():
                if model_data and isinstance(model_data, dict):
                    metrics = model_data.get("metrics", {})
                    serializable_results[str(account_id)][model_name] = {
                        "mae": float(metrics.get("mae", float('inf'))),
                        "rmse": float(metrics.get("rmse", float('inf'))),
                        "mape": float(metrics.get("mape", float('inf'))),
                        "transition_error": float(metrics.get("transition_error", float('inf'))),
                        "y_pred": metrics.get("y_pred", []).tolist() if isinstance(metrics.get("y_pred"), np.ndarray) else [],
                        "y_test": metrics.get("y_test", []).tolist() if isinstance(metrics.get("y_test"), np.ndarray) else []
                    }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Métricas guardadas en: {json_file}")
        
        # Guardar resumen en CSV
        summary = []
        for account_id, models in all_results.items():
            for model_name, model_data in models.items():
                if model_data and isinstance(model_data, dict):
                    metrics = model_data.get("metrics", {})
                    summary.append({
                        "account_id": account_id,
                        "model": model_name,
                        "mae": metrics.get("mae", float('inf')),
                        "rmse": metrics.get("rmse", float('inf')),
                        "mape": metrics.get("mape", float('inf')),
                        "transition_error": metrics.get("transition_error", float('inf'))
                    })
        
        if summary:
            df_summary = pd.DataFrame(summary)
            csv_file = os.path.join(self.metrics_dir, f"metrics_summary_{self.timestamp}.csv")
            df_summary.to_csv(csv_file, index=False)
            print(f"   ✅ Resumen de métricas guardado en: {csv_file}")
    
    def save_forecasts(self, forecasts, future_dates):
        """
        Guarda los forecasts en formato CSV.
        
        Args:
            forecasts: Puede ser:
                - Dict con estructura {account_id: [predicciones]}
                - Dict con estructura {account_id: {model_name: [predicciones]}}
            future_dates: Lista de fechas futuras
        """
        if not forecasts or future_dates is None:
            print("⚠️ No hay forecasts para guardar")
            return
        
        print(f"\n📊 Guardando forecasts para {len(forecasts)} cuentas...")
        
        # Detectar estructura
        first_account = next(iter(forecasts))
        first_value = forecasts[first_account]
        
        # CORRECCIÓN: Verificar si es lista o dict
        if isinstance(first_value, list):
            # Estructura simple: {account_id: [predicciones]}
            print("   📋 Estructura: forecasts simples")
            self._save_forecasts_simple(forecasts, future_dates)
        elif isinstance(first_value, dict):
            # Estructura por modelos: {account_id: {model_name: [predicciones]}}
            print("   📋 Estructura: forecasts por modelo")
            self._save_forecasts_by_model(forecasts, future_dates)
        else:
            print(f"   ⚠️ Estructura no reconocida: {type(first_value)}")
            # Intentar guardar de forma genérica
            self._save_forecasts_generic(forecasts, future_dates)
    
    def _save_forecasts_simple(self, forecasts, future_dates):
        """Guarda forecasts cuando son {account_id: [predicciones]}"""
        rows = []
        
        for account_id, preds in forecasts.items():
            if not isinstance(preds, (list, np.ndarray)):
                print(f"   ⚠️ Predicciones para {account_id} no son iterables: {type(preds)}")
                continue
            
            # Alinear longitud
            min_len = min(len(preds), len(future_dates))
            for i in range(min_len):
                try:
                    value = float(preds[i])
                    rows.append({
                        "BALANCE GENERAL": str(account_id),
                        "Fecha": future_dates[i],
                        "Forecast": value
                    })
                except (ValueError, TypeError) as e:
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Formato ancho (pivot)
            try:
                df_wide = df.pivot(
                    index="BALANCE GENERAL",
                    columns="Fecha",
                    values="Forecast"
                )
                df_wide = df_wide.sort_index(axis=1)
                df_wide.columns = df_wide.columns.strftime("%b-%y")
                df_wide = df_wide.reset_index()
                
                file_path = os.path.join(self.forecasts_dir, f"forecast_{self.timestamp}.csv")
                df_wide.to_csv(file_path, index=False)
                print(f"   ✅ Forecasts guardados: {file_path}")
                
                # También guardar en formato largo
                long_file = os.path.join(self.forecasts_dir, f"forecast_long_{self.timestamp}.csv")
                df.to_csv(long_file, index=False)
                print(f"   ✅ Forecasts (formato largo) guardados: {long_file}")
                
            except Exception as e:
                print(f"   ❌ Error guardando forecast: {e}")
                # Fallback a formato largo
                file_path = os.path.join(self.forecasts_dir, f"forecast_long_{self.timestamp}.csv")
                df.to_csv(file_path, index=False)
                print(f"   ✅ Forecasts guardados (formato largo): {file_path}")
    
    def _save_forecasts_by_model(self, forecasts, future_dates):
        """Guarda forecasts cuando son {account_id: {model_name: [predicciones]}}"""
        # Obtener todos los modelos
        model_names = set()
        for account_data in forecasts.values():
            if isinstance(account_data, dict):
                model_names.update(account_data.keys())
        
        if not model_names:
            print("   ⚠️ No se encontraron modelos")
            return
        
        print(f"   Modelos encontrados: {list(model_names)}")
        
        for model_name in model_names:
            rows = []
            
            for account_id, model_dict in forecasts.items():
                if not isinstance(model_dict, dict):
                    continue
                
                preds = model_dict.get(model_name)
                if preds is None:
                    continue
                
                if not isinstance(preds, (list, np.ndarray)):
                    continue
                
                # Alinear longitud
                min_len = min(len(preds), len(future_dates))
                for i in range(min_len):
                    try:
                        value = float(preds[i])
                        rows.append({
                            "BALANCE GENERAL": str(account_id),
                            "Fecha": future_dates[i],
                            "Forecast": value
                        })
                    except (ValueError, TypeError):
                        continue
            
            if rows:
                df = pd.DataFrame(rows)
                
                try:
                    df_wide = df.pivot(
                        index="BALANCE GENERAL",
                        columns="Fecha",
                        values="Forecast"
                    )
                    df_wide = df_wide.sort_index(axis=1)
                    df_wide.columns = df_wide.columns.strftime("%b-%y")
                    df_wide = df_wide.reset_index()
                    
                    file_path = os.path.join(self.forecasts_dir, f"forecast_{model_name}_{self.timestamp}.csv")
                    df_wide.to_csv(file_path, index=False)
                    print(f"   ✅ Forecasts guardados para {model_name}: {file_path}")
                except Exception as e:
                    print(f"   ❌ Error guardando {model_name}: {e}")
                    # Fallback
                    file_path = os.path.join(self.forecasts_dir, f"forecast_{model_name}_long_{self.timestamp}.csv")
                    df.to_csv(file_path, index=False)
                    print(f"   ✅ Forecasts guardados (formato largo) para {model_name}: {file_path}")
    
    def _save_forecasts_generic(self, forecasts, future_dates):
        """Intenta guardar forecasts en cualquier formato."""
        try:
            rows = []
            for account_id, value in forecasts.items():
                if isinstance(value, (list, np.ndarray)):
                    for i, pred in enumerate(value[:len(future_dates)]):
                        try:
                            rows.append({
                                "account_id": str(account_id),
                                "fecha": future_dates[i],
                                "forecast": float(pred)
                            })
                        except:
                            continue
                else:
                    # Valor único
                    rows.append({
                        "account_id": str(account_id),
                        "fecha": future_dates[0] if future_dates else None,
                        "forecast": float(value) if value is not None else 0
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                file_path = os.path.join(self.forecasts_dir, f"forecast_generic_{self.timestamp}.csv")
                df.to_csv(file_path, index=False)
                print(f"   ✅ Forecasts guardados (formato genérico): {file_path}")
        except Exception as e:
            print(f"   ❌ Error guardando forecasts: {e}")