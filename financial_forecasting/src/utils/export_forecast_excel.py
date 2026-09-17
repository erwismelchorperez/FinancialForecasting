# src/utils/export_forecast_excel.py

import pandas as pd
import numpy as np
from datetime import datetime
import os


class ForecastExcelExporter:
    """Exporta resultados a Excel con los 3 mejores modelos como columnas."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export(self, original_df, forecasts, results, future_dates, top3_forecasts=None, df_long=None, filename=None):
        """
        Exporta a Excel con formato original: fechas como columnas, cuentas como filas.
        Los 3 mejores modelos aparecen como columnas adicionales.
        """
        if filename is None:
            filename = f"forecast_original_format_{self.timestamp}.xlsx"
        
        print(f"   📁 Generando Excel en formato original: {filename}")
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # ============================================================
                # 1. HOJA: Datos Históricos + Forecast (Formato Original)
                # ============================================================
                self._write_historical_with_forecast(writer, df_long, top3_forecasts, future_dates)
                
                # ============================================================
                # 2. HOJA: Top 3 Modelos (métricas)
                # ============================================================
                self._write_top3_models(writer, results)
            
            print(f"   ✅ Excel guardado: {filename}")
            return filename
            
        except Exception as e:
            print(f"   ❌ Error guardando Excel: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _write_historical_with_forecast(self, writer, df_long, top3_forecasts, future_dates):
        """
        Escribe los datos históricos + forecast de los 3 modelos.
        Los 3 modelos aparecen como columnas adicionales.
        """
        if df_long is None or df_long.empty:
            print("   ⚠️ No hay datos históricos para exportar")
            return
        
        print("   📊 Generando hoja: Histórico + Top 3 Forecasts...")
        
        # ============================================================
        # 1. OBTENER DATOS HISTÓRICOS POR CUENTA
        # ============================================================
        # Pivot para tener cuentas como filas y fechas como columnas
        df_pivot = df_long.pivot_table(
            index=['account_id', 'BALANCE GENERAL'],
            columns='Fecha',
            values='Valor'
        ).reset_index()
        
        # Renombrar columnas
        df_pivot = df_pivot.rename(columns={
            'account_id': 'CUENTA_ID',
            'BALANCE GENERAL': 'BALANCE GENERAL'
        })
        
        # Ordenar fechas
        date_cols = [col for col in df_pivot.columns if col not in ['CUENTA_ID', 'BALANCE GENERAL']]
        date_cols = sorted(date_cols)
        
        # ============================================================
        # 2. AGREGAR FORECASTS DE LOS 3 MEJORES MODELOS
        # ============================================================
        if top3_forecasts and future_dates is not None:
            # Crear nombres de columnas para los forecasts
            forecast_cols = []
            
            # Obtener los modelos para cada cuenta
            for account_id, models in top3_forecasts.items():
                # Encontrar la fila de esta cuenta
                mask = df_pivot['CUENTA_ID'] == account_id
                if not mask.any():
                    continue
                
                # Para cada modelo, agregar sus predicciones como columnas
                for model_key, predictions in models.items():
                    if predictions:
                        col_name = f"Forecast_{model_key}"
                        if col_name not in forecast_cols:
                            forecast_cols.append(col_name)
                        
                        # Crear valores de forecast para cada fecha
                        forecast_values = {}
                        for i, date in enumerate(future_dates[:len(predictions)]):
                            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                            forecast_values[date_str] = float(predictions[i]) if i < len(predictions) else None
                        
                        # Agregar los valores al DataFrame en las columnas correspondientes
                        for date_str, value in forecast_values.items():
                            # Buscar la columna de fecha en el DataFrame
                            # La fecha puede estar como datetime o como string
                            found = False
                            for col in df_pivot.columns:
                                if hasattr(col, 'strftime') and col.strftime('%Y-%m-%d') == date_str:
                                    # Si la columna existe (fecha futura), agregar el valor
                                    df_pivot.loc[mask, col] = value
                                    found = True
                                    break
                            
                            # Si no se encontró la columna, crearla
                            if not found:
                                # Convertir string a datetime para la columna
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                # Crear la columna si no existe
                                if date_obj not in df_pivot.columns:
                                    df_pivot[date_obj] = None
                                df_pivot.loc[mask, date_obj] = value
                        
                        # También agregar el nombre del modelo como información adicional
                        # Podemos agregar una fila adicional con el nombre del modelo
                        # o agregar una columna con el modelo
                        
        # ============================================================
        # 3. ORDENAR COLUMNAS
        # ============================================================
        # Obtener todas las columnas
        all_cols = list(df_pivot.columns)
        
        # Separar columnas de fecha (incluyendo las nuevas)
        date_cols = [col for col in all_cols if col not in ['CUENTA_ID', 'BALANCE GENERAL']]
        date_cols = sorted(date_cols, key=lambda x: x if not hasattr(x, 'strftime') else x)
        
        # Reordenar: ID, Nombre, fechas...
        cols = ['CUENTA_ID', 'BALANCE GENERAL'] + date_cols
        cols = [col for col in cols if col in df_pivot.columns]
        df_pivot = df_pivot[cols]
        
        # ============================================================
        # 4. FORMATEAR NOMBRES DE COLUMNAS
        # ============================================================
        new_cols = []
        for col in df_pivot.columns:
            if col in ['CUENTA_ID', 'BALANCE GENERAL']:
                new_cols.append(col)
            elif hasattr(col, 'strftime'):
                new_cols.append(col.strftime('%b-%y').lower())
            else:
                new_cols.append(str(col))
        
        df_pivot.columns = new_cols
        
        # Guardar
        df_pivot.to_excel(writer, sheet_name="Historico_Forecast", index=False)
        print(f"   ✅ Histórico + Forecast guardado: {len(df_pivot)} filas, {len(df_pivot.columns)} columnas")
    
    def _write_top3_models(self, writer, results):
        """Hoja con los 3 mejores modelos y sus métricas."""
        rows = []
        for account_id, models in results.items():
            if not models:
                continue
            
            valid_models = {k: v for k, v in models.items() if v is not None and 'metrics' in v}
            if not valid_models:
                continue
            
            sorted_models = sorted(
                valid_models.items(),
                key=lambda x: x[1]["metrics"].get("mape", float('inf'))
            )
            
            top_3 = sorted_models[:3]
            for i, (model_name, model_data) in enumerate(top_3, 1):
                metrics = model_data.get("metrics", {})
                rows.append({
                    "Cuenta": str(account_id),
                    "Ranking": i,
                    "Modelo": model_name,
                    "MAE": float(metrics.get("mae", float('inf'))),
                    "RMSE": float(metrics.get("rmse", float('inf'))),
                    "MAPE_%": float(metrics.get("mape", float('inf')))
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="Top3_Modelos", index=False)
            print(f"   ✅ Top 3 modelos guardados: {len(df)} filas")