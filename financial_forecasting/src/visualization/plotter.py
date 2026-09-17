# src/visualization/plotter.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime


class ForecastPlotter:
    """Genera gráficas de forecasts."""
    
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_top3_models(self, results):
        """
        Obtiene los 3 mejores modelos basados en MAPE.
        """
        if not results:
            return {}, []
        
        # Filtrar modelos válidos
        valid_models = {k: v for k, v in results.items() if v is not None and 'metrics' in v}
        if not valid_models:
            return {}, []
        
        # Ordenar por MAPE
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["metrics"].get("mape", float('inf'))
        )
        
        # Tomar top 3
        top_3 = sorted_models[:3]
        top_3_dict = dict(top_3)
        
        return top_3_dict, top_3
    
    def plot_test_vs_pred(self, results, df_acc, title, save=True):
        """
        Grafica Test vs Predicción mostrando SOLO los 3 mejores modelos.
        """
        if df_acc is None or df_acc.empty:
            print(f"   ⚠️ No hay datos para graficar: {title}")
            return
        
        if not results:
            print(f"   ⚠️ No hay resultados para graficar: {title}")
            return
        
        # Obtener top 3 modelos
        top_3_dict, top_3_list = self._get_top3_models(results)
        
        if not top_3_dict:
            print(f"   ⚠️ No hay modelos válidos para {title}")
            return
        
        print(f"\n   📊 Top 3 modelos para {title}:")
        for i, (name, data) in enumerate(top_3_list, 1):
            mape = data['metrics'].get('mape', 0)
            print(f"      {i}. {name}: MAPE = {mape:.2f}%")
        
        plt.figure(figsize=(14, 7))
        
        # ============================================================
        # 1. DATOS HISTÓRICOS
        # ============================================================
        plt.plot(df_acc["Fecha"], df_acc["Valor"], 
                label="Datos históricos", 
                color="blue", 
                alpha=0.5, 
                linewidth=1.5)
        
        # ============================================================
        # 2. VALORES REALES DE TEST (SOLO UNA VEZ)
        # ============================================================
        test_values = None
        test_dates = None
        
        for model_name, model_data in top_3_dict.items():
            metrics = model_data.get("metrics", {})
            y_test = metrics.get("y_test")
            
            if y_test is not None and len(y_test) > 0:
                n_test = len(y_test)
                if len(df_acc) >= n_test:
                    test_dates = df_acc["Fecha"].iloc[-n_test:].values
                    test_values = y_test
                    break
        
        if test_values is not None and test_dates is not None:
            plt.plot(test_dates, test_values, 
                    label="Valores reales (test)", 
                    color="green", 
                    marker='o', 
                    markersize=5,
                    linewidth=2,
                    linestyle='-',
                    alpha=0.8)
        
        # ============================================================
        # 3. PREDICCIONES DE LOS 3 MEJORES MODELOS
        # ============================================================
        colors = ['red', 'orange', 'purple']
        linestyles = ['--', '-.', ':']
        markers = ['s', '^', 'D']
        
        for idx, (model_name, model_data) in enumerate(top_3_dict.items()):
            metrics = model_data.get("metrics", {})
            y_pred = metrics.get("y_pred")
            
            if y_pred is not None and len(y_pred) > 0:
                if test_dates is not None and len(test_dates) == len(y_pred):
                    pred_dates = test_dates
                else:
                    n_pred = len(y_pred)
                    if len(df_acc) >= n_pred:
                        pred_dates = df_acc["Fecha"].iloc[-n_pred:].values
                    else:
                        pred_dates = df_acc["Fecha"].values[-n_pred:]
                
                color = colors[idx % len(colors)]
                linestyle = linestyles[idx % len(linestyles)]
                marker = markers[idx % len(markers)]
                
                mape = metrics.get('mape', 0)
                
                plt.plot(pred_dates, y_pred, 
                        label=f"{model_name} (MAPE: {mape:.2f}%)", 
                        color=color, 
                        linestyle=linestyle,
                        marker=marker,
                        markersize=5,
                        linewidth=2,
                        alpha=0.8)
        
        # ============================================================
        # 4. LÍNEA DE SEPARACIÓN
        # ============================================================
        if test_dates is not None and len(test_dates) > 0:
            split_date = test_dates[0]
            plt.axvline(x=split_date, 
                       color='gray', 
                       linestyle=':', 
                       alpha=0.7, 
                       linewidth=2,
                       label="Inicio de test")
        
        # ============================================================
        # 5. FORMATO DE LA GRÁFICA
        # ============================================================
        plt.title(f"Test vs Predicción (Top 3 modelos) - {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Valor", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/test_vs_pred_top3_{safe_title}_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Gráfica guardada: {filename}")
        else:
            plt.show()
    
    def plot_forecast(self, df_acc, forecast, future_dates, title, results=None, save=True):
        """
        Grafica el forecast MOSTRANDO LOS 3 MEJORES MODELOS.
        CORREGIDO: Ahora muestra los 3 modelos individualmente.
        """
        if df_acc is None or df_acc.empty:
            print(f"   ⚠️ No hay datos para graficar: {title}")
            return
        
        if forecast is None or future_dates is None:
            print(f"   ⚠️ No hay forecast para graficar: {title}")
            return
        
        plt.figure(figsize=(14, 7))
        
        # ============================================================
        # 1. DATOS HISTÓRICOS
        # ============================================================
        plt.plot(df_acc["Fecha"], df_acc["Valor"], 
                label="Datos históricos", 
                color="blue", 
                alpha=0.6, 
                linewidth=2)
        
        # ============================================================
        # 2. OBTENER LOS 3 MEJORES MODELOS (si results está disponible)
        # ============================================================
        if results:
            top_3_dict, top_3_list = self._get_top3_models(results)
            
            if top_3_dict:
                print(f"\n   📊 Forecast - Top 3 modelos para {title}:")
                for i, (name, data) in enumerate(top_3_list, 1):
                    mape = data['metrics'].get('mape', 0)
                    print(f"      {i}. {name}: MAPE = {mape:.2f}%")
                
                # ============================================================
                # 3. MOSTRAR LOS 3 MODELOS EN EL FORECAST
                # ============================================================
                colors = ['red', 'orange', 'purple']
                linestyles = ['--', '-.', ':']
                markers = ['s', '^', 'D']
                
                for idx, (model_name, model_data) in enumerate(top_3_dict.items()):
                    metrics = model_data.get("metrics", {})
                    y_pred = metrics.get("y_pred")
                    
                    if y_pred is not None and len(y_pred) > 0:
                        n_forecast = min(len(y_pred), len(future_dates))
                        if n_forecast > 0:
                            color = colors[idx % len(colors)]
                            linestyle = linestyles[idx % len(linestyles)]
                            marker = markers[idx % len(markers)]
                            mape = metrics.get('mape', 0)
                            
                            plt.plot(future_dates[:n_forecast], y_pred[:n_forecast], 
                                    label=f"{model_name} (MAPE: {mape:.2f}%)", 
                                    color=color, 
                                    linestyle=linestyle,
                                    marker=marker,
                                    markersize=5,
                                    linewidth=2,
                                    alpha=0.8)
        
        # ============================================================
        # 4. FORECAST PROMEDIO (línea más gruesa)
        # ============================================================
        n_forecast = min(len(forecast), len(future_dates))
        if n_forecast > 0:
            plt.plot(future_dates[:n_forecast], forecast[:n_forecast], 
                    label="Promedio Top 3", 
                    color="darkred", 
                    linestyle="-", 
                    marker='o', 
                    markersize=6,
                    linewidth=3,
                    alpha=0.9)
        
        # ============================================================
        # 5. LÍNEA DE SEPARACIÓN
        # ============================================================
        if not df_acc.empty:
            last_date = df_acc["Fecha"].iloc[-1]
            plt.axvline(x=last_date, 
                       color='gray', 
                       linestyle=':', 
                       alpha=0.7, 
                       linewidth=2,
                       label="Fin de datos históricos")
        
        # ============================================================
        # 6. FORMATO
        # ============================================================
        plt.title(f"Forecast - Top 3 modelos - {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Valor", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/forecast_top3_{safe_title}_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Gráfica guardada: {filename}")
        else:
            plt.show()
    
    def plot_all_model_forecasts(self, df_acc, results, future_dates, title, save=True):
        """
        Grafica TODOS los modelos pero DESTACA los 3 mejores.
        """
        if df_acc is None or df_acc.empty or not results:
            print(f"   ⚠️ No hay datos para graficar: {title}")
            return
        
        # Obtener top 3 modelos
        top_3_dict, top_3_list = self._get_top3_models(results)
        top_3_names = [name for name, _ in top_3_list]
        
        if not top_3_dict:
            print(f"   ⚠️ No hay modelos válidos para {title}")
            return
        
        print(f"\n   📊 Top 3 modelos destacados para {title}:")
        for i, (name, data) in enumerate(top_3_list, 1):
            mape = data['metrics'].get('mape', 0)
            print(f"      {i}. {name}: MAPE = {mape:.2f}%")
        
        plt.figure(figsize=(15, 8))
        
        # ============================================================
        # 1. DATOS HISTÓRICOS
        # ============================================================
        plt.plot(df_acc["Fecha"], df_acc["Valor"], 
                label="Datos históricos", 
                color="black", 
                alpha=0.5, 
                linewidth=2)
        
        # ============================================================
        # 2. VALORES REALES DE TEST (SOLO UNA VEZ)
        # ============================================================
        test_values = None
        test_dates = None
        
        for model_name, model_data in results.items():
            if model_data is None:
                continue
            metrics = model_data.get("metrics", {})
            y_test = metrics.get("y_test")
            
            if y_test is not None and len(y_test) > 0:
                n_test = len(y_test)
                if len(df_acc) >= n_test:
                    test_dates = df_acc["Fecha"].iloc[-n_test:].values
                    test_values = y_test
                    break
        
        if test_values is not None and test_dates is not None:
            plt.plot(test_dates, test_values, 
                    label="Valores reales (test)", 
                    color="green", 
                    marker='o', 
                    markersize=6,
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.9)
        
        # ============================================================
        # 3. FORECASTS DE TODOS LOS MODELOS
        #    - Los top 3 con líneas más gruesas y colores destacados
        #    - Los demás con líneas más delgadas y grises
        # ============================================================
        colors_top3 = ['red', 'blue', 'orange']
        colors_others = ['gray', 'lightgray', 'darkgray', 'silver', 'dimgray']
        linestyles_top3 = ['--', '-.', ':']
        
        # Primero graficar los que NO son top 3 (fondo)
        for model_name, model_data in results.items():
            if model_data is None or model_name in top_3_names:
                continue
            
            metrics = model_data.get("metrics", {})
            y_pred = metrics.get("y_pred")
            
            if y_pred is not None and future_dates is not None:
                n_forecast = min(len(y_pred), len(future_dates))
                if n_forecast > 0:
                    plt.plot(future_dates[:n_forecast], y_pred[:n_forecast], 
                            label=None, 
                            color='lightgray', 
                            linestyle=':',
                            linewidth=1,
                            alpha=0.3)
        
        # Luego graficar los top 3 (encima)
        for idx, (model_name, model_data) in enumerate(top_3_dict.items()):
            metrics = model_data.get("metrics", {})
            y_pred = metrics.get("y_pred")
            
            if y_pred is not None and future_dates is not None:
                n_forecast = min(len(y_pred), len(future_dates))
                if n_forecast > 0:
                    color = colors_top3[idx % len(colors_top3)]
                    linestyle = linestyles_top3[idx % len(linestyles_top3)]
                    mape = metrics.get('mape', 0)
                    
                    plt.plot(future_dates[:n_forecast], y_pred[:n_forecast], 
                            label=f"{model_name} (MAPE: {mape:.2f}%)", 
                            color=color, 
                            linestyle=linestyle,
                            marker='s' if idx % 2 == 0 else '^',
                            markersize=5,
                            linewidth=2.5,
                            alpha=0.9)
        
        # ============================================================
        # 4. LÍNEA DE SEPARACIÓN
        # ============================================================
        if not df_acc.empty:
            last_date = df_acc["Fecha"].iloc[-1]
            plt.axvline(x=last_date, 
                       color='gray', 
                       linestyle=':', 
                       alpha=0.7, 
                       linewidth=2)
        
        # ============================================================
        # 5. FORMATO
        # ============================================================
        plt.title(f"Comparación de todos los modelos (Top 3 destacados) - {title}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Valor", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/all_models_top3_{safe_title}_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ Gráfica guardada: {filename}")
        else:
            plt.show()