# src/models/model_selector.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging


class ModelSelector:
    """
    Selecciona y reentrena los mejores modelos basados en métricas.
    """
    
    def __init__(self, n_best_models=3, metric='mape'):
        """
        Args:
            n_best_models: Número de mejores modelos a seleccionar
            metric: Métrica para seleccionar ('mae', 'rmse', 'mape')
        """
        self.n_best_models = n_best_models
        self.metric = metric
        self.logger = logging.getLogger(__name__)
    
    def select_best_models(self, results: Dict) -> Dict:
        """
        Selecciona los mejores modelos para cada cuenta.
        
        Args:
            results: Diccionario con resultados de entrenamiento
            
        Returns:
            Dict con los mejores modelos seleccionados
        """
        best_models = {}
        
        for account_id, account_results in results.items():
            if not account_results:
                continue
            
            # Filtrar modelos válidos
            valid_models = {
                name: data for name, data in account_results.items()
                if data is not None and 'metrics' in data
            }
            
            if not valid_models:
                print(f"   ⚠️ No hay modelos válidos para la cuenta {account_id}")
                continue
            
            # Ordenar por métrica
            sorted_models = sorted(
                valid_models.items(),
                key=lambda x: x[1]['metrics'].get(self.metric, float('inf'))
            )
            
            # Seleccionar los N mejores
            n_selected = min(self.n_best_models, len(sorted_models))
            selected = sorted_models[:n_selected]
            
            best_models[account_id] = {
                'selected_models': selected,
                'best_model': selected[0][0] if selected else None,
                'all_models': sorted_models
            }
            
            # Logging
            print(f"\n   📊 Cuenta {account_id}:")
            print(f"      Mejores {n_selected} modelos por {self.metric.upper()}:")
            for i, (name, data) in enumerate(selected, 1):
                metrics = data['metrics']
                print(f"         {i}. {name}: {self.metric.upper()} = {metrics.get(self.metric, 'N/A'):.4f}")
                print(f"            MAE: {metrics.get('mae', 'N/A'):.2f}, RMSE: {metrics.get('rmse', 'N/A'):.2f}")
        
        return best_models
    
    def get_best_model_names(self, best_models: Dict) -> Dict:
        """
        Obtiene los nombres de los mejores modelos por cuenta.
        
        Returns:
            Dict con {account_id: [lista_de_modelos]}
        """
        result = {}
        for account_id, data in best_models.items():
            if 'selected_models' in data:
                result[account_id] = [name for name, _ in data['selected_models']]
        return result