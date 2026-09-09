# src/models/trainer_factory.py

from src.models.independent_trainer import IndependentAccountTrainer
from src.models.dependent_trainer import DependentAccountTrainer
from config.account_dependencies import ACCOUNT_DEPENDENCIES


class TrainerFactory:
    """
    Fábrica que decide qué tipo de entrenador usar según la cuenta.
    """
    
    @staticmethod
    def get_trainer(account_id, models, test_size=12):
        """
        Retorna el entrenador adecuado para la cuenta.
        
        Args:
            account_id: ID de la cuenta
            models: Diccionario de modelos disponibles
            test_size: Tamaño del conjunto de prueba
            
        Returns:
            Trainer apropiado (Independent o Dependent)
        """
        # Verificar si la cuenta tiene dependencias configuradas
        config = ACCOUNT_DEPENDENCIES.get(account_id)
        
        if config and config.get("dependencies", []):
            print(f"🔗 Cuenta {account_id} tiene dependencias → Usando DependentAccountTrainer")
            return DependentAccountTrainer(models, test_size)
        else:
            print(f"📊 Cuenta {account_id} NO tiene dependencias → Usando IndependentAccountTrainer")
            return IndependentAccountTrainer(models, test_size)
    
    @staticmethod
    def get_trainer_by_type(trainer_type, models, test_size=12):
        """
        Retorna un entrenador según el tipo especificado.
        
        Args:
            trainer_type: 'independent' o 'dependent'
            models: Diccionario de modelos disponibles
            test_size: Tamaño del conjunto de prueba
        """
        if trainer_type == "dependent":
            return DependentAccountTrainer(models, test_size)
        elif trainer_type == "independent":
            return IndependentAccountTrainer(models, test_size)
        else:
            raise ValueError(f"Tipo de trainer no válido: {trainer_type}")