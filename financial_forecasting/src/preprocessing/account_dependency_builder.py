"""
import pandas as pd
from config.account_dependencies import ACCOUNT_DEPENDENCIES
class AccountDependencyBuilder:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def build(self, target_account):
        dependencies = ACCOUNT_DEPENDENCIES.get(target_account,[])
        # ----------------------------
        # Cuenta objetivo
        # ----------------------------
        target = self.dataframe[ self.dataframe["BALANCE GENERAL"] == target_account][ [ "Fecha", "Valor"] ].copy()

        target.rename(columns={ "Valor":"target" }, inplace=True)

        # ----------------------------
        # Agregar dependencias
        # ----------------------------
        for dependency in dependencies:
            dep = self.dataframe[self.dataframe["BALANCE GENERAL"] == dependency ][ [ "Fecha", "Valor" ]].copy()

            dep.rename( columns={ "Valor":dependency }, inplace=True)

            target = target.merge(dep,on="Fecha",how="left")

        target = target.sort_values("Fecha")
        return target
"""

import pandas as pd
from config.account_dependencies import ACCOUNT_DEPENDENCIES
class AccountDependencyBuilder:
    """
    Builder que construye un DataFrame con la cuenta objetivo y sus dependencias.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame fuente con los datos de cuentas.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa el builder con el DataFrame fuente.
        
        Args:
            dataframe (pd.DataFrame): DataFrame que contiene los datos de las cuentas.
                                      Debe tener columnas 'Fecha' y 'Valor', y el índice
                                      debe contener los identificadores de las cuentas.
        """
        self.dataframe = dataframe
    
    def build(self, account_index):
        """
        Construye un DataFrame combinando la cuenta objetivo con sus dependencias.
        
        Args:
            account_index: Identificador de la cuenta objetivo.
            
        Returns:
            pd.DataFrame: DataFrame ordenado por 'Fecha' que contiene:
                         - Columna 'target': valores de la cuenta objetivo
                         - Columnas 'dependency_{index}': valores de cada dependencia
                         
        Example:
            builder = AccountDependencyBuilder(df)
            result = builder.build('CUENTA_001')
            # Resultado: DataFrame con columnas Fecha, target, dependency_002, dependency_003...
        """
        # Obtener las dependencias configuradas para la cuenta
        dependencies = ACCOUNT_DEPENDENCIES.get(account_index, {}).get("dependencies", [])
        
        # ==========================
        # Cuenta objetivo
        # ==========================
        target = self.dataframe[self.dataframe.index == account_index][["Fecha", "Valor"]].copy()
        target.rename(columns={"Valor": "target"}, inplace=True)
        
        # ==========================
        # Cuentas relacionadas
        # ==========================
        for dep_index in dependencies:
            dep = self.dataframe[self.dataframe.index == dep_index][["Fecha", "Valor"]].copy()
            dep.rename(columns={"Valor": f"dependency_{dep_index}"}, inplace=True)
            target = target.merge(dep, on="Fecha", how="left")
        
        return target.sort_values("Fecha")