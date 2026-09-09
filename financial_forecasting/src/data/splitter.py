# src/data/splitter.py

import pandas as pd


class AccountSplitter:
    """Separa los datos por cuenta."""
    
    def __init__(self, df_long):
        self.df_long = df_long
    
    def get_accounts(self):
        """Obtiene todas las cuentas disponibles."""
        return self.df_long["account_id"].unique().tolist()
    
    def get_account_df(self, account_id):
        """
        Obtiene el DataFrame de una cuenta específica.
        CORREGIDO: Maneja el caso cuando no encuentra datos.
        """
        result = self.df_long[self.df_long["account_id"] == account_id].copy()
        
        # CORRECCIÓN: Verificar si está vacío
        if result.empty:
            print(f"   ⚠️ No se encontraron datos para la cuenta {account_id}")
            # Devolver un DataFrame vacío con las mismas columnas
            return pd.DataFrame(columns=self.df_long.columns)
        
        return result
    
    def get_account_by_name(self, account_name):
        """Obtiene una cuenta por su nombre."""
        result = self.df_long[self.df_long["BALANCE GENERAL"] == account_name].copy()
        return result
    
    def get_account_names(self):
        """Obtiene los nombres de todas las cuentas."""
        return self.df_long["BALANCE GENERAL"].unique().tolist()