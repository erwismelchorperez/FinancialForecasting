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