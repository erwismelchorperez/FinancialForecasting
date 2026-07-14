import pandas as pd


class FinancialTransformer:

    def transform(self, df):

        id_vars = ["NIVEL", "BALANCE GENERAL"]

        value_vars = [
            col for col in df.columns
            if col not in id_vars
        ]

        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Fecha",
            value_name="Valor"
        )

        # limpiar números
        df_long["Valor"] = (
            df_long["Valor"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df_long["Valor"] = pd.to_numeric(
            df_long["Valor"],
            errors="coerce"
        )

        # fechas
        df_long["Fecha"] = pd.to_datetime(
            df_long["Fecha"],
            format="%b-%y"
        )

        df_long = df_long.sort_values(
            ["BALANCE GENERAL", "Fecha"]
        )

        return df_long