import pandas as pd

class FinancialPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        df = self.df.copy()

        # limpiar nombres de columnas (solo si son strings)
        df.columns = [
            col.strip() if isinstance(col, str) else col
            for col in df.columns
        ]

        # eliminar columnas duplicadas
        df = df.loc[:, ~df.columns.duplicated()]

        for col in df.columns:

            if col in ["NIVEL", "BALANCE GENERAL"]:
                continue

            series = df[col]

            # 🔥 SOLO limpiar si es tipo objeto (string)
            if series.dtype == "object":

                series = (
                    series
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace(r"[^\d\.-]", "", regex=True)
                    .str.strip()
                )

                df[col] = pd.to_numeric(series, errors="coerce")

            else:
                # ya es numérico → no tocar
                df[col] = pd.to_numeric(series, errors="coerce")
        df = df.dropna()
        return df

    def to_long(self):
        df = self.df.copy()
        df = df[~df["BALANCE GENERAL"].str.contains("ESTADO DE RESULTADOS", case=False, na=False)]
        # 🔹 Convertir a formato largo
        df_long = df.melt(
            id_vars=["NIVEL", "BALANCE GENERAL"],
            var_name="Fecha",
            value_name="Valor"
        )

        # 🔹 Convertir fecha de forma robusta (funciona con texto o datetime)
        df_long["Fecha"] = pd.to_datetime(df_long["Fecha"], errors="coerce")

        # 🔹 Ordenar correctamente
        df_long = df_long.sort_values(["BALANCE GENERAL", "Fecha"])

        # 🔹 Limpiar valores nulos en la variable objetivo
        df_long = df_long.dropna(subset=["Valor"])
        df_long['Valor'] = pd.to_numeric(df_long['Valor'], errors='coerce')


        # 🔹 FEATURES DE TIEMPO (muy importantes)
        df_long["year"] = df_long["Fecha"].dt.year
        df_long["month"] = df_long["Fecha"].dt.month

        # 🔹 LAGS (memoria temporal del modelo)
        df_long["lag_1"] = df_long.groupby("BALANCE GENERAL")["Valor"].shift(1)
        df_long['lag_1'] = pd.to_numeric(df_long['lag_1'], errors='coerce')
        df_long["lag_2"] = df_long.groupby("BALANCE GENERAL")["Valor"].shift(2)
        df_long['lag_2'] = pd.to_numeric(df_long['lag_2'], errors='coerce')
        df_long["lag_3"] = df_long.groupby("BALANCE GENERAL")["Valor"].shift(3)
        df_long['lag_3'] = pd.to_numeric(df_long['lag_3'], errors='coerce')

        # 🔹 Eliminar filas con NaN generadas por lags
        df_long = df_long.dropna()

        # 🔹 Reset index (buena práctica)
        df_long = df_long.reset_index(drop=True)

        # 🔍 Debug opcional
        print("Columnas finales:", df_long.columns)
        print(df_long.head())

        return df_long