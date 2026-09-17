import pandas as pd

from config.account_dependencies import ACCOUNT_DEPENDENCIES


class DependencyPreprocessor:

    def __init__(self, df_long, n_lags=3):

        self.df = df_long.copy()
        self.df_long = None
        self.n_lags = n_lags

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

        # -------------------------------------
        # Conservar índice original del Excel
        # -------------------------------------
        df = (
            df
            .reset_index()
            .rename(columns={"index": "account_id"})
        )

        # -------------------------------------
        # Eliminar estado de resultados
        # -------------------------------------
        df = df[
            ~df["BALANCE GENERAL"].str.contains(
                "ESTADO DE RESULTADOS",
                case=False,
                na=False
            )
        ]

        # -------------------------------------
        # Convertir a formato largo
        # -------------------------------------
        df_long = df.melt(
            id_vars=[
                "account_id",
                "NIVEL",
                "BALANCE GENERAL"
            ],
            var_name="Fecha",
            value_name="Valor"
        )

        # -------------------------------------
        # Fechas
        # -------------------------------------
        df_long["Fecha"] = pd.to_datetime(
            df_long["Fecha"],
            errors="coerce"
        )

        # -------------------------------------
        # Valor numérico
        # -------------------------------------
        df_long["Valor"] = pd.to_numeric(
            df_long["Valor"],
            errors="coerce"
        )

        # -------------------------------------
        # Ordenar
        # -------------------------------------
        df_long = df_long.sort_values(
            ["account_id", "Fecha"]
        )

        # -------------------------------------
        # Eliminar nulos
        # -------------------------------------
        df_long = df_long.dropna(
            subset=["Fecha", "Valor"]
        )

        self.df_long = df_long.reset_index(drop=True)

        return df_long.reset_index(drop=True)
    
    def build(self, account_id):

        # ============================
        # Obtener dependencias
        # ============================
        config = ACCOUNT_DEPENDENCIES.get(account_id)
        
        # Verificar si existe configuración para la cuenta
        if config is None:
            print(f"⚠ No hay configuración de dependencias para la cuenta {account_id}")
            print(f"   Usando modelo sin dependencias (solo target)")
            dependencies = []
        else:
            dependencies = config.get("dependencies", [])

        print(f"\n==============================")
        print(f"Cuenta objetivo ID: {account_id}")
        
        # Obtener el nombre de la cuenta para mejor legibilidad
        account_name = self.df_long[self.df_long["account_id"] == account_id]["BALANCE GENERAL"].iloc[0] if not self.df_long[self.df_long["account_id"] == account_id].empty else "Desconocido"
        print(f"Nombre: {account_name}")
        print(f"Dependencias (IDs): {dependencies}")
        
        # Mostrar nombres de las dependencias
        for dep_id in dependencies:
            dep_name = self.df_long[self.df_long["account_id"] == dep_id]["BALANCE GENERAL"].iloc[0] if not self.df_long[self.df_long["account_id"] == dep_id].empty else "Desconocido"
            print(f"  - {dep_id}: {dep_name}")

        # ============================
        # Cuenta objetivo
        # ============================
        target = (
            self.df_long[
                self.df_long["account_id"] == account_id
            ][["Fecha", "Valor"]]
            .copy()
        )

        if target.empty:
            raise ValueError(
                f"No existen datos para la cuenta {account_id}"
            )

        target.rename(
            columns={"Valor": "target"},
            inplace=True
        )

        dataset = target

        # ============================
        # Agregar dependencias
        # ============================
        for dep_id in dependencies:

            dependency = (
                self.df_long[
                    self.df_long["account_id"] == dep_id
                ][["Fecha", "Valor"]]
                .copy()
            )

            if dependency.empty:
                print(f"⚠ Dependencia {dep_id} sin datos.")
                continue

            # Usar el nombre de la dependencia para la columna
            dep_name = self.df_long[self.df_long["account_id"] == dep_id]["BALANCE GENERAL"].iloc[0] if not self.df_long[self.df_long["account_id"] == dep_id].empty else f"dep_{dep_id}"
            
            dependency.rename(
                columns={
                    "Valor": f"dep_{dep_id}"
                },
                inplace=True
            )

            dataset = dataset.merge(
                dependency,
                on="Fecha",
                how="left"
            )

        # ============================
        # Variables de tiempo
        # ============================
        dataset["year"] = dataset["Fecha"].dt.year
        dataset["month"] = dataset["Fecha"].dt.month

        # ============================
        # Crear lags
        # ============================
        variables = [
            c
            for c in dataset.columns
            if c not in ["Fecha", "year", "month"]
        ]

        for variable in variables:

            for lag in range(1, self.n_lags + 1):

                dataset[f"{variable}_lag_{lag}"] = (
                    dataset[variable].shift(lag)
                )

        # ============================
        # Eliminar NaN
        # ============================
        dataset.dropna(inplace=True)

        dataset.reset_index(
            drop=True,  
            inplace=True
        )

        print("\nDataset construido")
        print(dataset.head())

        return dataset
    
    def get_accounts(self):
        """
        Devuelve todas las cuentas disponibles.
        """
        accounts = (
            self.df_long[
                ["account_id", "BALANCE GENERAL"]
            ]
            .drop_duplicates()
            .sort_values("account_id")
        )

        print("\n=== Cuentas disponibles ===")
        for _, row in accounts.iterrows():
            print(f"ID: {row['account_id']} - Nombre: {row['BALANCE GENERAL']}")
        
        return self.df_long["account_id"].drop_duplicates().tolist()
    
    def get_account_info(self, account_id):
        """
        Obtiene información de una cuenta específica.
        """
        info = self.df_long[self.df_long["account_id"] == account_id]
        if not info.empty:
            return {
                "account_id": account_id,
                "name": info["BALANCE GENERAL"].iloc[0],
                "level": info["NIVEL"].iloc[0]
            }
        return None