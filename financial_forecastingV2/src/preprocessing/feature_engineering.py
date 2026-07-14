from config.settings import LAGS


class FeatureEngineering:

    def create_features(self, df):

        df = df.copy()

        df["year"] = df["Fecha"].dt.year
        df["month"] = df["Fecha"].dt.month

        for lag in LAGS:
            df[f"lag_{lag}"] = (
                df.groupby("BALANCE GENERAL")["Valor"]
                .shift(lag)
            )

        return df