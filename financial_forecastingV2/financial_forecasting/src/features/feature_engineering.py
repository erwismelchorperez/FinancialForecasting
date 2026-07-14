class FeatureEngineer:

    def __init__(self, lags):
        self.lags = lags

    def transform(self, df):
        df = df.copy()

        for lag in self.lags:
            df[f"lag_{lag}"] = df["Valor"].shift(lag)

        df["trend"] = range(len(df))
        df["month"] = df["Fecha"].dt.month

        df = df.dropna()

        return df