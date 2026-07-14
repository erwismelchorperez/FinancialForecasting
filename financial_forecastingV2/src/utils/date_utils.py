import pandas as pd

def generate_future_dates(last_date, steps):
    return pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="ME"
    )
def get_last_values(df_account, lags):
    return df_account["Valor"].values[-lags:]