import pandas as pd


class ForecastSaver:

    def __init__(self, output_dir):

        self.output_dir = output_dir

    def save(self, forecasts, future_dates):

        rows = []

        for account, values in forecasts.items():

            row = {
                "BALANCE GENERAL": account
            }

            for date, value in zip(future_dates, values):

                row[date.strftime("%b-%y")] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        filepath = (
            f"{self.output_dir}/forecasts.csv"
        )

        df.to_csv(filepath, index=False)

        print(f"Forecast guardado: {filepath}")