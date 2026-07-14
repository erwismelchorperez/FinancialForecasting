import matplotlib.pyplot as plt


class ForecastPlotter:

    def plot(
        self,
        df_account,
        forecast,
        future_dates,
        account_name
    ):

        plt.figure(figsize=(12, 5))

        # histórico
        plt.plot(
            df_account["Fecha"],
            df_account["Valor"],
            label="Histórico"
        )

        # forecast
        plt.plot(
            future_dates,
            forecast,
            linestyle="--",
            label="Forecast"
        )

        plt.title(account_name)

        plt.legend()

        plt.grid(True)

        plt.show()