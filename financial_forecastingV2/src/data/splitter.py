class AccountSplitter:

    def __init__(self, df_long):
        self.df = df_long

    def get_accounts(self):
        return self.df["BALANCE GENERAL"].unique()

    def get_account_df(self, account):
        return self.df[self.df["BALANCE GENERAL"] == account].copy()