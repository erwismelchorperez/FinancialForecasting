LAGS = [1, 2, 3, 6, 12]

FORECAST_STEPS = 12

TRAIN_SIZE = 0.80

TARGET_COLUMN = "Valor"

DATE_COLUMN = "Fecha"

ACCOUNT_COLUMN = "BALANCE GENERAL"

HIERARCHY = {
    "ACTIVO": [
        "Disponibilidades",
        "Inversiones en valores"
    ]
}