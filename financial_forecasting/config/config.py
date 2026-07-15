LAGS = [1, 2, 3, 6, 12]
TRAIN_END = "2023-12-01"
FORECAST_STEPS = 12  # 2025–2027
"""
HIERARCHY = {
    "ACTIVO": [
        "Disponibilidades",
        "Inversiones en valores",
        "Deudores por reporto",
        "Cartera de crédito vigente",
        "Créditos comerciales",
        "Créditos de consumo",
        "Créditos a la vivienda"
    ]
}
"""

HIERARCHY = {
    "ACTIVO": [
        "Inversiones en valores",
    ]
}

# ==================================
# Forecasting configuration
# ==================================
USE_ACCOUNT_DEPENDENCIES = True