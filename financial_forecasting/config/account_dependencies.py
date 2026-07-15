# src/config/account_dependencies.py
"""
ACCOUNT_DEPENDENCIES = {
    "Inversiones en valores": [
        "Intereses y rendimientos a favor provenientes de inversiones en valores",
        "Tasa de interés de inversiones en valores"
    ]
}
"""
ACCOUNT_DEPENDENCIES = {
    7: {# cuenta objetivo
        "dependencies": [64,127]
    }
}