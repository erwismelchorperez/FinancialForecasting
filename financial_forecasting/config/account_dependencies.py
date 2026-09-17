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
    5: {# cuenta objetivo
        "dependencies": [62,125]
    },
    62: {# cuenta objetivo
        "dependencies": [5,125]
    },
    125: {# cuenta objetivo
        "dependencies": [62,5]
    }
}