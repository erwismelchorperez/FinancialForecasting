"""
build_forecast_dependencies.py
=================================

Genera el grafo de dependencias para FORECASTING (distinto del de
cuadre contable en build_account_dependencies.py).

Fuente: únicamente la columna "CUENTA RELACIONADA" del Balance (la que
tu archivo llena con una fórmula tipo '=+B60' apuntando exacto a la
fila de la cuenta del Estado de Resultados). Ej.:

    Disponibilidades (balance)  <->  Intereses de disponibilidades (resultados)

Es una relación 1 a 1, bidireccional: si pronosticas Disponibilidades,
Intereses de disponibilidades debería moverse acorde (y viceversa).

NO incluye las cadenas de suma/indicador (eso ya lo cubre
build_account_dependencies.py, pensado para el cuadre contable).

Uso:
    python build_forecast_dependencies.py catalogo_cuentas.json --outdir ./salida

Salidas:
    - relaciones_forecast.json
    - account_dependencies_forecast.py   (ACCOUNT_DEPENDENCIES_FORECAST)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_forecast_pairs(catalog: dict[str, dict]) -> dict[int, set[int]]:
    deps: dict[int, set[int]] = defaultdict(set)
    for fila_str, meta in catalog.items():
        fila = int(fila_str)
        rel_fila = meta.get("cuenta_relacionada_fila")
        if rel_fila is None:
            continue
        deps[fila].add(rel_fila)
        deps[rel_fila].add(fila)
    return deps


def remap_to_account_id(deps: dict[int, set[int]], offset: int) -> dict[int, list[int]]:
    out = {fila - offset: sorted(t - offset for t in targets) for fila, targets in deps.items()}
    return dict(sorted(out.items()))


def write_json(account_deps: dict[int, list[int]], path: Path):
    payload = {str(k): {"dependencies": v} for k, v in account_deps.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_python(account_deps: dict[int, list[int]], path: Path, catalog: dict[str, dict], offset: int):
    nombre_por_id = {int(f) - offset: m.get("cuenta", "") for f, m in catalog.items()}
    lines = ["ACCOUNT_DEPENDENCIES_FORECAST = {"]
    for account_id, dep_ids in account_deps.items():
        nombre = nombre_por_id.get(account_id, "")
        lines.append(f"    {account_id}: {{  # {nombre}")
        lines.append(f'        "dependencies": {dep_ids}')
        lines.append("    },")
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("catalogo", help="Ruta a catalogo_cuentas.json")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--offset", type=int, default=2)
    args = ap.parse_args()

    catalog = json.load(open(args.catalogo, encoding="utf-8"))
    deps = build_forecast_pairs(catalog)
    account_deps = remap_to_account_id(deps, args.offset)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(account_deps, outdir / "relaciones_forecast.json")
    write_python(account_deps, outdir / "account_dependencies_forecast.py", catalog, args.offset)

    print(f"{len(account_deps)} cuentas con relación 1-a-1 -> {outdir}/relaciones_forecast.json")


if __name__ == "__main__":
    main()
