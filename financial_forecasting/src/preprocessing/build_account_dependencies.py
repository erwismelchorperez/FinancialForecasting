"""
build_account_dependencies.py
===============================

Construye el grafo de dependencias entre cuentas (ACCOUNT_DEPENDENCIES)
a partir de catalogo_cuentas.json, usando el mismo `account_id` que ya
tienes en tu `procesado.csv`.

Regla:
  Cada fórmula (de un agregado del balance/resultados, o de un
  indicador) conecta un conjunto de renglones: la cuenta objetivo +
  todas las cuentas que aparecen en su fórmula. Ese conjunto se trata
  como una "camarilla" de coherencia: cada cuenta del conjunto queda
  dependiente de TODAS las demás del mismo conjunto.

  Ejemplo real de tu archivo — "tasa de interés de créditos vigentes,
  Créditos comerciales" (fila 127) = (fila64 / MES * 12) / fila7:
      conjunto = {fila7, fila64, fila127}
      -> account_id (fila-2) = {5, 62, 125}
      -> 5: [62,125] | 62: [5,125] | 125: [5,62]
  (exactamente el ejemplo que diste)

  Una misma cuenta puede aparecer en varias fórmulas (p.ej. "Créditos
  comerciales vigentes" participa en la tasa de interés Y en el % de
  cartera vencida), así que sus dependencias son la UNIÓN de todos los
  conjuntos en los que participa.

account_id = fila_excel_original - (fila_del_encabezado_balance + 1)
Para este archivo, el encabezado de BALANCE GENERAL está en la fila 1,
por lo que account_id = fila - 2 (coincide con tu procesado.csv).

Salidas:
  - relaciones.json            -> {"5": {"dependencies": [62, 125]}, ...}
  - account_dependencies.py    -> ACCOUNT_DEPENDENCIES = {5: {"dependencies": [62, 125]}, ...}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_refs(formula_generica: str | None) -> set[int]:
    """Extrae los números de fila referenciados en una fórmula genérica
    (formato '{c}<fila>'). Se excluye la fila 1 porque ahí solo vive la
    fecha del periodo (MONTH({c}$1)), no es una cuenta."""
    if not formula_generica:
        return set()
    refs = {int(x) for x in re.findall(r"\{c\}\$?(\d+)", formula_generica)}
    refs.discard(1)
    return refs


def build_dependency_graph(catalog: dict[str, dict]) -> dict[int, set[int]]:
    deps: dict[int, set[int]] = defaultdict(set)
    for fila_str, meta in catalog.items():
        fila = int(fila_str)
        refs = extract_refs(meta.get("formula_generica"))
        if not refs:
            continue
        grupo = refs | {fila}
        for nodo in grupo:
            deps[nodo] |= (grupo - {nodo})
    return deps


def remap_to_account_id(deps: dict[int, set[int]], offset: int) -> dict[int, list[int]]:
    out = {}
    for fila, targets in deps.items():
        out[fila - offset] = sorted(t - offset for t in targets)
    return dict(sorted(out.items()))


def write_json(account_deps: dict[int, list[int]], path: Path):
    payload = {str(k): {"dependencies": v} for k, v in account_deps.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_python(account_deps: dict[int, list[int]], path: Path, catalog: dict[str, dict], offset: int):
    """Escribe el .py con el formato EXACTO que pediste, y agrega el
    nombre de la cuenta como comentario (en vez de '# cuenta objetivo'
    genérico) para que sea legible."""
    nombre_por_id = {int(f) - offset: m.get("cuenta", "") for f, m in catalog.items()}
    lines = ["ACCOUNT_DEPENDENCIES = {"]
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
    ap.add_argument(
        "--offset",
        type=int,
        default=2,
        help="account_id = fila - offset (por defecto 2, igual que tu procesado.csv)",
    )
    args = ap.parse_args()

    catalog = json.load(open(args.catalogo, encoding="utf-8"))
    deps = build_dependency_graph(catalog)
    account_deps = remap_to_account_id(deps, args.offset)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(account_deps, outdir / "relaciones.json")
    write_python(account_deps, outdir / "account_dependencies.py", catalog, args.offset)

    print(f"{len(account_deps)} cuentas con dependencias -> {outdir}/relaciones.json y account_dependencies.py")


if __name__ == "__main__":
    main()
