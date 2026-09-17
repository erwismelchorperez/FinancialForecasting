"""
preprocess_financials.py
=========================

Preprocesa archivos Excel de "Históricos" bancarios (Balance General,
Estado de Resultados e Indicadores) que siguen la plantilla tipo
`Historicos_Florencio.xlsx`:

    Col A: NIVEL (jerarquía contable)
    Col B: Nombre de la cuenta
    Col C: Cuenta relacionada (solo Balance)
    Col D..: una columna por mes (fecha en la fila de encabezado)

El script:
  1. Detecta automáticamente las 3 secciones (Balance, Resultados,
     Indicadores) buscando los encabezados dentro de la hoja, en vez
     de usar números de fila fijos (para que funcione con archivos
     "similares" cuyo tamaño cambie).
  2. Distingue cuentas "hoja" (dato duro, se debe pronosticar) de
     cuentas "agregado"/"indicador" (se calculan con fórmula a partir
     de otras, NO se deben pronosticar directamente).
  3. Limpia cada serie: recorta el periodo antes de que la cuenta
     "exista" (NaN estructural) y marca huecos internos para
     interpolar, en vez de asumir 0.
  4. Conserva las relaciones que pidió el usuario:
       - "cuenta relacionada" Balance <-> Resultados
         (ej. Disponibilidades <-> Intereses de disponibilidades)
       - jerarquía padre/hijo (para poder reconstruir agregados)
       - fórmulas de los indicadores (para poder validar coherencia)
  5. Exporta:
       - <seccion>_limpio.csv       (formato largo: fecha, cuenta...)
       - <seccion>_wide.csv         (fecha x cuenta, solo hojas)
       - catalogo_cuentas.json      (metadatos + fórmulas genéricas)
       - reporte_calidad.csv        (huecos / fecha de arranque por cuenta)

Uso:
    python preprocess_financials.py archivo.xlsx --outdir ./salida
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import openpyxl
import pandas as pd


# --------------------------------------------------------------------------
# Configuración de patrones de detección de secciones (ajustable si otro
# archivo "similar" usa otro texto de encabezado)
# --------------------------------------------------------------------------
BALANCE_HEADER_PAT = re.compile(r"BALANCE\s+GENERAL", re.IGNORECASE)
RESULTADOS_HEADER_PAT = re.compile(r"ESTADO\s+DE\s+RESULTADOS", re.IGNORECASE)
INDICADORES_HEADER_PAT = re.compile(r"INDICADOR", re.IGNORECASE)
COMPROBACION_PAT = re.compile(r"COMPROBACI[ÓO]N", re.IGNORECASE)


@dataclass
class CuentaMeta:
    fila: int
    seccion: str                      # balance | resultados | indicadores
    nivel: Optional[str]              # '1','2','3'... o 'Operación' o None
    cuenta: str
    cuenta_relacionada: Optional[str] = None
    cuenta_relacionada_fila: Optional[int] = None   # fila exacta referenciada por "=+B<fila>"
    nota: Optional[str] = None        # texto libre no-fórmula en cuenta_relacionada
    tipo: str = "hoja"                # hoja | agregado | indicador | encabezado
    formula_generica: Optional[str] = None   # fórmula con columna genérica {c}
    hijos: list = field(default_factory=list)   # filas hijas (jerarquía)


class FinancialModelParser:
    def __init__(self, path: str, sheet_name: Optional[str] = None):
        self.path = Path(path)
        self.wb_val = openpyxl.load_workbook(path, data_only=True)
        self.wb_frm = openpyxl.load_workbook(path, data_only=False)
        self.sheet_name = sheet_name or self.wb_val.sheetnames[0]
        self.ws_val = self.wb_val[self.sheet_name]
        self.ws_frm = self.wb_frm[self.sheet_name]
        self.max_row = self.ws_val.max_row
        self.max_col = self.ws_val.max_column

        self.date_cols: list[tuple[int, "pd.Timestamp"]] = []
        self.catalog: dict[int, CuentaMeta] = {}
        self._section_bounds: dict[str, tuple[int, int]] = {}

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _col_letter(idx: int) -> str:
        return openpyxl.utils.get_column_letter(idx)

    def _find_date_columns(self, header_row: int) -> list[tuple[int, pd.Timestamp]]:
        cols = []
        for c in range(1, self.max_col + 1):
            v = self.ws_val.cell(row=header_row, column=c).value
            if hasattr(v, "year"):  # datetime-like
                cols.append((c, pd.Timestamp(v)))
        return cols

    # ------------------------------------------------------------ secciones
    def _find_sections(self):
        balance_hdr = resultados_hdr = indicadores_hdr = None
        for r in range(1, self.max_row + 1):
            b = self.ws_val.cell(row=r, column=2).value
            if not isinstance(b, str):
                continue
            if balance_hdr is None and BALANCE_HEADER_PAT.search(b):
                balance_hdr = r
            elif resultados_hdr is None and RESULTADOS_HEADER_PAT.search(b):
                resultados_hdr = r
            elif indicadores_hdr is None and INDICADORES_HEADER_PAT.search(b):
                indicadores_hdr = r

        if balance_hdr is None or resultados_hdr is None:
            raise ValueError(
                "No se encontraron los encabezados 'BALANCE GENERAL' / "
                "'ESTADO DE RESULTADOS' en la columna B. ¿Es este el "
                "layout esperado?"
            )

        # columnas de fecha: se toman del encabezado de balance (deberían
        # ser las mismas columnas para las 3 secciones)
        self.date_cols = self._find_date_columns(balance_hdr)

        # última fila con datos reales (para no cargar basura/filas vacías)
        last_row_with_data = balance_hdr
        for r in range(balance_hdr, self.max_row + 1):
            if any(
                self.ws_frm.cell(row=r, column=c).value is not None
                for c in range(1, self.max_col + 1)
            ):
                last_row_with_data = r

        indicadores_end = last_row_with_data if indicadores_hdr else resultados_hdr
        resultados_end = (indicadores_hdr - 1) if indicadores_hdr else last_row_with_data

        self._section_bounds = {
            "balance": (balance_hdr + 1, resultados_hdr - 1),
            "resultados": (resultados_hdr + 1, resultados_end),
            "indicadores": (indicadores_hdr, indicadores_end) if indicadores_hdr else None,
        }
        self.balance_header_row = balance_hdr

    # ------------------------------------------------------- lectura de fila
    def _row_has_formula(self, row: int) -> bool:
        if not self.date_cols:
            return False
        first_col = self.date_cols[0][0]
        v = self.ws_frm.cell(row=row, column=first_col).value
        return isinstance(v, str) and v.startswith("=")

    def _genericize_formula(self, row: int) -> Optional[str]:
        """Toma la fórmula de la primera columna de fecha y sustituye la
        letra de columna por un placeholder {c}, para poder re-aplicarla
        a cualquier periodo/columna futura."""
        if not self.date_cols:
            return None
        first_col_idx = self.date_cols[0][0]
        col_letter = self._col_letter(first_col_idx)
        v = self.ws_frm.cell(row=row, column=first_col_idx).value
        if not (isinstance(v, str) and v.startswith("=")):
            return None
        pattern = re.compile(rf"\b{col_letter}(\$?\d+)\b")
        return pattern.sub(r"{c}\1", v)

    def _resolve_related(self, row: int) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """Columna C (cuenta relacionada). Puede ser:
        - una fórmula tipo '=+B60' -> apunta EXACTO a la fila 60 (ya
          viene resuelta en ws_val por el nombre cacheado)
        - texto libre (una nota/regla de negocio, no una cuenta)
        Devuelve (cuenta_relacionada, nota, fila_relacionada)
        """
        raw_formula = self.ws_frm.cell(row=row, column=3).value
        val = self.ws_val.cell(row=row, column=3).value
        if val is None:
            return None, None, None
        if isinstance(raw_formula, str) and raw_formula.startswith("="):
            m = re.match(r"^=\+?[A-Z]{1,2}(\d+)$", raw_formula.strip())
            fila_rel = int(m.group(1)) if m else None
            return str(val), None, fila_rel
        # texto libre => es una nota de negocio, no un nombre de cuenta real
        return None, str(val), None

    # ------------------------------------------------------------- parseo
    def parse(self) -> dict[str, pd.DataFrame]:
        self._find_sections()
        frames = {}
        for seccion in ("balance", "resultados", "indicadores"):
            bounds = self._section_bounds.get(seccion)
            if not bounds:
                continue
            frames[seccion] = self._parse_section(seccion, *bounds)
        return frames

    def _parse_section(self, seccion: str, start: int, end: int) -> pd.DataFrame:
        records = []
        last_nivel_stack: list[tuple[int, int]] = []  # (nivel_numerico, fila)

        for r in range(start, end + 1):
            nivel_raw = self.ws_val.cell(row=r, column=1).value
            nombre_raw = self.ws_val.cell(row=r, column=2).value
            has_formula = self._row_has_formula(r)
            if nombre_raw is None and nivel_raw is None and not has_formula:
                continue  # fila realmente vacía / separador

            nombre = str(nombre_raw) if nombre_raw is not None else None
            nivel_txt = None if nivel_raw is None else str(nivel_raw)
            formula_generica = self._genericize_formula(r)

            # Filas tipo "COMPROBACIÓN BALANCE" traen su etiqueta en la
            # columna A (nivel_raw) en vez de la columna B (nombre).
            es_comprobacion_por_col_a = bool(
                nivel_txt and COMPROBACION_PAT.search(nivel_txt)
            )
            if es_comprobacion_por_col_a and nombre is None:
                nombre = nivel_txt

            # Filas de chequeo sin ninguna etiqueta (p.ej. total
            # ingresos/egresos usado solo para la comprobación) reciben
            # un nombre sintético para no perderlas ni chocar columnas.
            if nombre is None and has_formula:
                nombre = f"(renglón de control {r})"

            nivel = None if (nivel_txt is None or es_comprobacion_por_col_a) else nivel_txt

            cuenta_relacionada, nota, cuenta_relacionada_fila = (None, None, None)
            if seccion == "balance" and nombre is not None:
                cuenta_relacionada, nota, cuenta_relacionada_fila = self._resolve_related(r)

            # clasificación del tipo de fila
            if nombre and (COMPROBACION_PAT.search(nombre) or es_comprobacion_por_col_a):
                tipo = "comprobacion"
            elif nivel is None and nombre is not None and not has_formula:
                tipo = "encabezado_categoria"      # p.ej. "cartera vencida"
            elif seccion == "indicadores":
                tipo = "indicador" if has_formula else "encabezado_categoria"
            elif has_formula:
                tipo = "agregado"
            elif nombre is not None:
                tipo = "hoja"
            else:
                continue

            # jerarquía padre/hijo (solo aplica donde nivel es numérico)
            padre_fila = None
            if nivel is not None and nivel.isdigit():
                nivel_num = int(nivel)
                last_nivel_stack = [x for x in last_nivel_stack if x[0] < nivel_num]
                if last_nivel_stack:
                    padre_fila = last_nivel_stack[-1][1]
                last_nivel_stack.append((nivel_num, r))

            meta = CuentaMeta(
                fila=r,
                seccion=seccion,
                nivel=nivel,
                cuenta=nombre or "",
                cuenta_relacionada=cuenta_relacionada,
                cuenta_relacionada_fila=cuenta_relacionada_fila,
                nota=nota,
                tipo=tipo,
                formula_generica=formula_generica,
            )
            self.catalog[r] = meta
            if padre_fila is not None:
                self.catalog[padre_fila].hijos.append(r)

            if tipo in ("encabezado_categoria",):
                continue  # no tiene serie de tiempo propia

            for col, fecha in self.date_cols:
                valor = self.ws_val.cell(row=r, column=col).value
                records.append(
                    {
                        "seccion": seccion,
                        "fila": r,
                        "nivel": nivel,
                        "cuenta": nombre,
                        "cuenta_relacionada": cuenta_relacionada,
                        "nota": nota,
                        "tipo": tipo,
                        "fecha": fecha,
                        "valor": valor,
                    }
                )

        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------
def clean_series(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza por cuenta (fila):
    - Antes de la primera fecha con dato real: NaN estructural
      (columna 'activa' = False) -> la cuenta no aplicaba aún.
    - Huecos dentro del rango activo: se interpolan linealmente y se
      marca 'imputado' = True para que el forecasting sepa qué es dato
      real vs relleno.
    - No se sustituyen NaN por 0: un 0 real (p.ej. la cuenta llegó a
      cero) es distinto de "no hay dato".
    """
    out = []
    for fila, g in df.groupby("fila", sort=False):
        g = g.sort_values("fecha").reset_index(drop=True)
        serie = g["valor"]
        primero_valido = serie.first_valid_index()
        g["activa"] = False
        g["imputado"] = False
        g["valor_limpio"] = serie

        if primero_valido is not None:
            g.loc[primero_valido:, "activa"] = True
            sub = g.loc[primero_valido:, "valor"]
            interpolada = sub.interpolate(method="linear", limit_direction="forward")
            g.loc[primero_valido:, "valor_limpio"] = interpolada
            g.loc[primero_valido:, "imputado"] = sub.isna() & interpolada.notna()

        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def to_wide(df: pd.DataFrame, value_col: str = "valor_limpio") -> pd.DataFrame:
    """Tabla ancha fecha x cuenta, usando 'fila' + 'cuenta' como
    identificador de columna para no chocar nombres duplicados
    (p.ej. 'Créditos comerciales' aparece varias veces)."""
    if df.empty:
        return df
    d = df.copy()
    d["cuenta"] = d["cuenta"].fillna("(sin nombre)")
    d["col_id"] = d["fila"].astype(str) + " | " + d["cuenta"]
    wide = d.pivot(index="fecha", columns="col_id", values=value_col)
    return wide.sort_index()


# --------------------------------------------------------------------------
# Reporte de calidad de datos
# --------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fila, g in df.groupby("fila", sort=False):
        g = g.sort_values("fecha")
        activo = g[g["activa"]] if "activa" in g else g
        n = len(activo)
        n_imput = int(activo["imputado"].sum()) if "imputado" in activo else 0
        rows.append(
            {
                "fila": fila,
                "cuenta": g["cuenta"].iloc[0],
                "tipo": g["tipo"].iloc[0],
                "fecha_inicio": activo["fecha"].min() if n else None,
                "fecha_fin": activo["fecha"].max() if n else None,
                "n_periodos_activos": n,
                "n_imputados": n_imput,
                "pct_imputado": round(100 * n_imput / n, 2) if n else None,
            }
        )
    return pd.DataFrame(rows).sort_values("fila")


# --------------------------------------------------------------------------
# Exportación
# --------------------------------------------------------------------------
def export_all(parser: FinancialModelParser, frames: dict[str, pd.DataFrame], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    catalog_export = {}

    for seccion, df in frames.items():
        if df.empty:
            continue
        df_clean = clean_series(df)
        df_clean.to_csv(outdir / f"{seccion}_limpio.csv", index=False)

        hojas = df_clean[df_clean["tipo"] == "hoja"]
        if not hojas.empty:
            to_wide(hojas).to_csv(outdir / f"{seccion}_wide.csv")

        agregados = df_clean[df_clean["tipo"].isin(["agregado", "indicador"])]
        if not agregados.empty:
            to_wide(agregados).to_csv(outdir / f"{seccion}_agregados_wide.csv")

        qr = quality_report(df_clean)
        qr.to_csv(outdir / f"{seccion}_reporte_calidad.csv", index=False)

    for fila, meta in parser.catalog.items():
        catalog_export[fila] = asdict(meta)

    relations = {
        "balance_a_resultados": {
            m.fila: {
                "cuenta": m.cuenta,
                "cuenta_relacionada": m.cuenta_relacionada,
                "cuenta_relacionada_fila": m.cuenta_relacionada_fila,
            }
            for m in parser.catalog.values()
            if m.seccion == "balance" and m.cuenta_relacionada
        },
        "notas_negocio": {
            m.fila: {"cuenta": m.cuenta, "nota": m.nota}
            for m in parser.catalog.values()
            if m.nota
        },
        "jerarquia": {
            m.fila: {"cuenta": m.cuenta, "nivel": m.nivel, "hijos": m.hijos}
            for m in parser.catalog.values()
            if m.hijos
        },
        "formulas": {
            m.fila: {
                "cuenta": m.cuenta,
                "seccion": m.seccion,
                "tipo": m.tipo,
                "formula_generica": m.formula_generica,
            }
            for m in parser.catalog.values()
            if m.formula_generica
        },
    }

    with open(outdir / "catalogo_cuentas.json", "w", encoding="utf-8") as f:
        json.dump(catalog_export, f, ensure_ascii=False, indent=2, default=str)

    with open(outdir / "relaciones.json", "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2, default=str)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archivo", help="Ruta al Excel de históricos (.xlsx)")
    ap.add_argument("--outdir", default="./salida", help="Carpeta de salida")
    ap.add_argument("--hoja", default=None, help="Nombre de la hoja (opcional)")
    args = ap.parse_args()

    parser = FinancialModelParser(args.archivo, sheet_name=args.hoja)
    frames = parser.parse()
    export_all(parser, frames, Path(args.outdir))

    print("Listo. Archivos generados en:", Path(args.outdir).resolve())
    for seccion, df in frames.items():
        if df.empty:
            print(f"  - {seccion}: 0 cuentas")
            continue
        n_hojas = df.loc[df["tipo"] == "hoja", "fila"].nunique()
        n_filas = df["fila"].nunique()
        print(f"  - {seccion}: {n_filas} cuentas ({n_hojas} tipo 'hoja')")


if __name__ == "__main__":
    main()
