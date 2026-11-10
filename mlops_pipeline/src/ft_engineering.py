"""Ingeniería de características para el modelo de riesgo crediticio.

Recoge las decisiones de `comprension_eda.ipynb` y deja los datos listos para modelar:

1. Carga de `Base_de_datos.xlsx` y limpieza (nulos unificados, valores imposibles -> NaN).
2. Generación de atributos derivados (razones, indicadores, variables de fecha, tramos de edad).
3. División train/test (temporal por defecto, o aleatoria estratificada).
4. `ColumnTransformer` con un pipeline por tipo de variable:
   - numéricas: `SimpleImputer` (mediana + indicador de nulo) + log + `StandardScaler`
   - categóricas nominales: `SimpleImputer` + `OneHotEncoder`
   - categóricas ordinales: `SimpleImputer` + `OrdinalEncoder`

Uso desde otro módulo o notebook:

```python
from ft_engineering import preparar_datos

X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
X_train_t = preprocesador.fit_transform(X_train)
X_test_t = preprocesador.transform(X_test)
```
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes (las mismas de Cargar_datos.ipynb y comprension_eda.ipynb)
# ---------------------------------------------------------------------------
NOMBRE_ARCHIVO = "Base_de_datos.xlsx"
COLUMNA_FECHA = "fecha_prestamo"
TARGET = "Pago_atiempo"

CAT_TENDENCIA = ["Decreciente", "Estable", "Creciente"]          # orden lógico (ordinal)
CAT_LABORAL = ["Empleado", "Independiente"]
CAT_GRUPO_EDAD = ["18-30", "31-40", "41-50", "51-60", "61+"]
TOKENS_NULOS = {"", "na", "n/a", "nan", "null", "none", "nulo", "-", "?", "sin dato", "s/d"}

# Rango "duro": fuera de él el valor se considera imposible y se pasa a NaN. (None = sin límite)
RANGO_DURO = {
    "capital_prestado":              (1, None),
    "plazo_meses":                   (1, 60),
    "edad_cliente":                  (18, 100),
    "salario_cliente":               (100_000, 100_000_000),
    "total_otros_prestamos":         (0, None),
    "cuota_pactada":                 (1, None),
    "puntaje_datacredito":           (1, 999),   # 0 y negativos = "sin puntaje", no un puntaje real
    "cant_creditosvigentes":         (0, None),
    "huella_consulta":               (0, None),
    "saldo_mora":                    (0, None),
    "saldo_total":                   (0, None),
    "saldo_principal":               (0, None),
    "saldo_mora_codeudor":           (0, None),
    "creditos_sectorFinanciero":     (0, None),
    "creditos_sectorCooperativo":    (0, None),
    "creditos_sectorReal":           (0, None),
    "promedio_ingresos_datacredito": (1, None),
}

# saldo_mora_codeudor: casi constante; saldo_principal: redundante con saldo_total (EDA, sección 3)
# puntaje: fuga de información. Separa el objetivo sin error (puntaje >= 63 -> paga, AUC 1.0 en
# los 10 763 créditos), así que se calcula con el resultado de pago y no existe al originar.
COLUMNAS_ELIMINAR = ["saldo_mora_codeudor", "saldo_principal", "puntaje"]

# ---------------------------------------------------------------------------
# Variables que entran al modelo, agrupadas por el tratamiento que reciben
# ---------------------------------------------------------------------------
NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente", "total_otros_prestamos",
    "cuota_pactada", "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta",
    "saldo_mora", "saldo_total", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "promedio_ingresos_datacredito",
    # derivadas
    "carga_cuota_salario", "carga_cuota_ingreso_dc", "capital_sobre_salario", "deuda_otros_sobre_salario",
    "ratio_ingreso_declarado_vs_dc", "total_creditos_sectores", "pct_sector_financiero", "mora_sobre_saldo",
    "tiene_mora", "sin_ingresos_datacredito", "consultas_altas", "dia_semana", "es_fin_de_semana",
]
CATEGORICAS = ["tipo_credito", "tipo_laboral"]
ORDINALES = ["tendencia_ingresos", "grupo_edad"]
ORDEN_ORDINALES = [CAT_TENDENCIA, CAT_GRUPO_EDAD]   # mismo orden que ORDINALES


# ---------------------------------------------------------------------------
# 1. Carga y limpieza
# ---------------------------------------------------------------------------
def buscar_archivo(nombre: str = NOMBRE_ARCHIVO, desde: Path | None = None) -> Path:
    """Busca `nombre` en la carpeta indicada (por defecto, la de este archivo) y en sus carpetas padre."""
    desde = Path(desde or Path(__file__).parent).resolve()
    for carpeta in [desde, *desde.parents]:
        candidato = carpeta / nombre
        if candidato.exists():
            return candidato
    raise FileNotFoundError(f"No se encontró '{nombre}' desde {desde} hacia arriba.")


def cargar_datos(ruta: Path | str | None = None, hoja: str | int = 0) -> pd.DataFrame:
    """Lee la base de datos de créditos y ajusta el tipo de la fecha y del objetivo."""
    ruta = Path(ruta) if ruta else buscar_archivo()
    datos = pd.read_excel(ruta, sheet_name=hoja, engine="openpyxl")

    if pd.api.types.is_numeric_dtype(datos[COLUMNA_FECHA]):
        datos[COLUMNA_FECHA] = pd.to_datetime(datos[COLUMNA_FECHA], unit="D", origin="1899-12-30")
    else:
        datos[COLUMNA_FECHA] = pd.to_datetime(datos[COLUMNA_FECHA], errors="coerce")

    datos[TARGET] = datos[TARGET].astype(int)
    return datos


def fuera_de_rango(serie: pd.Series, lo, hi) -> pd.Series:
    """Máscara de valores no nulos que violan el rango [lo, hi]."""
    m = pd.Series(False, index=serie.index)
    if lo is not None:
        m |= serie < lo
    if hi is not None:
        m |= serie > hi
    return m & serie.notna()


def limpiar(datos: pd.DataFrame) -> pd.DataFrame:
    """Unifica nulos (NaN), anula valores imposibles y elimina las columnas descartadas en el EDA.

    Las reglas son fijas (no se aprenden de los datos), así que aplicarlas antes de dividir
    train/test no introduce fuga de información.
    """
    d = datos.copy()

    # Texto: strip + marcadores de nulo -> NaN; valores fuera de catálogo -> NaN
    for col, catalogo in {"tipo_laboral": CAT_LABORAL, "tendencia_ingresos": CAT_TENDENCIA}.items():
        s = d[col].astype(object)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: np.nan if isinstance(v, str) and v.lower() in TOKENS_NULOS else v)
        d[col] = s.where(s.isin(catalogo), np.nan)

    # tipo_credito es un código, no una cantidad: se trata como texto
    d["tipo_credito"] = d["tipo_credito"].astype("Int64").astype(str).replace("<NA>", np.nan)

    # Numéricas: valores imposibles / centinela -> NaN
    for col, (lo, hi) in RANGO_DURO.items():
        d[col] = d[col].astype("float64")
        d.loc[fuera_de_rango(d[col], lo, hi), col] = np.nan

    return d.drop(columns=COLUMNAS_ELIMINAR)


# ---------------------------------------------------------------------------
# 2. Generación de atributos
# ---------------------------------------------------------------------------
def crear_atributos(datos: pd.DataFrame) -> pd.DataFrame:
    """Agrega los atributos derivados propuestos en el EDA (sección 8).

    Se omiten `pago_total_sobre_capital` (definición dudosa, ρ ≈ 0), `hora` (sin patrón) y `mes`
    (con partición temporal confunde el mes con la madurez de la etiqueta).
    """
    d = datos.copy()
    sectores = d[["creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal"]].sum(axis=1)

    # Razones de capacidad de pago y endeudamiento
    d["carga_cuota_salario"] = d["cuota_pactada"] / d["salario_cliente"]
    d["carga_cuota_ingreso_dc"] = d["cuota_pactada"] / d["promedio_ingresos_datacredito"]
    d["capital_sobre_salario"] = d["capital_prestado"] / d["salario_cliente"]
    d["deuda_otros_sobre_salario"] = d["total_otros_prestamos"] / d["salario_cliente"]
    d["ratio_ingreso_declarado_vs_dc"] = d["salario_cliente"] / d["promedio_ingresos_datacredito"]
    d["total_creditos_sectores"] = sectores
    d["pct_sector_financiero"] = d["creditos_sectorFinanciero"] / sectores.replace(0, np.nan)
    d["mora_sobre_saldo"] = d["saldo_mora"] / d["saldo_total"].replace(0, np.nan)

    # Indicadores binarios (0/1; NaN cuando el dato de origen falta)
    d["tiene_mora"] = (d["saldo_mora"] > 0).astype(float).where(d["saldo_mora"].notna())
    d["sin_ingresos_datacredito"] = d["promedio_ingresos_datacredito"].isna().astype(int)
    d["consultas_altas"] = (d["huella_consulta"] >= 6).astype(int)

    # Fecha de desembolso
    d["dia_semana"] = d[COLUMNA_FECHA].dt.dayofweek
    d["es_fin_de_semana"] = (d["dia_semana"] >= 5).astype(int)

    # Tramos de edad (ordinal)
    d["grupo_edad"] = pd.cut(d["edad_cliente"], [17, 30, 40, 50, 60, 100], labels=CAT_GRUPO_EDAD).astype(object)

    # Protección ante divisiones que den infinito
    return d.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 3. División train / test
# ---------------------------------------------------------------------------
def dividir_train_test(
    datos: pd.DataFrame,
    test_size: float = 0.2,
    temporal: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X/y en train y test.

    - `temporal=True` (recomendado en el EDA): los créditos más recientes van a test, simulando
      el uso del modelo sobre originaciones futuras y evitando fuga temporal.
    - `temporal=False`: partición aleatoria estratificada por el objetivo (clase 0 = 4.8 %).
    """
    if temporal:
        datos = datos.sort_values(COLUMNA_FECHA)
        corte = int(len(datos) * (1 - test_size))
        train, test = datos.iloc[:corte], datos.iloc[corte:]
    else:
        train, test = train_test_split(datos, test_size=test_size, stratify=datos[TARGET], random_state=random_state)

    columnas_x = [c for c in NUMERICAS + CATEGORICAS + ORDINALES if c in datos.columns]
    return train[columnas_x], test[columnas_x], train[TARGET], test[TARGET]


# ---------------------------------------------------------------------------
# 4. ColumnTransformer
# ---------------------------------------------------------------------------
def log_con_signo(X):
    """log(1 + |x|) conservando el signo: comprime colas largas (salarios, saldos) sin alterar el orden."""
    return np.sign(X) * np.log1p(np.abs(X))


def construir_preprocesador(numericas: list[str], categoricas: list[str], ordinales: list[str]) -> ColumnTransformer:
    """Pipeline de preprocesamiento por tipo de variable (se ajusta solo con train)."""
    pipe_numericas = Pipeline([
        # Mediana: robusta a las colas largas. El indicador conserva la señal de los nulos informativos.
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        # Log + escalado: necesarios para modelos lineales y de distancia (montos de miles a millones).
        # A los árboles no les afecta porque el orden de los valores se mantiene.
        ("log", FunctionTransformer(log_con_signo, feature_names_out="one-to-one")),
        ("escalado", StandardScaler()),
    ])

    pipe_categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Categorías con < 1 % de los casos (tipo_credito 6, 7, 68) se agrupan en una sola columna
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
    ])

    pipe_ordinales = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[ORDEN_ORDINALES[ORDINALES.index(c)] for c in ordinales],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        [
            ("num", pipe_numericas, numericas),
            ("cat", pipe_categoricas, categoricas),
            ("ord", pipe_ordinales, ordinales),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def preparar_datos(
    ruta: Path | str | None = None,
    test_size: float = 0.2,
    temporal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Carga, limpia, genera atributos, divide y devuelve el preprocesador sin ajustar."""
    datos = crear_atributos(limpiar(cargar_datos(ruta)))
    X_train, X_test, y_train, y_test = dividir_train_test(datos, test_size=test_size, temporal=temporal)
    preprocesador = construir_preprocesador(NUMERICAS, CATEGORICAS, ORDINALES)
    return X_train, X_test, y_train, y_test, preprocesador


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()

    X_train_t = pd.DataFrame(preprocesador.fit_transform(X_train), columns=preprocesador.get_feature_names_out(),
                             index=X_train.index)
    X_test_t = pd.DataFrame(preprocesador.transform(X_test), columns=preprocesador.get_feature_names_out(),
                            index=X_test.index)

    print(f"Train: {X_train.shape} -> {X_train_t.shape} | tasa pago a tiempo {y_train.mean():.2%}")
    print(f"Test:  {X_test.shape} -> {X_test_t.shape} | tasa pago a tiempo {y_test.mean():.2%}")
    print(f"Nulos tras transformar: train {int(X_train_t.isna().sum().sum())}, test {int(X_test_t.isna().sum().sum())}")
    print("Columnas generadas:", X_train_t.columns.tolist())
