"""Monitoreo del modelo de riesgo crediticio y detección de data drift.

Compara cada período de créditos contra los datos de entrenamiento (referencia) y genera las
tablas que consume `app_monitoreo.py`. Hay dos escenarios, que nunca se mezclan:

- **real**: los créditos de test en orden cronológico, tratados como si fueran producción. Responde
  "¿cambiaron de verdad los clientes?".
- **simulacion**: lotes con una perturbación controlada y reproducible de intensidad creciente.
  Responde "¿el monitor detecta un cambio cuando lo hay?". No dice nada sobre la población real.

Métricas (el semáforo usa solo el PSI: con muchos datos, KS y chi-cuadrado marcan como
significativas diferencias mínimas):

- PSI y Jensen-Shannon para todas las variables, con una categoría propia para los nulos.
- Kolmogorov-Smirnov para las continuas y chi-cuadrado para las categóricas y binarias.

Regla de alertas:

- Crítica (sugiere reentrenar): una variable importante o la probabilidad del modelo en rojo en
  dos ventanas consecutivas.
- Advertencia (revisar): una variable importante o la probabilidad en rojo por primera vez.
- Informativa: rojo en una variable poco importante, o un salto brusco del PSI.

Se ejecuta con `python model_monitoring.py` desde esta carpeta. Para que corra con la periodicidad
definida en producción, se programa ese comando (Programador de tareas de Windows o cron).

Salidas en `mlops_pipeline/monitoreo/`:

- `predicciones.db`: tabla SQLite con datos y pronósticos por crédito (fuera de git).
- CSV y JSON agregados, sin datos individuales: es lo único que lee la app.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ft_engineering import (CATEGORICAS, COLUMNA_FECHA, NUMERICAS, ORDINALES, RANGO_DURO, TARGET, cargar_datos,
                            crear_atributos, dividir_train_test, limpiar)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PERIODICIDAD = "M"              # frecuencia de pandas: "M" mensual, "Q" trimestral
MIN_CREDITOS_VENTANA = 100      # los períodos con menos créditos se agrupan: su PSI sería casi puro ruido
TAM_MUESTRA = 500               # máximo de créditos por ventana que entran al cálculo de drift
BINS = 10
PROP_MINIMA_CEROS = 0.20        # desde este % de ceros, el cero tiene su propio bin (si no, los cuantiles colapsan)

PSI_AMARILLO = 0.10             # convención de la industria
PSI_ROJO = 0.25
SALTO_ABRUPTO = 0.15            # cambio de PSI entre dos ventanas consecutivas
P_TENDENCIA = 0.05
TOP_VARIABLES = 10              # variables "importantes": las de mayor importancia en el XGBoost guardado

FOLDS = 5                       # los mismos de model_training_evaluation.py
SEMILLA = 42

VARIABLE_PROBA = "probabilidad_pago"
ESCENARIO_REAL = "real"
ESCENARIO_SIMULADO = "simulacion"

# Variables numéricas que en realidad son discretas: se comparan por categoría, no con KS ni cuantiles
NUMERICAS_DISCRETAS = ["tiene_mora", "sin_ingresos_datacredito", "consultas_altas", "es_fin_de_semana", "dia_semana"]
DISCRETAS = CATEGORICAS + ORDINALES + NUMERICAS_DISCRETAS
CONTINUAS = [c for c in NUMERICAS if c not in NUMERICAS_DISCRETAS]

# Simulación: se perturban columnas originales y las derivadas se recalculan con crear_atributos
INTENSIDADES_SIMULACION = [0.0, 0.5, 1.0, 1.5]
FRACCION_LOTE = 0.3
MONTOS_CON_RUIDO = ["capital_prestado", "cuota_pactada", "salario_cliente", "total_otros_prestamos", "saldo_total",
                    "promedio_ingresos_datacredito"]
CATEGORICAS_PERTURBADAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_MONITOREO = RAIZ / "monitoreo"


# ---------------------------------------------------------------------------
# Métricas de drift (sobre conteos por bin ya alineados entre referencia y actual)
# ---------------------------------------------------------------------------
def calcular_psi(conteo_ref: np.ndarray, conteo_act: np.ndarray) -> float:
    """Population Stability Index: suma de (%actual - %referencia) * ln(%actual / %referencia).

    Suma medio caso a cada bin para evitar log(0). Un piso fijo muy chico (1e-6) inflaría el PSI:
    un solo bin vacío que debería tener el 10 % aportaría más de 1.
    """
    p_ref = (conteo_ref + 0.5) / (conteo_ref.sum() + 0.5 * len(conteo_ref))
    p_act = (conteo_act + 0.5) / (conteo_act.sum() + 0.5 * len(conteo_act))
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def psi_por_ruido(n_bins: int, n_ref: int, n_act: int) -> float:
    """PSI esperado si no hubiera ningún cambio real: crece cuanto más chica es la ventana."""
    return (n_bins - 1) * (1 / n_ref + 1 / n_act)


def calcular_jensen_shannon(conteo_ref: np.ndarray, conteo_act: np.ndarray) -> float:
    """Distancia de Jensen-Shannon, de 0 (iguales) a 1 (sin nada en común)."""
    return float(jensenshannon(conteo_ref / conteo_ref.sum(), conteo_act / conteo_act.sum(), base=2))


def calcular_ks(serie_ref: pd.Series, serie_act: pd.Series) -> tuple[float, float]:
    """Kolmogorov-Smirnov: máxima distancia entre las dos distribuciones acumuladas (sin nulos)."""
    resultado = stats.ks_2samp(serie_ref.dropna(), serie_act.dropna())
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_chi2(conteo_ref: np.ndarray, conteo_act: np.ndarray) -> tuple[float, float]:
    """Chi-cuadrado de homogeneidad entre las proporciones de cada categoría."""
    if len(conteo_ref) < 2:
        return 0.0, 1.0
    chi2, p_valor, *_ = stats.chi2_contingency(np.column_stack([conteo_ref, conteo_act]))
    return float(chi2), float(p_valor)


def semaforo(psi: float) -> str:
    if psi >= PSI_ROJO:
        return "rojo"
    return "amarillo" if psi >= PSI_AMARILLO else "verde"


# ---------------------------------------------------------------------------
# Bins: se definen con la referencia y se aplican igual a cada ventana
# ---------------------------------------------------------------------------
def _numero_corto(valor: float) -> str:
    if abs(valor) >= 1e6:
        return f"{valor / 1e6:.1f}M"
    if abs(valor) >= 1e3:
        return f"{valor / 1e3:.0f}k"
    return f"{valor:.3g}"


def _texto_categoria(valor) -> str:
    return str(int(valor)) if isinstance(valor, (int, float, np.number)) and float(valor).is_integer() else str(valor)


def cortes_continua(referencia: pd.Series) -> np.ndarray:
    """Cortes interiores por deciles; si hay muchos ceros, el cero queda en un bin propio."""
    valores = referencia.dropna()
    probs = np.linspace(0, 1, BINS + 1)[1:-1]
    if (valores == 0).mean() >= PROP_MINIMA_CEROS:
        return np.unique([0.0, *np.quantile(valores[valores != 0], probs)])
    return np.unique(np.quantile(valores, probs))


def definir_bins(referencia: pd.Series, variable: str) -> dict:
    if variable in DISCRETAS:
        return {"categorias": sorted(referencia.dropna().map(_texto_categoria).unique())}
    return {"cortes": cortes_continua(referencia)}


def etiquetas_bins(bins: dict) -> list[str]:
    if "categorias" in bins:
        return [*bins["categorias"], "Otra", "Nulo"]
    c = [_numero_corto(v) for v in bins["cortes"]]
    return [f"<= {c[0]}", *[f"{a} - {b}" for a, b in zip(c, c[1:])], f"> {c[-1]}", "Nulo"]


def contar_bins(serie: pd.Series, bins: dict) -> np.ndarray:
    """Casos por bin, en el mismo orden que `etiquetas_bins` (el último bin siempre son los nulos)."""
    validos = serie.dropna()
    if "categorias" in bins:
        categorias = bins["categorias"]
        posiciones = {c: i for i, c in enumerate(categorias)}
        indices = validos.map(_texto_categoria).map(posiciones).fillna(len(categorias)).astype(int)
        conteo = np.bincount(indices, minlength=len(categorias) + 1)
    else:
        indices = np.searchsorted(bins["cortes"], validos.to_numpy(dtype=float), side="left")
        conteo = np.bincount(indices, minlength=len(bins["cortes"]) + 1)
    return np.append(conteo, serie.isna().sum()).astype(float)


# ---------------------------------------------------------------------------
# Tabla de pronósticos y ventanas
# ---------------------------------------------------------------------------
def importancia_variables(pipeline) -> pd.Series:
    """Importancia del XGBoost agrupada por variable original (suma one-hot e indicadores de nulo)."""
    columnas = pipeline.named_steps["preprocesador"].get_feature_names_out()

    def origen(columna: str) -> str:
        if columna.startswith("missingindicator_"):
            return columna.removeprefix("missingindicator_")
        return next((c for c in CATEGORICAS if columna.startswith(f"{c}_")), columna)

    importancia = pd.Series(pipeline.named_steps["modelo"].feature_importances_, index=columnas)
    agrupada = importancia.groupby(origen).sum().sort_values(ascending=False)
    return agrupada / agrupada.sum()


def probabilidad_referencia(pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Probabilidades de train con validación cruzada: cada crédito se predice con un modelo que no lo vio.

    Puntuar train con el modelo final daría probabilidades demasiado buenas y se vería drift falso
    en cualquier dato nuevo.
    """
    X_transformado = pipeline.named_steps["preprocesador"].transform(X)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEMILLA)
    modelo = clone(pipeline.named_steps["modelo"])
    return cross_val_predict(modelo, X_transformado, y, cv=cv, method="predict_proba")[:, 1]


def asignar_ventanas(fechas: pd.Series, frecuencia: str = PERIODICIDAD,
                     minimo: int = MIN_CREDITOS_VENTANA) -> pd.Series:
    """Etiqueta cada fecha con su ventana, uniendo períodos consecutivos hasta alcanzar `minimo` créditos."""
    periodos = fechas.dt.to_period(frecuencia)
    grupos, actual = [], []
    for periodo, n in periodos.value_counts().sort_index().items():
        actual.append((periodo, n))
        if sum(c for _, c in actual) >= minimo:
            grupos.append(actual)
            actual = []
    if actual:   # lo que sobra al final se une a la última ventana
        grupos[-1].extend(actual) if grupos else grupos.append(actual)

    etiquetas = {}
    for grupo in grupos:
        inicio, fin = grupo[0][0], grupo[-1][0]
        etiqueta = str(inicio) if inicio == fin else f"{inicio} a {fin}"
        etiquetas.update({periodo: etiqueta for periodo, _ in grupo})
    return periodos.map(etiquetas)


def generar_batch_actual(limpios: pd.DataFrame, intensidad: float, semilla: int) -> pd.DataFrame:
    """Simula un lote de producción con una perturbación controlada y reproducible.

    Parte de una muestra de créditos de test (no solapa con la referencia) y altera las columnas
    originales para que las derivadas sigan siendo coherentes. Con intensidad 0 no se toca nada:
    sirve de control. Representa un deterioro de la cartera: bajan salarios y puntaje, sube la mora
    y cambia la mezcla de categorías. `edad_cliente`, `huella_consulta` y los créditos por sector
    no se alteran, para comprobar que el monitor no marca lo que no cambió.
    """
    rng = np.random.default_rng(semilla)
    lote = limpios.sample(frac=FRACCION_LOTE, random_state=semilla).copy()
    if intensidad == 0:
        return crear_atributos(lote)

    # Ruido multiplicativo: con colas largas, sumar desviaciones estándar daría montos negativos,
    # y los montos en cero siguen en cero.
    for columna in MONTOS_CON_RUIDO:
        lote[columna] *= np.exp(rng.normal(0, 0.10 * intensidad, len(lote)))
    lote["salario_cliente"] *= 1 - 0.40 * intensidad

    sigma = limpios["puntaje_datacredito"].std()
    desplazado = lote["puntaje_datacredito"] + rng.normal(-0.8 * intensidad * sigma, 0.10 * intensidad * sigma, len(lote))
    lote["puntaje_datacredito"] = desplazado.round().clip(*RANGO_DURO["puntaje_datacredito"])

    # La mora se altera cambiando quién la tiene, para que el indicador siga siendo 0/1
    con_mora = limpios.loc[limpios["saldo_mora"] > 0, "saldo_mora"].to_numpy()   # muy pocos: se toman de todo el test
    sin_mora = lote.index[lote["saldo_mora"] == 0]
    nuevos = rng.choice(sin_mora, size=int(0.05 * intensidad * len(sin_mora)), replace=False)
    lote.loc[nuevos, "saldo_mora"] = rng.choice(con_mora, size=len(nuevos))

    for columna in CATEGORICAS_PERTURBADAS:
        proporciones = limpios[columna].value_counts(normalize=True, dropna=False)
        pesos = proporciones.to_numpy().copy()
        pesos[0] *= 1 - 0.45 * intensidad    # la categoría dominante pierde peso
        pesos[1:] *= 1 + 0.80 * intensidad   # las demás lo ganan
        elegidos = rng.choice(len(pesos), size=len(lote), p=pesos / pesos.sum())
        lote[columna] = proporciones.index.to_numpy()[elegidos]

    return crear_atributos(lote)


def puntuar(pipeline, umbral: float, datos: pd.DataFrame, columnas: list[str], escenario: str,
            ventanas: pd.Series, etiquetas_conocidas: bool, muestra_maxima: int | None = None) -> pd.DataFrame:
    """Devuelve la tabla de pronósticos de un escenario: datos, probabilidad, decisión y ventana."""
    tabla = datos[[COLUMNA_FECHA, *columnas]].copy()
    tabla[VARIABLE_PROBA] = pipeline.predict_proba(datos[columnas])[:, 1]
    tabla["decision_aprobar"] = (tabla[VARIABLE_PROBA] >= umbral).astype(int)
    tabla[TARGET] = datos[TARGET] if etiquetas_conocidas else np.nan
    tabla["escenario"] = escenario
    tabla["ventana"] = ventanas
    return marcar_muestra(tabla, muestra_maxima)


def marcar_muestra(tabla: pd.DataFrame, muestra_maxima: int | None) -> pd.DataFrame:
    """Marca con `en_muestra` los créditos que entran al cálculo de drift de cada ventana."""
    orden = {v: i for i, v in enumerate(pd.unique(tabla.sort_values(COLUMNA_FECHA)["ventana"]))}
    tabla["orden"] = tabla["ventana"].map(orden)
    tabla["en_muestra"] = False
    for numero, grupo in tabla.groupby("orden"):
        tamano = min(len(grupo), muestra_maxima or len(grupo))
        tabla.loc[grupo.sample(tamano, random_state=SEMILLA + int(numero)).index, "en_muestra"] = True
    return tabla


# ---------------------------------------------------------------------------
# Cálculo por ventana
# ---------------------------------------------------------------------------
def medir_drift(referencia: pd.DataFrame, tabla: pd.DataFrame, variables: list[str], importantes: pd.Series,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula las métricas y las distribuciones de cada variable en cada ventana de un escenario."""
    bins_por_variable = {v: definir_bins(referencia[v], v) for v in variables}
    filas, distribuciones = [], []
    for orden, ventana in tabla[["orden", "ventana"]].drop_duplicates().sort_values("orden").itertuples(index=False):
        actual = tabla[(tabla["orden"] == orden) & tabla["en_muestra"]]
        for variable in variables:
            bins = bins_por_variable[variable]
            serie_ref, serie_act = referencia[variable], actual[variable]
            conteo_ref, conteo_act = contar_bins(serie_ref, bins), contar_bins(serie_act, bins)

            usados = (conteo_ref + conteo_act) > 0   # un bin vacío en ambos lados no aporta información
            ref, act = conteo_ref[usados], conteo_act[usados]
            psi = calcular_psi(ref, act)
            discreta = variable in DISCRETAS
            ks, ks_p = (np.nan, np.nan) if discreta else calcular_ks(serie_ref, serie_act)
            chi2, chi2_p = calcular_chi2(ref, act) if discreta else (np.nan, np.nan)

            filas.append({
                "escenario": tabla["escenario"].iloc[0], "orden": orden, "ventana": ventana, "variable": variable,
                "tipo": "prediccion" if variable == VARIABLE_PROBA else "discreta" if discreta else "continua",
                "importante": variable == VARIABLE_PROBA or variable in importantes.index,
                "importancia": importantes.get(variable, np.nan),
                "n_ref": len(serie_ref), "n_act": len(serie_act),
                "psi": psi, "psi_ruido": psi_por_ruido(int(usados.sum()), len(serie_ref), len(serie_act)),
                "js": calcular_jensen_shannon(ref, act), "ks": ks, "ks_p": ks_p, "chi2": chi2, "chi2_p": chi2_p,
                "nulos_ref": serie_ref.isna().mean(), "nulos_act": serie_act.isna().mean(),
                "media_ref": np.nan if discreta else serie_ref.mean(),
                "media_act": np.nan if discreta else serie_act.mean(),
                "semaforo": semaforo(psi),
            })
            for posicion, (etiqueta, n_ref, n_act, usado) in enumerate(
                    zip(etiquetas_bins(bins), conteo_ref, conteo_act, usados)):
                if usado:
                    distribuciones.append({
                        "escenario": tabla["escenario"].iloc[0], "orden": orden, "ventana": ventana,
                        "variable": variable, "posicion": posicion, "bin": etiqueta,
                        "prop_ref": n_ref / conteo_ref.sum(), "prop_act": n_act / conteo_act.sum(),
                    })

    metricas = pd.DataFrame(filas)
    metricas["salto_psi"] = metricas.sort_values("orden").groupby("variable")["psi"].diff()
    return metricas, pd.DataFrame(distribuciones)


def analizar_tendencias(metricas: pd.DataFrame) -> pd.DataFrame:
    """Tendencia del PSI por variable. Se descuenta el ruido: si no, las ventanas chicas simulan un aumento."""
    filas = []
    for (escenario, variable), grupo in metricas.sort_values("orden").groupby(["escenario", "variable"]):
        neto = grupo["psi"] - grupo["psi_ruido"]
        pendiente, p_valor = np.nan, np.nan
        tendencia = "sin datos suficientes"
        if len(grupo) >= 4:
            ajuste = stats.linregress(grupo["orden"], neto)
            pendiente, p_valor = float(ajuste.slope), float(ajuste.pvalue)
            tendencia = "estable"
            if p_valor < P_TENDENCIA:
                tendencia = "creciente" if pendiente > 0 else "decreciente"

        saltos = grupo["salto_psi"].abs()
        hay_salto = saltos.max() >= SALTO_ABRUPTO
        filas.append({
            "escenario": escenario, "variable": variable, "importante": grupo["importante"].iloc[0],
            "psi_ultimo": grupo["psi"].iloc[-1], "psi_maximo": grupo["psi"].max(),
            "ventanas_rojo": int((grupo["semaforo"] == "rojo").sum()), "pendiente": pendiente, "p_tendencia": p_valor,
            "tendencia": tendencia, "salto_maximo": saltos.max(),
            "ventana_salto": grupo.loc[saltos.idxmax(), "ventana"] if hay_salto else "",
        })
    return pd.DataFrame(filas)


def generar_alertas(metricas: pd.DataFrame) -> pd.DataFrame:
    """Aplica la regla de alertas ventana por ventana y variable por variable."""
    filas = []
    for (escenario, variable), grupo in metricas.sort_values("orden").groupby(["escenario", "variable"]):
        rojo_anterior = False
        for f in grupo.itertuples():
            base = {"escenario": escenario, "orden": f.orden, "ventana": f.ventana, "variable": variable, "psi": f.psi}
            rojo = f.semaforo == "rojo"
            nombre = "La probabilidad que entrega el modelo" if variable == VARIABLE_PROBA else f"'{variable}'"
            if rojo and f.importante and rojo_anterior:
                filas.append({**base, "nivel": "critica",
                              "mensaje": f"{nombre} lleva dos ventanas seguidas en rojo (PSI {f.psi:.2f}).",
                              "accion": "Evaluar reentrenar el modelo con datos recientes y comprobar antes si el "
                                        "cambio es real o un problema de captura del dato."})
            elif rojo and f.importante:
                filas.append({**base, "nivel": "advertencia",
                              "mensaje": f"{nombre} entró en rojo (PSI {f.psi:.2f}).",
                              "accion": "Revisar la variable y su fuente. Si se repite en la próxima ventana, "
                                        "evaluar reentrenar."})
            elif rojo:
                filas.append({**base, "nivel": "informativa",
                              "mensaje": f"{nombre} está en rojo (PSI {f.psi:.2f}), pero pesa poco en el modelo.",
                              "accion": "Seguir su evolución; no justifica reentrenar por sí sola."})
            if f.importante and abs(f.salto_psi) >= SALTO_ABRUPTO:   # NaN en la primera ventana: no entra
                sentido = "subió" if f.salto_psi > 0 else "bajó"
                filas.append({**base, "nivel": "informativa",
                              "mensaje": f"Cambio brusco: el PSI de {nombre} {sentido} {abs(f.salto_psi):.2f} "
                                         "respecto de la ventana anterior.",
                              "accion": "Buscar qué cambió en ese período (política comercial, canal o fuente)."})
            rojo_anterior = rojo
    columnas = ["escenario", "orden", "ventana", "nivel", "variable", "psi", "mensaje", "accion"]
    return pd.DataFrame(filas, columns=columnas)


def resumir_ventanas(tabla: pd.DataFrame, metricas: pd.DataFrame, alertas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ventana: volumen, comportamiento del modelo, semáforo y estado general."""
    filas = []
    for (orden, ventana), grupo in tabla.groupby(["orden", "ventana"]):
        escenario = grupo["escenario"].iloc[0]
        m = metricas[metricas["orden"] == orden]
        a = alertas[(alertas["escenario"] == escenario) & (alertas["orden"] == orden)]["nivel"]
        amarillo_importante = ((m["semaforo"] == "amarillo") & m["importante"]).any()
        estado = ("Reentrenar" if (a == "critica").any() else "Revisar" if (a == "advertencia").any()
                  else "Observar" if amarillo_importante or len(a) else "Estable")
        filas.append({
            "escenario": escenario, "orden": orden, "ventana": ventana, "n_creditos": len(grupo),
            "n_muestra": int(grupo["en_muestra"].sum()), "tasa_rechazo": 1 - grupo["decision_aprobar"].mean(),
            "proba_media": grupo[VARIABLE_PROBA].mean(), "tasa_pago": grupo[TARGET].mean(),
            "no_paga": int((grupo[TARGET] == 0).sum()), "rojos": int((m["semaforo"] == "rojo").sum()),
            "amarillos": int((m["semaforo"] == "amarillo").sum()), "verdes": int((m["semaforo"] == "verde").sum()),
            "estado": estado,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def main() -> None:
    artefacto = joblib.load(DIR_MODELOS / "mejor_modelo.joblib")
    pipeline, umbral = artefacto["pipeline"], artefacto["umbral"]

    limpios = limpiar(cargar_datos())
    datos = crear_atributos(limpios)
    X_train, X_test, y_train, y_test = dividir_train_test(datos)
    columnas = X_train.columns.tolist()
    variables = [VARIABLE_PROBA, *[c for c in CONTINUAS + DISCRETAS if c in columnas]]
    importancia = importancia_variables(pipeline)
    importantes = importancia.head(TOP_VARIABLES)

    # Referencia: créditos de entrenamiento, con probabilidades de validación cruzada
    referencia = X_train.copy()
    referencia[VARIABLE_PROBA] = probabilidad_referencia(pipeline, X_train, y_train)

    # Escenario real: créditos de test por ventana temporal
    test = datos.loc[X_test.index]
    tabla_real = puntuar(pipeline, umbral, test, columnas, ESCENARIO_REAL, asignar_ventanas(test[COLUMNA_FECHA]),
                         etiquetas_conocidas=True, muestra_maxima=TAM_MUESTRA)

    # Escenario simulado: un lote por cada intensidad, tratados como ventanas consecutivas
    limpios_test = limpios.loc[X_test.index]
    lotes = []
    for numero, intensidad in enumerate(INTENSIDADES_SIMULACION, start=1):
        lote = generar_batch_actual(limpios_test, intensidad, SEMILLA + numero)
        etiqueta = pd.Series(f"Lote {numero} (intensidad {intensidad:.1f})", index=lote.index)
        tabla_lote = puntuar(pipeline, umbral, lote, columnas, ESCENARIO_SIMULADO, etiqueta, etiquetas_conocidas=False)
        lotes.append(tabla_lote.assign(orden=numero - 1))
    tabla_simulada = pd.concat(lotes)

    metricas_real, distrib_real = medir_drift(referencia, tabla_real, variables, importantes)
    metricas_sim, distrib_sim = medir_drift(referencia, tabla_simulada, variables, importantes)
    metricas = pd.concat([metricas_real, metricas_sim], ignore_index=True)
    alertas = generar_alertas(metricas)
    tendencias = analizar_tendencias(metricas)
    resumen = pd.concat([resumir_ventanas(tabla_real, metricas_real, alertas),
                         resumir_ventanas(tabla_simulada, metricas_sim, alertas)], ignore_index=True)

    meta = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "periodicidad": PERIODICIDAD, "min_creditos_ventana": MIN_CREDITOS_VENTANA, "tam_muestra": TAM_MUESTRA,
        "psi_amarillo": PSI_AMARILLO, "psi_rojo": PSI_ROJO, "salto_abrupto": SALTO_ABRUPTO,
        "umbral_modelo": umbral, "variable_proba": VARIABLE_PROBA,
        "variables_importantes": {k: float(v) for k, v in importantes.items()},
        "referencia": {
            "n": len(referencia), "desde": str(datos.loc[X_train.index, COLUMNA_FECHA].min().date()),
            "hasta": str(datos.loc[X_train.index, COLUMNA_FECHA].max().date()),
            "tasa_rechazo": float((referencia[VARIABLE_PROBA] < umbral).mean()),
            "proba_media": float(referencia[VARIABLE_PROBA].mean()), "tasa_pago": float(y_train.mean()),
        },
        "auc_test": float(roc_auc_score(y_test, tabla_real[VARIABLE_PROBA])),
        "no_paga_test": int((y_test == 0).sum()),
    }

    DIR_MONITOREO.mkdir(exist_ok=True)
    with sqlite3.connect(DIR_MONITOREO / "predicciones.db") as conexion:
        pd.concat([tabla_real, tabla_simulada]).rename_axis("id_credito").reset_index().to_sql(
            "predicciones", conexion, if_exists="replace", index=False)
    metricas.to_csv(DIR_MONITOREO / "metricas_drift.csv", index=False)
    pd.concat([distrib_real, distrib_sim]).to_csv(DIR_MONITOREO / "distribuciones.csv", index=False)
    alertas.to_csv(DIR_MONITOREO / "alertas.csv", index=False)
    tendencias.to_csv(DIR_MONITOREO / "tendencias.csv", index=False)
    resumen.to_csv(DIR_MONITOREO / "resumen_ventanas.csv", index=False)
    (DIR_MONITOREO / "resumen_monitoreo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")

    print(f"Referencia: {len(referencia)} créditos | tasa de rechazo {meta['referencia']['tasa_rechazo']:.1%}")
    print(resumen.drop(columns=["tasa_pago", "proba_media"]).round(3).to_string(index=False))
    print(f"\nAlertas: {alertas.groupby(['escenario', 'nivel']).size().to_dict()}")
    print(f"Resultados en {DIR_MONITOREO}")


if __name__ == "__main__":
    main()
