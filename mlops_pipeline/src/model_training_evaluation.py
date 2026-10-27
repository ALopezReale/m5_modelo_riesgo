"""Entrenamiento y evaluación de los primeros modelos de riesgo crediticio con PyCaret.

Flujo (se repite para dos escenarios: con y sin la variable `puntaje`, marcada en el EDA como
posible fuga de información):

1. Datos de `ft_engineering.preparar_datos()`: limpieza, atributos, división temporal y
   `ColumnTransformer` ajustado solo con train.
2. `setup` de PyCaret con validación cruzada estratificada y balanceo (SMOTE) dentro de cada fold.
3. `compare_models`: muchos clasificadores de una sola vez, ordenados por la métrica objetivo.
4. `build_model`: los mejores se optimizan con `tune_model`.
5. `summarize_classification`: métricas sobre el conjunto de test (datos que el modelo no vio).
6. Gráficos comparativos, matrices de confusión, curva de umbral y tabla resumen en `reportes/`.
7. Se guarda el mejor modelo (preprocesador + modelo) en `models/`.

Objetivo: `Pago_atiempo` (1 = paga a tiempo -> se le puede aprobar el crédito).

- Falso positivo (FP): el modelo dice "paga", se aprueba y NO paga -> se pierde el capital. Es el
  error más caro, por eso se reporta la **Especificidad** (% de no pagadores que el modelo rechaza).
- Falso negativo (FN): el modelo dice "no paga", se rechaza y SÍ pagaba -> se pierde el margen.

Métrica objetivo = **AUC**: mide qué tan bien ordena el modelo a los clientes de menor a mayor
riesgo, sin depender del umbral ni del desbalance (95 % paga). Con ese orden, el negocio elige el
umbral según cuántos FP está dispuesto a tolerar (ver `umbral_fp_fn.png`).

Para ver el detalle mientras corre: `python model_training_evaluation.py` desde esta carpeta.
"""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))   # evita un aviso de joblib en Windows

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycaret.classification import (add_metric, compare_models, plot_model, predict_model, pull, setup,
                                    tune_model)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from ft_engineering import TARGET, preparar_datos

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
METRICA_OBJETIVO = "AUC"
TOP_N = 3                 # modelos que pasan de compare_models a tune_model
N_ITER_TUNING = 30        # combinaciones de hiperparámetros probadas por modelo
FOLDS = 5
SEMILLA = 42
UMBRAL = 0.5              # probabilidad de pagar a partir de la cual se aprueba
VERBOSE = True            # muestra en la terminal las tablas de PyCaret mientras entrena

# Modelos sin predict_proba (no permiten AUC ni mover el umbral de decisión)
EXCLUIR_MODELOS = ["svm", "ridge"]

ESCENARIOS = {"sin_puntaje": False, "con_puntaje": True}   # nombre -> incluir_puntaje

RAIZ = Path(__file__).resolve().parents[1]
DIR_REPORTES = RAIZ / "reportes"
DIR_MODELOS = RAIZ / "models"

ETIQUETAS = ["No paga (0)", "Paga (1)"]
METRICAS_GRAFICO = ["AUC", "Especificidad", "Precision", "Recall", "F1"]


def especificidad(y_true, y_pred, **kwargs) -> float:
    """% de clientes que NO pagan y el modelo rechaza (1 - tasa de falsos positivos).

    `**kwargs` es necesario: PyCaret pasa argumentos extra y, si la función falla, registra 0 sin avisar.
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def summarize_classification(y_true: pd.Series, proba: np.ndarray, nombre: str, umbral: float = UMBRAL) -> dict:
    """Métricas de clasificación sobre un conjunto de datos (clase positiva = paga a tiempo)."""
    y_pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "modelo": nombre,
        "AUC": roc_auc_score(y_true, proba),
        "Especificidad": especificidad(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "KS": float(np.max(tpr - fpr)),
        "VP (paga, aprobado)": tp,
        "FP (no paga, aprobado)": fp,
        "FN (paga, rechazado)": fn,
        "VN (no paga, rechazado)": tn,
    }


def build_model(modelo, n_iter: int = N_ITER_TUNING) -> tuple[object, float]:
    """Optimiza los hiperparámetros de un modelo con `tune_model` según la métrica objetivo.

    Devuelve el modelo optimizado y su métrica media en validación cruzada. `choose_better=True`
    conserva el modelo original si ninguna combinación lo supera.
    """
    ajustado = tune_model(modelo, optimize=METRICA_OBJETIVO, n_iter=n_iter, choose_better=True, verbose=VERBOSE)
    cv = pull()
    return ajustado, float(cv.loc["Mean", METRICA_OBJETIVO])


def nombre_modelo(modelo) -> str:
    return type(modelo).__name__


def probabilidad_pago(modelo) -> tuple[pd.Series, np.ndarray]:
    """Probabilidad de pagar a tiempo sobre el test de PyCaret (hold-out temporal)."""
    pred = predict_model(modelo, raw_score=True, verbose=False)
    return pred[TARGET], pred["prediction_score_1"].to_numpy()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_metricas(resumen: pd.DataFrame, ruta: Path, titulo: str) -> None:
    datos = resumen.set_index("modelo")[METRICAS_GRAFICO]
    ax = datos.T.plot.bar(figsize=(10, 5), rot=0, width=0.8)
    ax.set(title=titulo, ylabel="Valor (0 a 1)", ylim=(0, 1.05))
    ax.legend(title="Modelo", fontsize=8)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def grafico_roc(curvas: dict[str, tuple[pd.Series, np.ndarray]], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for nombre, (y, proba) in curvas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set(title="Curvas ROC en test", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def grafico_matrices(curvas: dict[str, tuple[pd.Series, np.ndarray]], ruta: Path) -> None:
    fig, axes = plt.subplots(1, len(curvas), figsize=(5 * len(curvas), 4.5))
    for ax, (nombre, (y, proba)) in zip(np.atleast_1d(axes), curvas.items()):
        cm = confusion_matrix(y, (proba >= UMBRAL).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=nombre, xticks=[0, 1], yticks=[0, 1], xticklabels=ETIQUETAS, yticklabels=ETIQUETAS,
               xlabel="Predicción del modelo", ylabel="Realidad")
        ax.grid(False)
    fig.suptitle(f"Matrices de confusión en test (umbral {UMBRAL})")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def grafico_umbral(y: pd.Series, proba: np.ndarray, nombre: str, ruta: Path) -> None:
    """Cuántos falsos positivos y falsos negativos deja cada umbral de aprobación."""
    umbrales = np.linspace(0.05, 0.99, 95)
    fp = [int(((proba >= u) & (y == 0)).sum()) for u in umbrales]
    fn = [int(((proba < u) & (y == 1)).sum()) for u in umbrales]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(umbrales, fp, label="Falsos positivos (no paga y se aprueba)", color="tab:red")
    ax.plot(umbrales, fn, label="Falsos negativos (paga y se rechaza)", color="tab:blue")
    ax.axvline(UMBRAL, ls="--", color="gray", label=f"Umbral actual ({UMBRAL})")
    ax.set(title=f"Efecto del umbral de aprobación — {nombre}", xlabel="Probabilidad mínima de pago para aprobar",
           ylabel="Clientes en test")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def preparar_tablas(incluir_puntaje: bool) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Aplica el ColumnTransformer (ajustado con train) y arma los DataFrames para PyCaret."""
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos(incluir_puntaje=incluir_puntaje)
    preprocesador.set_output(transform="pandas")
    train = preprocesador.fit_transform(X_train).assign(**{TARGET: y_train})
    test = preprocesador.transform(X_test).assign(**{TARGET: y_test})
    return train, test, preprocesador


def ejecutar_escenario(escenario: str, incluir_puntaje: bool) -> pd.DataFrame:
    dir_out = DIR_REPORTES / escenario
    dir_out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nEscenario: {escenario}\n{'=' * 70}", flush=True)

    train, test, preprocesador = preparar_tablas(incluir_puntaje)

    # Los datos ya llegan imputados y codificados: PyCaret solo agrega el balanceo de clases,
    # que se aplica únicamente sobre los folds de entrenamiento.
    setup(
        data=train, test_data=test, target=TARGET, session_id=SEMILLA,
        imputation_type=None, fix_imbalance=True,
        fold=FOLDS, fold_strategy="stratifiedkfold",
        html=False, verbose=False,
    )
    add_metric("Especificidad", "Especificidad", especificidad)

    # 1) Comparar muchos modelos de una sola vez
    top = compare_models(sort=METRICA_OBJETIVO, n_select=TOP_N, exclude=EXCLUIR_MODELOS, verbose=VERBOSE)
    comparacion = pull()
    comparacion.to_csv(dir_out / "comparacion_modelos_cv.csv")
    print("Comparación en validación cruzada (train):")
    print(comparacion[["Model", "AUC", "Especificidad", "Prec.", "Recall", "F1"]].round(3).to_string(), flush=True)

    # 2) Optimizar hiperparámetros de los mejores y evaluar en test
    filas, curvas, ajustados = [], {}, {}
    for modelo in top:
        nombre = nombre_modelo(modelo)
        print(f"\nOptimizando {nombre}...", flush=True)
        y, proba = probabilidad_pago(modelo)
        filas.append({**summarize_classification(y, proba, f"{nombre} (base)"), "AUC cv": np.nan})

        modelo_ajustado, auc_cv = build_model(modelo)
        y, proba = probabilidad_pago(modelo_ajustado)
        filas.append({**summarize_classification(y, proba, f"{nombre} (tuned)"), "AUC cv": auc_cv})
        curvas[nombre] = (y, proba)
        ajustados[nombre] = (modelo_ajustado, auc_cv)

    resumen = pd.DataFrame(filas)
    resumen.to_csv(dir_out / "resumen_test.csv", index=False)
    print("\nResumen en test (clase positiva = paga a tiempo):")
    print(resumen.round(3).to_string(index=False), flush=True)

    # 3) Gráficos comparativos de los modelos optimizados
    tuned = resumen[resumen["modelo"].str.endswith("(tuned)")].assign(
        modelo=lambda t: t["modelo"].str.replace(" (tuned)", "", regex=False))
    grafico_metricas(tuned, dir_out / "comparacion_metricas.png",
                     f"Modelos optimizados en test — {escenario.replace('_', ' ')}")
    grafico_roc(curvas, dir_out / "curvas_roc.png")
    grafico_matrices(curvas, dir_out / "matrices_confusion.png")

    # 4) El mejor se elige por AUC en validación cruzada (no en test, para no sobreajustar al test)
    mejor_nombre = max(ajustados, key=lambda n: ajustados[n][1])
    mejor = ajustados[mejor_nombre][0]
    grafico_umbral(*curvas[mejor_nombre], mejor_nombre, dir_out / "umbral_fp_fn.png")
    try:
        ruta_imp = plot_model(mejor, plot="feature", save=str(dir_out))
        Path(ruta_imp).replace(dir_out / "importancia_variables.png")
    except (TypeError, ValueError, AttributeError):
        print(f"{mejor_nombre} no expone importancia de variables.")

    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(Pipeline([("preprocesador", preprocesador), ("modelo", mejor)]),
                DIR_MODELOS / f"mejor_modelo_{escenario}.joblib")
    print(f"\nMejor modelo ({escenario}): {mejor_nombre} | AUC cv {ajustados[mejor_nombre][1]:.3f}", flush=True)
    return resumen.assign(escenario=escenario, es_mejor=resumen["modelo"] == f"{mejor_nombre} (tuned)")


def main() -> None:
    resultados = pd.concat([ejecutar_escenario(e, incluir) for e, incluir in ESCENARIOS.items()])
    resultados.to_csv(DIR_REPORTES / "resumen_escenarios.csv", index=False)
    print("\nMejor modelo por escenario (métricas en test):")
    print(resultados[resultados["es_mejor"]].drop(columns="es_mejor").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
