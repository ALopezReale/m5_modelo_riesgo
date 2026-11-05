"""Entrenamiento y evaluación de los primeros modelos de riesgo crediticio con PyCaret.

Flujo:

1. Datos de `ft_engineering.preparar_datos()`: limpieza, atributos, división temporal y
   `ColumnTransformer` (imputación, log, escalado, codificación) ajustado solo con train. La
   variable `puntaje` queda excluida por ser fuga de información (separa el objetivo sin error).
2. `setup` de PyCaret con validación cruzada temporal (cada fold valida con créditos posteriores
   a los de entrenamiento, igual que el test).
3. `compare_models`: muchos clasificadores de una sola vez, ordenados por la métrica objetivo.
   Se suman versiones regularizadas de XGBoost y LightGBM (árboles poco profundos).
4. `build_model`: los mejores se optimizan con `tune_model`; el ajuste solo se acepta si mejora
   el AUC de validación cruzada en al menos `MEJORA_MINIMA_TUNING`.
5. `elegir_umbral`: punto de corte que maximiza el F1 de "no paga", calculado solo con train.
6. `summarize_classification`: métricas sobre el conjunto de test (datos que el modelo no vio).
7. Gráficos comparativos, matrices de confusión, curva de umbral y tabla resumen en `reportes/`.
8. Se guarda el mejor modelo (preprocesador + modelo) y su umbral en `models/`.

Objetivo: `Pago_atiempo` (1 = paga a tiempo -> se le puede aprobar el crédito).

- Falso positivo (FP): el modelo dice "paga", se aprueba y NO paga -> se pierde el capital. Es el
  error más caro, por eso se reporta la **Especificidad** (% de no pagadores que el modelo rechaza).
- Falso negativo (FN): el modelo dice "no paga", se rechaza y SÍ pagaba -> se pierde el margen.

Métricas:

- **AUC** (para elegir el modelo): mide qué tan bien ordena a los clientes de menor a mayor
  riesgo, sin depender del umbral ni del desbalance (95 % paga).
- **F1 no paga** (para elegir el umbral y comparar en test): equilibrio entre detectar a los
  malos pagadores y no rechazar de más. El F1 de "paga" no sirve: aprobar a todos ya da 0.98.

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
from lightgbm import LGBMClassifier
from pycaret.classification import (add_metric, compare_models, plot_model, predict_model, pull, setup,
                                    tune_model)
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ft_engineering import TARGET, preparar_datos

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
METRICA_OBJETIVO = "AUC"
TOP_N = 3                 # modelos que pasan de compare_models a tune_model
N_ITER_TUNING = 30        # combinaciones de hiperparámetros probadas por modelo
MEJORA_MINIMA_TUNING = 0.01   # con pocos malos pagadores, mejoras menores de AUC en CV suelen ser ruido
FOLDS = 5
SEMILLA = 42
VERBOSE = True            # muestra en la terminal las tablas de PyCaret mientras entrena

# Modelos de PyCaret que se comparan. Fuera: svm y ridge (sin predict_proba: no permiten AUC ni
# mover el umbral) y catboost (no instalado).
MODELOS_PYCARET = ["lr", "knn", "nb", "dt", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "dummy"]

# Árboles poco profundos: con ~500 malos pagadores, un árbol profundo memoriza casos sueltos.
# Profundidad 3 (máx. 8 hojas), hojas con muchos casos, aprendizaje lento, submuestreo y
# penalización L2 para que cada árbol aporte poco y el conjunto generalice.
MODELOS_REGULARIZADOS = [
    XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=3, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=5, random_state=SEMILLA, n_jobs=-1),
    LGBMClassifier(n_estimators=400, learning_rate=0.02, max_depth=3, num_leaves=8, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, random_state=SEMILLA,
                   n_jobs=-1, verbose=-1),
]

RAIZ = Path(__file__).resolve().parents[1]
DIR_REPORTES = RAIZ / "reportes"
DIR_MODELOS = RAIZ / "models"

ETIQUETAS = ["No paga (0)", "Paga (1)"]
METRICAS_GRAFICO = ["AUC", "F1 no paga", "Especificidad", "Precision", "Recall"]


def especificidad(y_true, y_pred, **kwargs) -> float:
    """% de clientes que NO pagan y el modelo rechaza (1 - tasa de falsos positivos).

    `**kwargs` es necesario: PyCaret pasa argumentos extra y, si la función falla, registra 0 sin avisar.
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def f1_no_paga(y_true, y_pred, **kwargs) -> float:
    """F1 de la clase minoritaria (no paga): combina cuántos malos pagadores se detectan y con qué acierto."""
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def summarize_classification(y_true: pd.Series, proba: np.ndarray, nombre: str, umbral: float) -> dict:
    """Métricas de clasificación sobre un conjunto de datos (clase positiva = paga a tiempo)."""
    y_pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "modelo": nombre,
        "umbral": umbral,
        "AUC": roc_auc_score(y_true, proba),
        "F1 no paga": f1_no_paga(y_true, y_pred),
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


def build_model(modelo, auc_base: float, n_iter: int = N_ITER_TUNING) -> tuple[object, float]:
    """Optimiza los hiperparámetros de un modelo con `tune_model` según la métrica objetivo.

    Devuelve el modelo final y su métrica media en validación cruzada: el optimizado si supera a
    `auc_base` en al menos `MEJORA_MINIMA_TUNING`; si no, el original.
    """
    ajustado = tune_model(modelo, optimize=METRICA_OBJETIVO, n_iter=n_iter, verbose=VERBOSE)
    auc_ajustado = float(pull().loc["Mean", METRICA_OBJETIVO])
    aceptado = auc_ajustado - auc_base >= MEJORA_MINIMA_TUNING
    print(f"AUC cv: base {auc_base:.4f}, ajustado {auc_ajustado:.4f} -> "
          f"{'se acepta el ajuste' if aceptado else 'se conserva el modelo base'}", flush=True)
    return (ajustado, auc_ajustado) if aceptado else (modelo, auc_base)


def elegir_umbral(modelo, train: pd.DataFrame) -> float:
    """Umbral de aprobación que maximiza el F1 de "no paga", usando solo datos de train.

    Las probabilidades salen de validación cruzada (cada crédito se predice con un modelo que no
    lo vio). Se usan folds estratificados porque los temporales no predicen el primer bloque.
    """
    X, y = train.drop(columns=TARGET), train[TARGET]
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEMILLA)
    proba = cross_val_predict(clone(modelo), X, y, cv=cv, method="predict_proba")[:, 1]
    # Candidatos por percentil (rechazar del 1 % al 50 % más riesgoso): no depende de la escala de
    # las probabilidades, que cambia mucho entre modelos (algunos dan > 0.99 a casi todos).
    umbrales = np.quantile(proba, np.linspace(0.01, 0.50, 50))
    f1s = [f1_no_paga(y, (proba >= u).astype(int)) for u in umbrales]
    return float(umbrales[int(np.argmax(f1s))])


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


def grafico_roc(curvas: dict[str, tuple[pd.Series, np.ndarray, float]], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for nombre, (y, proba, _) in curvas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set(title="Curvas ROC en test", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def grafico_matrices(curvas: dict[str, tuple[pd.Series, np.ndarray, float]], ruta: Path) -> None:
    fig, axes = plt.subplots(1, len(curvas), figsize=(5 * len(curvas), 4.5))
    for ax, (nombre, (y, proba, umbral)) in zip(np.atleast_1d(axes), curvas.items()):
        cm = confusion_matrix(y, (proba >= umbral).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=f"{nombre}\n(umbral {umbral:.3f})", xticks=[0, 1], yticks=[0, 1], xticklabels=ETIQUETAS,
               yticklabels=ETIQUETAS, xlabel="Predicción del modelo", ylabel="Realidad")
        ax.grid(False)
    fig.suptitle("Matrices de confusión en test")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def grafico_umbral(y: pd.Series, proba: np.ndarray, umbral: float, nombre: str, ruta: Path) -> None:
    """Cuántos falsos positivos y falsos negativos deja cada umbral de aprobación."""
    umbrales = np.linspace(0.05, 0.99, 95)
    fp = [int(((proba >= u) & (y == 0)).sum()) for u in umbrales]
    fn = [int(((proba < u) & (y == 1)).sum()) for u in umbrales]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(umbrales, fp, label="Falsos positivos (no paga y se aprueba)", color="tab:red")
    ax.plot(umbrales, fn, label="Falsos negativos (paga y se rechaza)", color="tab:blue")
    ax.axvline(umbral, ls="--", color="gray", label=f"Umbral elegido ({umbral:.3f})")
    ax.set(title=f"Efecto del umbral de aprobación — {nombre}", xlabel="Probabilidad mínima de pago para aprobar",
           ylabel="Clientes en test")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def preparar_tablas() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Aplica el ColumnTransformer (ajustado con train) y arma los DataFrames para PyCaret."""
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos()
    preprocesador.set_output(transform="pandas")
    train = preprocesador.fit_transform(X_train).assign(**{TARGET: y_train})
    test = preprocesador.transform(X_test).assign(**{TARGET: y_test})
    return train, test, preprocesador


def main() -> None:
    dir_out = DIR_REPORTES
    dir_out.mkdir(parents=True, exist_ok=True)

    train, test, preprocesador = preparar_tablas()

    # Los datos ya llegan imputados, escalados y codificados. train viene ordenado por fecha, así que
    # los folds temporales validan siempre con créditos posteriores a los de entrenamiento.
    setup(
        data=train, test_data=test, target=TARGET, session_id=SEMILLA,
        imputation_type=None,
        fold=FOLDS, fold_strategy="timeseries", data_split_shuffle=False, fold_shuffle=False,
        html=False, verbose=False,
    )
    add_metric("Especificidad", "Especificidad", especificidad)
    add_metric("F1 no paga", "F1 no paga", f1_no_paga)

    # 1) Comparar muchos modelos de una sola vez
    top = compare_models(include=MODELOS_PYCARET + MODELOS_REGULARIZADOS, sort=METRICA_OBJETIVO, n_select=TOP_N,
                         verbose=VERBOSE)
    comparacion = pull()
    comparacion.to_csv(dir_out / "comparacion_modelos_cv.csv")
    print("Comparación en validación cruzada temporal (train):")
    print(comparacion[["Model", "AUC", "F1 no paga", "Especificidad", "Prec.", "Recall"]].round(3).to_string(),
          flush=True)

    # 2) Optimizar hiperparámetros de los mejores, elegir su umbral con train y evaluar en test
    filas, curvas, ajustados = [], {}, {}
    for i, modelo in enumerate(top):
        nombre = nombre_modelo(modelo)
        if nombre in curvas:                     # p. ej. XGBoost de PyCaret y el regularizado
            nombre = f"{nombre} {i + 1}"
        print(f"\nOptimizando {nombre}...", flush=True)
        y, proba = probabilidad_pago(modelo)
        filas.append({**summarize_classification(y, proba, f"{nombre} (base)", elegir_umbral(modelo, train)),
                      "AUC cv": np.nan})

        # comparacion está ordenada igual que top (por la métrica objetivo)
        modelo_final, auc_cv = build_model(modelo, float(comparacion[METRICA_OBJETIVO].iloc[i]))
        umbral = elegir_umbral(modelo_final, train)
        y, proba = probabilidad_pago(modelo_final)
        filas.append({**summarize_classification(y, proba, f"{nombre} (final)", umbral), "AUC cv": auc_cv})
        curvas[nombre] = (y, proba, umbral)
        ajustados[nombre] = (modelo_final, auc_cv)

    resumen = pd.DataFrame(filas)
    resumen.to_csv(dir_out / "resumen_test.csv", index=False)
    print("\nResumen en test (clase positiva = paga a tiempo):")
    print(resumen.round(3).to_string(index=False), flush=True)

    # 3) Gráficos comparativos de los modelos finales
    finales = resumen[resumen["modelo"].str.endswith("(final)")].assign(
        modelo=lambda t: t["modelo"].str.replace(" (final)", "", regex=False))
    grafico_metricas(finales, dir_out / "comparacion_metricas.png", "Modelos finales en test")
    grafico_roc(curvas, dir_out / "curvas_roc.png")
    grafico_matrices(curvas, dir_out / "matrices_confusion.png")

    # 4) El mejor se elige por AUC en validación cruzada (no en test, para no sobreajustar al test)
    mejor_nombre = max(ajustados, key=lambda n: ajustados[n][1])
    mejor = ajustados[mejor_nombre][0]
    umbral_mejor = curvas[mejor_nombre][2]
    grafico_umbral(*curvas[mejor_nombre], mejor_nombre, dir_out / "umbral_fp_fn.png")
    try:
        ruta_imp = plot_model(mejor, plot="feature", save=str(dir_out))
        Path(ruta_imp).replace(dir_out / "importancia_variables.png")
    except (TypeError, ValueError, AttributeError):
        print(f"{mejor_nombre} no expone importancia de variables.")

    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": Pipeline([("preprocesador", preprocesador), ("modelo", mejor)]), "umbral": umbral_mejor},
                DIR_MODELOS / "mejor_modelo.joblib")
    print(f"\nMejor modelo: {mejor_nombre} | AUC cv {ajustados[mejor_nombre][1]:.3f} | umbral {umbral_mejor:.3f}",
          flush=True)
    print(resumen[resumen["modelo"] == f"{mejor_nombre} (final)"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
