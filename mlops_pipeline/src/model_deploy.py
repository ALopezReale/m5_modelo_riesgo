"""API de predicción por lotes del modelo de riesgo crediticio (FastAPI).

Carga `models/mejor_modelo.joblib` (preprocesador + modelo + umbral) y expone:

- `GET /health`: estado del servicio y modelo cargado.
- `POST /predict`: recibe uno o varios créditos y devuelve su probabilidad de pago.
  - JSON (`Content-Type: application/json`): `{"registros": [{...}, {...}]}`
  - CSV (`Content-Type: text/csv`): una fila por crédito, separado por comas.

Se envían los datos crudos del crédito: la API aplica `limpiar` y `crear_atributos` de
`ft_engineering`, las mismas reglas del entrenamiento. Las columnas de `COLUMNAS_IGNORADAS` se
aceptan pero no se usan (`puntaje` es fuga de información); cualquier otra columna desconocida se
rechaza, porque un nombre mal escrito se convertiría en un dato faltante sin que nadie lo note.

El modelo es débil (AUC 0.67): una predicción de "no paga" es una alerta para que un analista
revise la solicitud, no un rechazo automático.

Se ejecuta con `python model_deploy.py` desde esta carpeta. Documentación interactiva en
http://localhost:8000/docs
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator

from ft_engineering import COLUMNAS_ELIMINAR, TARGET, crear_atributos, limpiar

RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "mejor_modelo.joblib"
MAX_REGISTROS = 10_000
MAX_ERRORES_REPORTADOS = 50

COLUMNAS_IGNORADAS = {TARGET, *COLUMNAS_ELIMINAR}
ETIQUETAS = {0: "Riesgo de no pago: revisar", 1: "Paga a tiempo"}
TIPOS_JSON = {"application/json"}
TIPOS_CSV = {"text/csv", "application/csv"}


class Credito(BaseModel):
    """Datos crudos de un crédito. Son opcionales solo los campos que también faltan en la base de entrenamiento."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, coerce_numbers_to_str=True)

    tipo_credito: int
    fecha_prestamo: datetime
    capital_prestado: float
    plazo_meses: float
    edad_cliente: float
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    cant_creditosvigentes: float
    huella_consulta: float
    creditos_sectorFinanciero: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float
    puntaje_datacredito: float | None = None
    saldo_mora: float | None = None
    saldo_total: float | None = None
    promedio_ingresos_datacredito: float | None = None
    tendencia_ingresos: str | None = None

    @field_validator("fecha_prestamo", mode="before")
    @classmethod
    def rechazar_fecha_numerica(cls, valor):
        """Pydantic leería un número como segundos desde 1970: daría una fecha válida pero equivocada."""
        try:
            float(valor)
        except (TypeError, ValueError):
            return valor
        raise ValueError("la fecha debe ser texto, por ejemplo 2026-01-08 12:22:40")

    @field_validator("fecha_prestamo")
    @classmethod
    def quitar_zona_horaria(cls, valor: datetime) -> datetime:
        """El modelo se entrenó con fechas sin zona horaria."""
        return valor.replace(tzinfo=None)


class Prediccion(BaseModel):
    fila: int
    probabilidad_pago: float
    prediccion: int
    etiqueta: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    umbral: float
    total: int
    predicciones: list[Prediccion]


VALIDADOR_REGISTROS = TypeAdapter(list[Credito])

EJEMPLO = {
    "tipo_credito": 9, "fecha_prestamo": "2026-01-08 12:22:40", "capital_prestado": 5974028.4, "plazo_meses": 10,
    "edad_cliente": 36, "tipo_laboral": "Independiente", "salario_cliente": 4036000, "total_otros_prestamos": 829000,
    "cuota_pactada": 529554, "cant_creditosvigentes": 4, "huella_consulta": 5, "creditos_sectorFinanciero": 3,
    "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0, "puntaje_datacredito": 740, "saldo_mora": 0,
    "saldo_total": 18702, "promedio_ingresos_datacredito": None, "tendencia_ingresos": None,
}
EJEMPLO_CSV = ",".join(EJEMPLO) + "\n" + ",".join("" if v is None else str(v) for v in EJEMPLO.values())


# ---------------------------------------------------------------------------
# Lectura y validación de la solicitud
# ---------------------------------------------------------------------------
def leer_json(cuerpo: bytes) -> list:
    try:
        datos = json.loads(cuerpo)
    except ValueError as error:
        raise HTTPException(400, f"El cuerpo no es un JSON válido: {error}") from error
    if not isinstance(datos, dict) or not isinstance(datos.get("registros"), list):
        raise HTTPException(422, 'El JSON debe tener la forma {"registros": [{...}, {...}]}.')
    return datos["registros"]


def leer_csv(cuerpo: bytes) -> list[dict]:
    try:
        # Todo se lee como texto: la conversión de tipos la hace `Credito`, igual que con JSON
        tabla = pd.read_csv(BytesIO(cuerpo), dtype=str, keep_default_na=False, encoding="utf-8-sig").fillna("")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(400, f"El cuerpo no es un CSV válido: {error}") from error
    return [{campo: valor.strip() or None for campo, valor in fila.items()} for fila in tabla.to_dict("records")]


def validar_registros(filas: list) -> list[Credito]:
    if not filas:
        raise HTTPException(422, "La solicitud no trae registros.")
    if len(filas) > MAX_REGISTROS:
        raise HTTPException(413, f"Se admiten como máximo {MAX_REGISTROS} registros por solicitud.")

    filas = [{k: v for k, v in fila.items() if k not in COLUMNAS_IGNORADAS} if isinstance(fila, dict) else fila
             for fila in filas]
    try:
        return VALIDADOR_REGISTROS.validate_python(filas)
    except ValidationError as error:
        detalle = [{"fila": e["loc"][0], "campo": ".".join(map(str, e["loc"][1:])), "error": e["msg"]}
                   for e in error.errors(include_url=False, include_context=False, include_input=False)]
        raise HTTPException(422, detalle[:MAX_ERRORES_REPORTADOS]) from error


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir(registros: list[Credito], artefacto: dict) -> RespuestaPrediccion:
    """Aplica las transformaciones del entrenamiento y el umbral guardado con el modelo."""
    pipeline, umbral = artefacto["pipeline"], artefacto["umbral"]

    # `limpiar` espera las columnas descartadas para luego eliminarlas
    crudos = pd.DataFrame([r.model_dump() for r in registros]).assign(**{c: np.nan for c in COLUMNAS_ELIMINAR})
    probabilidades = pipeline.predict_proba(crear_atributos(limpiar(crudos)))[:, 1]

    predicciones = []
    for fila, probabilidad in enumerate(probabilidades):
        clase = int(probabilidad >= umbral)
        predicciones.append(Prediccion(fila=fila, probabilidad_pago=round(float(probabilidad), 6), prediccion=clase,
                                       etiqueta=ETIQUETAS[clase]))

    return RespuestaPrediccion(modelo=type(pipeline.steps[-1][1]).__name__, umbral=umbral, total=len(predicciones),
                               predicciones=predicciones)


def procesar(cuerpo: bytes, tipo: str, artefacto: dict) -> RespuestaPrediccion:
    filas = leer_csv(cuerpo) if tipo in TIPOS_CSV else leer_json(cuerpo)
    return predecir(validar_registros(filas), artefacto)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
@asynccontextmanager
async def ciclo_de_vida(aplicacion: FastAPI):
    # Se carga al arrancar: si el modelo falta o no abre, el servicio falla de inmediato
    aplicacion.state.artefacto = joblib.load(RUTA_MODELO)
    yield


app = FastAPI(title="API de riesgo crediticio", version="1.0.0", lifespan=ciclo_de_vida,
              description="Predicción por lotes de la probabilidad de que un cliente pague a tiempo.")


@app.get("/health")
def health(request: Request) -> dict:
    artefacto = request.app.state.artefacto
    return {"estado": "ok", "modelo": type(artefacto["pipeline"].steps[-1][1]).__name__, "umbral": artefacto["umbral"]}


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    summary="Predicción por lotes (JSON o CSV)",
    # El cuerpo se lee a mano para aceptar dos formatos en una ruta, así que se documenta aquí
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/json": {
            "schema": {"type": "object", "required": ["registros"],
                       "properties": {"registros": {"type": "array", "items": Credito.model_json_schema()}}},
            "example": {"registros": [EJEMPLO]},
        },
        "text/csv": {"schema": {"type": "string"}, "example": EJEMPLO_CSV},
    }}},
)
async def predict(request: Request) -> RespuestaPrediccion:
    tipo = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if tipo not in TIPOS_JSON | TIPOS_CSV:
        raise HTTPException(415, "Use Content-Type application/json o text/csv.")
    cuerpo = await request.body()
    # En un hilo aparte: predecir un lote grande no debe bloquear las demás solicitudes
    return await run_in_threadpool(procesar, cuerpo, tipo, request.app.state.artefacto)


if __name__ == "__main__":
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000)
