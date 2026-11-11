# Modelo de riesgo crediticio

## ¿Qué busca este proyecto?

Anticipar si un **nuevo cliente pagará su crédito a tiempo**, para decidir si es elegible antes de aprobarlo. El modelo aprende de 10 763 créditos otorgados entre noviembre de 2024 y abril de 2026, de los que ya se sabe si se pagaron a tiempo (`Pago_atiempo = 1`) o no (`0`).

## Lo que nos dijeron los datos

- **La gran mayoría paga:** 95 de cada 100 clientes pagan a tiempo. Por eso el "acierto" general engaña: un sistema que aprobara a todo el mundo acertaría el 95 % de las veces y no detectaría a ningún cliente que no paga.
- **Había datos erróneos escondidos**, como edades de 122 años, salarios de miles de millones o textos donde debía haber números. Se marcaron como "dato desconocido" en lugar de usarlos tal cual.
- **Que falte un dato también es una señal:** los clientes sin historial de ingresos en la central de riesgo (27 %) pagan un poco peor.
- **Señales de mayor riesgo:** tener mora vigente, muchas consultas recientes en la central, menor puntaje en la central, ingresos decrecientes, ser más joven y que el desembolso se haga en fin de semana. Todas tienen efectos **pequeños**.
- **`puntaje` (puntaje interno) se excluye del modelo:** separa perfectamente a quienes pagan de quienes no (*puntaje ≥ 63 paga, menos no paga*, sin ninguna excepción en los 10 763 créditos). Una variable así casi seguro se calcula con el propio resultado de pago y **no estará disponible al evaluar a un cliente nuevo** (fuga de información). Con ella, cualquier modelo da un resultado perfecto pero inútil. Se revisaron todas las demás variables y ninguna tiene una relación parecida: la más fuerte, el puntaje de la central de riesgo, ordena a los clientes solo de forma modesta (AUC 0.62 por sí sola).

## Cómo se construyó el modelo

1. **Limpieza y nuevas variables** ([ft_engineering.py](mlops_pipeline/src/ft_engineering.py)): se corrigen los datos erróneos, se crean indicadores útiles para el negocio, como el peso de la cuota sobre el salario o si el cliente tiene mora, y se llevan los montos a una escala comparable.
2. **Prueba realista:** el modelo aprende con los créditos más antiguos (80 %) y se evalúa con los **más recientes** (20 %, 2 153 créditos), como si fueran clientes nuevos que nunca vio.
3. **Comparación de 14 modelos** con PyCaret, incluidas versiones de XGBoost y LightGBM con árboles poco profundos, que memorizan menos ([model_training_evaluation.py](mlops_pipeline/src/model_training_evaluation.py)). La validación también respeta el orden temporal. A los 3 mejores se les intenta afinar la configuración, y el cambio solo se acepta si mejora el AUC de validación en al menos 0.01.
4. **Dos medidas, cada una para una decisión:** el **AUC** elige el modelo, porque indica qué tan bien ordena a los clientes del más confiable al más riesgoso (0.5 equivale a lanzar una moneda y 1.0 es perfecto). El **F1 de "no paga"** elige el punto de corte, usando solo los datos de entrenamiento. El F1 de "paga" no sirve aquí: aprobar a todos ya da 0.98.

## ¿Qué error es peor?

| Error | Qué pasa | Costo para el negocio |
|---|---|---|
| **Falso positivo** | El modelo dice "paga", se aprueba… y no paga | **Alto:** se pierde el capital prestado |
| **Falso negativo** | El modelo dice "no paga", se rechaza… y sí habría pagado | Menor: se pierde la ganancia de ese crédito |

**Conviene tolerar más falsos negativos que falsos positivos.** Es preferible rechazar a algunos buenos clientes que prestarle a quien no va a pagar. El punto de corte (la probabilidad mínima de pago exigida para aprobar) se puede subir para reducir los falsos positivos, a cambio de rechazar a más clientes buenos. Ese equilibrio lo debe fijar el negocio según cuánto cuesta cada error.

## Resultados

Los 3 mejores modelos, evaluados con los 2 153 créditos más recientes (sin `puntaje`):

| Modelo | AUC (clientes nuevos) | De 69 clientes que no pagaron, ¿cuántos detecta? | Clientes buenos rechazados | F1 "no paga" |
|---|---|---|---|---|
| **XGBoost** (elegido) | 0.67 | 19 (28 %) | 231 de 2 084 (11 %) | 0.12 |
| LightGBM | 0.67 | 19 (28 %) | 230 de 2 084 (11 %) | 0.12 |
| Linear Discriminant Analysis | 0.64 | 20 (29 %) | 346 de 2 084 (17 %) | 0.09 |

**El modelo todavía es débil (AUC 0.67).** Para detectar a 19 de los 69 malos pagadores rechaza a 231 clientes buenos: unos 12 buenos por cada malo detectado. Afinar la configuración no alcanzó la mejora mínima en ningún modelo, así que se conservaron las versiones sin afinar.

**Hay que leer estas cifras con cautela.** Con solo 69 malos pagadores, el AUC real podría estar entre 0.61 y 0.73 (intervalo de confianza del 95 %), así que las diferencias pequeñas entre modelos no son concluyentes. Además, parte de las decisiones de diseño se tomaron después de ver este conjunto de prueba, por lo que el 0.67 puede ser algo optimista.

![Matrices de confusión](mlops_pipeline/reportes/matrices_confusion.png)

La siguiente gráfica muestra el equilibrio entre los dos errores según el punto de corte. Para bajar los falsos positivos (rojo) hay que exigir una probabilidad muy alta, y entonces los clientes buenos rechazados (azul) se disparan.

![Efecto del umbral](mlops_pipeline/reportes/umbral_fp_fn.png)

Más gráficos (comparación de métricas, curvas ROC, variables más importantes) y las tablas completas están en [mlops_pipeline/reportes/](mlops_pipeline/reportes/).

## Conclusión

- **Todavía no hay un modelo listo para decidir solo** si se aprueba un crédito. La información disponible solo permite distinguir de forma modesta a quienes pagarán.
- Por ahora puede servir como **apoyo**: una alerta para que un analista revise con más cuidado las solicitudes de mayor riesgo.

## Monitoreo del modelo

Un modelo se entrena con los clientes de ayer y decide sobre los de hoy. El monitoreo compara cada período con los 8 610 créditos de entrenamiento para avisar si la población cambió, sin esperar a saber quién pagó.

**Cómo se mide.** Cada variable se compara con cuatro métricas: PSI y Jensen-Shannon miden cuánto cambió; Kolmogorov-Smirnov (numéricas) y chi-cuadrado (categóricas) indican si la diferencia es significativa. El semáforo usa solo el PSI (verde menor a 0.10, amarillo hasta 0.25, rojo desde 0.25), porque con muchos datos las pruebas estadísticas marcan incluso diferencias mínimas.

**Cuándo se alerta.** Se sugiere evaluar un reentrenamiento si la probabilidad que da el modelo o una de sus 10 variables más importantes está en rojo dos períodos seguidos. Un primer rojo pide revisar la variable.

Las 10 variables más importantes según el XGBoost son `carga_cuota_ingreso_dc`, `ratio_ingreso_declarado_vs_dc`, `puntaje_datacredito`, `tipo_laboral`, `promedio_ingresos_datacredito`, `consultas_altas`, `total_otros_prestamos`, `plazo_meses`, `tipo_credito` y `huella_consulta`.

**Dos escenarios, que no se mezclan:**

| Escenario | Qué son | Qué responde |
|---|---|---|
| **Créditos reales** | Los 2 153 créditos de test por período, tratados como si fueran producción | ¿Cambiaron de verdad los clientes? |
| **Simulación** | Lotes alterados a propósito con intensidad creciente y de forma reproducible | ¿El monitor detecta un cambio cuando lo hay? |

**Qué se encontró en los créditos reales (julio de 2025 a abril de 2026):**

- Hay drift persistente: `plazo_meses`, `promedio_ingresos_datacredito` y `total_otros_prestamos` están en rojo en 7 de los 8 períodos, y la regla sugiere evaluar un reentrenamiento desde agosto de 2025.
- Cambió la mezcla de plazos: los créditos a 12 meses pasaron del 28 % al 11 %, y los de 6 meses, del 28 % al 44 %.
- El modelo rechaza más: del 6.0 % en entrenamiento a entre 8.3 % y 15.7 % por período.
- No se sabe aún si estos cambios son de la población o de cómo se registran los datos. Hay que revisarlo con el negocio antes de reentrenar.

**Qué tan confiable es el monitor.** En la simulación, las variables alteradas suben con la intensidad (`puntaje_datacredito` pasa de PSI 0.02 a 1.49) y las no alteradas, como `edad_cliente`, siguen en verde. Al comparar subconjuntos aleatorios de entrenamiento contra el resto, donde no hay cambio real, no apareció ningún rojo, ni siquiera con 127 créditos.

**Límites.** Drift no significa que el modelo haya empeorado: para saberlo hay que conocer quién pagó, y con 69 malos pagadores en test no se puede medir el AUC por período. Además, como el modelo ya es débil, reentrenarlo podría no ser suficiente.

## API de predicción y Docker

El modelo se publica como un servicio web: otro sistema le envía los datos de uno o varios créditos y recibe la predicción. Todo va dentro de una imagen Docker, así que para usarlo no hace falta instalar Python ni las librerías.

**Qué hace** ([model_deploy.py](mlops_pipeline/src/model_deploy.py), con FastAPI):

1. Al arrancar carga `mejor_modelo.joblib` (preprocesador + XGBoost + punto de corte), el archivo que guarda `model_training_evaluation.py`.
2. Recibe los datos crudos de los créditos, en JSON o CSV, hasta 10 000 por solicitud.
3. Les aplica la misma limpieza y las mismas variables del entrenamiento (`limpiar` y `crear_atributos` de [ft_engineering.py](mlops_pipeline/src/ft_engineering.py)).
4. Devuelve, por cada crédito, la probabilidad de pagar a tiempo y la predicción según el punto de corte.

| Ruta | Para qué sirve |
|---|---|
| `GET /health` | Comprobar que el servicio está activo y qué modelo cargó |
| `POST /predict` | Predicción por lotes, en JSON o CSV |
| `GET /docs` | Documentación interactiva para probar la API desde el navegador |

**Qué datos recibe.** Las mismas columnas de la base de datos, sin `Pago_atiempo`:

- **Obligatorias (14):** `tipo_credito`, `fecha_prestamo`, `capital_prestado`, `plazo_meses`, `edad_cliente`, `tipo_laboral`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo` y `creditos_sectorReal`.
- **Opcionales (5):** `puntaje_datacredito`, `saldo_mora`, `saldo_total`, `promedio_ingresos_datacredito` y `tendencia_ingresos`. También faltan en algunos créditos de entrenamiento, y el modelo sabe tratarlas.
- **Se aceptan, pero se ignoran:** `puntaje` (fuga de información), `saldo_principal`, `saldo_mora_codeudor` y `Pago_atiempo`. Así se puede enviar un archivo con el mismo formato de la base.
- **Cualquier otra columna se rechaza.** Un nombre mal escrito se convertiría en un dato faltante y cambiaría la predicción sin avisar.

Si algún dato está mal, la API no predice nada y responde qué fila y qué campo hay que corregir.

Ejemplo de solicitud en JSON (en CSV es lo mismo: una fila por crédito y la primera fila con los nombres de las columnas):

```json
{"registros": [
  {"tipo_credito": 9, "fecha_prestamo": "2026-01-08 12:22:40", "capital_prestado": 5974028.4,
   "plazo_meses": 10, "edad_cliente": 36, "tipo_laboral": "Independiente", "salario_cliente": 4036000,
   "total_otros_prestamos": 829000, "cuota_pactada": 529554, "cant_creditosvigentes": 4,
   "huella_consulta": 5, "creditos_sectorFinanciero": 3, "creditos_sectorCooperativo": 0,
   "creditos_sectorReal": 0, "puntaje_datacredito": 740, "saldo_mora": 0, "saldo_total": 18702}
]}
```

Respuesta:

```json
{"modelo": "XGBClassifier", "umbral": 0.877, "total": 1,
 "predicciones": [{"fila": 0, "probabilidad_pago": 0.9318, "prediccion": 1, "etiqueta": "Paga a tiempo"}]}
```

`prediccion` vale 1 si la probabilidad de pago llega al punto de corte y 0 si no. Como el modelo todavía es débil, un 0 (*"Riesgo de no pago: revisar"*) es una alerta para que un analista revise la solicitud, no un rechazo automático.

**Construir y ejecutar la imagen.** Con Docker Desktop abierto, desde la carpeta raíz del proyecto (donde está el [Dockerfile](Dockerfile)):

```powershell
docker build -t modelo-riesgo-api:1.0 .
docker run -d --name riesgo-api -p 8000:8000 modelo-riesgo-api:1.0
```

Después se abre http://localhost:8000/docs, se elige `POST /predict` y *Try it out*. También se puede llamar desde PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "text/csv" -InFile creditos.csv
```

Para detener el servicio: `docker rm -f riesgo-api`.

**Qué contiene la imagen.** Python 3.11.9 (la misma versión del entrenamiento), las librerías de [requirements-api.txt](requirements-api.txt), `ft_engineering.py`, `model_deploy.py` y `mejor_modelo.joblib`. El [.dockerignore](.dockerignore) deja fuera todo lo demás (entorno virtual, historial de git y datos de clientes). El servicio corre con un usuario sin permisos de administrador, y Docker comprueba cada 30 segundos que siga respondiendo.

- **Por qué un `requirements-api.txt` aparte.** `requirements.txt` instala todo el proyecto (PyCaret, Jupyter, Streamlit), pero la API solo necesita lo que se usa para predecir. Así la imagen pesa mucho menos y se construye más rápido.
- **Versiones fijas.** `requirements-api.txt` fija numpy, pandas, scikit-learn y xgboost en las versiones del entrenamiento. Con otras, el modelo podría no cargar o dar probabilidades distintas sin avisar.
- **Si se reentrena el modelo**, hay que volver a construir la imagen para que incluya el nuevo `mejor_modelo.joblib`.

**Comprobación.** Al enviar a la API en Docker los 2 153 créditos de test, reproduce exactamente la evaluación: AUC 0.67, 19 de 69 malos pagadores detectados y 231 clientes buenos rechazados.

---

## Estructura y ejecución

```
├── mlops_pipeline/
│   ├── src/
│   │   ├── Cargar_datos.ipynb              # lectura de Base_de_datos.xlsx
│   │   ├── comprension_eda.ipynb           # análisis exploratorio
│   │   ├── ft_engineering.py               # limpieza, variables, train/test, ColumnTransformer
│   │   ├── model_training_evaluation.py    # comparación, ajuste y evaluación con PyCaret
│   │   ├── model_deploy.py                 # API de predicción por lotes (FastAPI)
│   │   ├── model_monitoring.py             # pronósticos por período y métricas de data drift
│   │   └── app_monitoreo.py                # tablero de monitoreo en Streamlit
│   ├── reportes/                           # gráficos y tablas de resultados
│   ├── models/                             # mejor modelo: preprocesador + modelo + punto de corte (.joblib)
│   └── monitoreo/                          # resultados agregados del monitoreo (predicciones.db queda fuera de git)
├── Base_de_datos.xlsx                      # datos de los 10 763 créditos
├── Dockerfile                              # imagen de la API
├── .dockerignore                           # qué archivos entran a la imagen
├── requirements.txt                        # dependencias de todo el proyecto
├── requirements-api.txt                    # solo las que necesita la imagen de la API
└── README.md
```

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd mlops_pipeline\src
python model_training_evaluation.py   # entrena y guarda el modelo
python model_monitoring.py            # calcula el monitoreo; en producción se programa con la periodicidad definida
python model_deploy.py                # API sin Docker, en http://localhost:8000/docs
cd ..\..
streamlit run mlops_pipeline\src\app_monitoreo.py
```

Requiere Python 3.9–3.11 (límite de PyCaret 3.3). Los scripts y notebooks buscan `Base_de_datos.xlsx` desde su carpeta hacia arriba, así que encuentran la de la raíz del proyecto sin configurar rutas.
