# Misma versión de Python con la que se entrenó el modelo
FROM python:3.11.9-slim-bookworm

# XGBoost necesita la librería de OpenMP, que la imagen slim no trae
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Las dependencias van primero: si solo cambia el código, Docker reutiliza esta capa
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY mlops_pipeline/src/ft_engineering.py mlops_pipeline/src/model_deploy.py mlops_pipeline/src/
COPY mlops_pipeline/models/mejor_modelo.joblib mlops_pipeline/models/

# El servicio no necesita permisos de administrador
RUN useradd --create-home --uid 1000 api
USER api

# El joblib referencia `ft_engineering`: se arranca desde su carpeta para que se pueda importar
WORKDIR /app/mlops_pipeline/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
