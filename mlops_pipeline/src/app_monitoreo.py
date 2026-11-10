"""Tablero de monitoreo del modelo de riesgo crediticio.

Muestra lo que calcula `model_monitoring.py`: solo lee tablas agregadas de `monitoreo/`, nunca
datos individuales de clientes.

Se ejecuta desde la raíz del proyecto con `streamlit run mlops_pipeline/src/app_monitoreo.py`.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DIR_MONITOREO = Path(__file__).resolve().parents[1] / "monitoreo"
ARCHIVO_META = DIR_MONITOREO / "resumen_monitoreo.json"

ESCENARIOS = {"Créditos reales": "real", "Simulación de drift": "simulacion"}
COLOR_ESTADO = {"verde": "#0ca30c", "amarillo": "#fab219", "rojo": "#d03b3b"}
ETIQUETA_SEMAFORO = {"verde": "🟢 Verde", "amarillo": "🟡 Amarillo", "rojo": "🔴 Rojo"}
ICONO_ESTADO = {"Estable": "🟢", "Observar": "🟡", "Revisar": "🟠", "Reentrenar": "🔴"}
ICONO_TENDENCIA = {"creciente": "↗ Creciente", "decreciente": "↘ Decreciente", "estable": "→ Estable",
                   "sin datos suficientes": "Sin datos suficientes"}
COLOR_REFERENCIA, COLOR_ACTUAL = "#2a78d6", "#eb6834"
COLOR_SECUNDARIO = "#b4b2a9"
COLOR_CONTEXTO = "#898781"


@st.cache_data
def cargar_tablas(_marca: float) -> dict:
    """Lee los resultados; `_marca` (fecha del archivo) refresca la caché cuando el job vuelve a correr."""
    tablas = {nombre: pd.read_csv(DIR_MONITOREO / f"{nombre}.csv")
              for nombre in ["metricas_drift", "distribuciones", "alertas", "tendencias", "resumen_ventanas"]}
    tablas["meta"] = json.loads(ARCHIVO_META.read_text(encoding="utf-8"))
    return tablas


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def aplicar_estilo(figura: go.Figure, alto: int, titulo_x: str = "", titulo_y: str = "") -> go.Figure:
    figura.update_layout(height=alto, margin={"l": 10, "r": 30, "t": 40, "b": 10}, barcornerradius=4,
                         legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
                         hoverlabel={"namelength": -1})
    figura.update_xaxes(title_text=titulo_x, showgrid=False)
    figura.update_yaxes(title_text=titulo_y, showgrid=True, gridwidth=1)
    return figura


def grafico_barras_riesgo(datos: pd.DataFrame, meta: dict) -> go.Figure:
    """Barras de riesgo: PSI por variable, de mayor a menor, con el color del semáforo."""
    datos = datos.sort_values("psi")
    figura = go.Figure()
    for estado in ["rojo", "amarillo", "verde"]:
        grupo = datos[datos["semaforo"] == estado]
        figura.add_bar(y=grupo["variable"], x=grupo["psi"], orientation="h", name=ETIQUETA_SEMAFORO[estado],
                       marker_color=COLOR_ESTADO[estado], text=grupo["psi"].map("{:.2f}".format),
                       textposition="outside", hovertemplate="%{y}<br>PSI %{x:.3f}<extra></extra>")
    figura.add_vline(x=meta["psi_amarillo"], line_width=1, line_color=COLOR_ESTADO["amarillo"])
    figura.add_vline(x=meta["psi_rojo"], line_width=1, line_color=COLOR_ESTADO["rojo"])
    aplicar_estilo(figura, max(320, 24 * len(datos) + 90), titulo_x="PSI respecto de la referencia")
    figura.update_layout(bargap=0.3)
    figura.update_yaxes(showgrid=False, categoryorder="array", categoryarray=datos["variable"].tolist())
    figura.update_xaxes(range=[0, max(datos["psi"].max() * 1.2, meta["psi_rojo"] * 1.3)])
    return figura


def grafico_distribucion(datos: pd.DataFrame) -> go.Figure:
    """Distribución histórica contra la actual, en los mismos bins usados para el PSI."""
    datos = datos.sort_values("posicion")
    figura = go.Figure()
    plantilla = "%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>"
    figura.add_bar(x=datos["bin"], y=datos["prop_ref"], name="Histórica (referencia)", marker_color=COLOR_REFERENCIA,
                   hovertemplate=plantilla)
    figura.add_bar(x=datos["bin"], y=datos["prop_act"], name="Actual", marker_color=COLOR_ACTUAL,
                   hovertemplate=plantilla)
    aplicar_estilo(figura, 400, titulo_y="% de créditos")
    figura.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    figura.update_xaxes(type="category", categoryorder="array", categoryarray=datos["bin"].tolist())
    figura.update_yaxes(tickformat=".0%")
    return figura


def grafico_mapa_calor(datos: pd.DataFrame) -> go.Figure:
    """Semáforo de cada variable en cada ventana, con el valor del PSI dentro de la celda."""
    ventanas = datos.sort_values("orden")["ventana"].drop_duplicates().tolist()
    orden_variables = (datos.groupby("variable")["psi"].mean().sort_values(ascending=False).index.tolist())
    psi = datos.pivot(index="variable", columns="ventana", values="psi").loc[orden_variables, ventanas]
    niveles = psi.map(lambda v: 2 if v >= META["psi_rojo"] else 1 if v >= META["psi_amarillo"] else 0)
    escala = [[0, COLOR_ESTADO["verde"]], [0.33, COLOR_ESTADO["verde"]], [0.33, COLOR_ESTADO["amarillo"]],
              [0.66, COLOR_ESTADO["amarillo"]], [0.66, COLOR_ESTADO["rojo"]], [1, COLOR_ESTADO["rojo"]]]
    figura = go.Figure(go.Heatmap(z=niveles.to_numpy(), x=ventanas, y=orden_variables, text=psi.round(2).to_numpy(),
                                  texttemplate="%{text:.2f}", colorscale=escala, zmin=0, zmax=2, showscale=False,
                                  xgap=2, ygap=2, customdata=psi.to_numpy(),
                                  hovertemplate="%{y}<br>%{x}<br>PSI %{customdata:.3f}<extra></extra>"))
    aplicar_estilo(figura, max(280, 26 * len(orden_variables) + 90))
    figura.update_yaxes(showgrid=False, autorange="reversed")
    figura.update_xaxes(type="category", side="top")
    return figura


def grafico_evolucion(datos: pd.DataFrame, variable: str, meta: dict) -> go.Figure:
    """PSI de la variable elegida resaltado sobre las demás, con los umbrales del semáforo."""
    figura = go.Figure()
    otra_primera = True
    for nombre, grupo in datos.sort_values("orden").groupby("variable"):
        if nombre != variable:
            figura.add_scatter(x=grupo["ventana"], y=grupo["psi"], mode="lines", name="Otras variables",
                               line={"color": COLOR_SECUNDARIO, "width": 1}, opacity=0.5, showlegend=otra_primera,
                               legendgroup="otras", hovertemplate=f"{nombre}<br>%{{x}}<br>PSI %{{y:.3f}}<extra></extra>")
            otra_primera = False
    elegida = datos[datos["variable"] == variable].sort_values("orden")
    figura.add_scatter(x=elegida["ventana"], y=elegida["psi_ruido"], mode="lines", name="Ruido esperado",
                       line={"color": COLOR_CONTEXTO, "width": 1, "dash": "dot"},
                       hovertemplate="%{x}<br>Ruido esperado %{y:.3f}<extra></extra>")
    figura.add_scatter(x=elegida["ventana"], y=elegida["psi"], mode="lines+markers", name=variable,
                       line={"color": COLOR_REFERENCIA, "width": 2}, marker={"size": 8},
                       hovertemplate=f"{variable}<br>%{{x}}<br>PSI %{{y:.3f}}<extra></extra>")
    figura.add_hline(y=meta["psi_amarillo"], line_width=1, line_color=COLOR_ESTADO["amarillo"],
                     annotation_text=f"Amarillo {meta['psi_amarillo']:.2f}", annotation_position="top left")
    figura.add_hline(y=meta["psi_rojo"], line_width=1, line_color=COLOR_ESTADO["rojo"],
                     annotation_text=f"Rojo {meta['psi_rojo']:.2f}", annotation_position="top left")
    aplicar_estilo(figura, 400, titulo_y="PSI")
    figura.update_xaxes(type="category", categoryorder="array", categoryarray=elegida["ventana"].tolist())
    return figura


def grafico_rechazo(datos: pd.DataFrame, tasa_referencia: float) -> go.Figure:
    """Tasa de rechazo por ventana: muestra el efecto del drift en la decisión del modelo."""
    datos = datos.sort_values("orden")
    figura = go.Figure()
    figura.add_scatter(x=datos["ventana"], y=datos["tasa_rechazo"], mode="lines+markers", name="Tasa de rechazo",
                       line={"color": COLOR_ACTUAL, "width": 2}, marker={"size": 8},
                       hovertemplate="%{x}<br>%{y:.1%} rechazados<extra></extra>")
    figura.add_hline(y=tasa_referencia, line_width=1, line_color=COLOR_REFERENCIA,
                     annotation_text=f"Referencia {tasa_referencia:.1%}", annotation_position="top left")
    aplicar_estilo(figura, 320, titulo_y="% de créditos rechazados")
    figura.update_layout(showlegend=False)
    figura.update_yaxes(tickformat=".0%", rangemode="tozero")
    figura.update_xaxes(type="category", categoryorder="array", categoryarray=datos["ventana"].tolist())
    return figura


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def alto_tabla(tabla: pd.DataFrame) -> int:
    """Alto suficiente para ver todas las filas sin scroll interno, con un tope para las tablas largas."""
    return min(35 * (len(tabla) + 1) + 3, 600)


def mostrar_tabla_metricas(datos: pd.DataFrame) -> None:
    tabla = datos.sort_values("psi", ascending=False).assign(
        semaforo=lambda t: t["semaforo"].map(ETIQUETA_SEMAFORO),
        importancia=lambda t: t["importancia"] * 100,
        nulos_ref=lambda t: t["nulos_ref"] * 100,
        nulos_act=lambda t: t["nulos_act"] * 100,
    )[["variable", "semaforo", "importancia", "psi", "psi_ruido", "js", "ks", "ks_p", "chi2", "chi2_p", "nulos_ref",
       "nulos_act", "media_ref", "media_act"]]
    numero = st.column_config.NumberColumn
    st.dataframe(tabla, hide_index=True, width="stretch", height=alto_tabla(tabla), column_config={
        "variable": "Variable", "semaforo": "Semáforo",
        "importancia": numero("Importancia", format="%.1f%%", help="Peso de la variable en el XGBoost."),
        "psi": numero("PSI", format="%.3f"),
        "psi_ruido": numero("Ruido esperado", format="%.3f", help="PSI que se vería solo por azar con este volumen."),
        "js": numero("Jensen-Shannon", format="%.3f"),
        "ks": numero("KS", format="%.3f"), "ks_p": numero("p KS", format="%.4f"),
        "chi2": numero("Chi²", format="%.1f"), "chi2_p": numero("p Chi²", format="%.4f"),
        "nulos_ref": numero("Nulos ref.", format="%.1f%%"), "nulos_act": numero("Nulos actual", format="%.1f%%"),
        "media_ref": numero("Media ref.", format="%.3g"), "media_act": numero("Media actual", format="%.3g"),
    })


def mostrar_tabla_tendencias(datos: pd.DataFrame) -> None:
    tabla = datos.sort_values("psi_ultimo", ascending=False).assign(
        tendencia=lambda t: t["tendencia"].map(ICONO_TENDENCIA)
    )[["variable", "tendencia", "psi_ultimo", "psi_maximo", "ventanas_rojo", "salto_maximo", "ventana_salto"]]
    numero = st.column_config.NumberColumn
    st.dataframe(tabla, hide_index=True, width="stretch", height=alto_tabla(tabla), column_config={
        "variable": "Variable", "tendencia": "Tendencia", "psi_ultimo": numero("PSI último", format="%.3f"),
        "psi_maximo": numero("PSI máximo", format="%.3f"), "ventanas_rojo": "Ventanas en rojo",
        "salto_maximo": numero("Mayor salto", format="%.3f"), "ventana_salto": "Cambio brusco en",
    })


def mostrar_alerta(alerta) -> None:
    texto = f"**{alerta.mensaje}**\n\n{alerta.accion}"
    if alerta.nivel == "critica":
        st.error(texto, icon="🚨")
    elif alerta.nivel == "advertencia":
        st.warning(texto, icon="⚠️")
    else:
        st.info(texto, icon="ℹ️")


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Monitoreo del modelo de riesgo", page_icon="📊", layout="wide")
st.title("Monitoreo del modelo de riesgo crediticio")

if not ARCHIVO_META.exists():
    st.error("Todavía no hay resultados. Ejecuta primero `python model_monitoring.py` desde `mlops_pipeline/src`.")
    st.stop()

TABLAS = cargar_tablas(ARCHIVO_META.stat().st_mtime)
META = TABLAS["meta"]
referencia = META["referencia"]
st.caption(f"Referencia: {referencia['n']:,} créditos de entrenamiento ({referencia['desde']} a {referencia['hasta']}) · "
           f"AUC en test {META['auc_test']:.2f} · Última corrida: {META['generado'].replace('T', ' ')}")

# Filtros: una sola fila que afecta a todo el tablero
col_escenario, col_alcance, col_ventana, col_variable = st.columns(4)
escenario = ESCENARIOS[col_escenario.selectbox("Escenario", list(ESCENARIOS))]
solo_importantes = col_alcance.selectbox("Variables", ["Las 10 más importantes", "Todas"]) != "Todas"

metricas = TABLAS["metricas_drift"].query("escenario == @escenario")
resumen = TABLAS["resumen_ventanas"].query("escenario == @escenario").sort_values("orden")
distribuciones = TABLAS["distribuciones"].query("escenario == @escenario")
alertas = TABLAS["alertas"].query("escenario == @escenario")
tendencias = TABLAS["tendencias"].query("escenario == @escenario")

ventanas = resumen["ventana"].tolist()
ventana = col_ventana.selectbox("Ventana", ventanas, index=len(ventanas) - 1)
en_alcance = metricas[metricas["importante"]] if solo_importantes else metricas
actual = en_alcance[en_alcance["ventana"] == ventana]
fila = resumen[resumen["ventana"] == ventana].iloc[0]

variables = actual.sort_values("psi", ascending=False)["variable"].tolist()
variable = col_variable.selectbox("Variable en detalle", variables, help="Por defecto, la de mayor PSI en la ventana.")

if escenario == "simulacion":
    st.warning("**Datos simulados.** Estos lotes se alteraron a propósito para comprobar que el monitor detecta "
               "cambios; no describen a clientes reales. El Lote 1 no tiene perturbación: sus rojos vienen del drift "
               "real de los créditos de test.", icon="🧪")
if fila["n_muestra"] < 200:
    st.caption(f"Ventana chica ({fila['n_muestra']} créditos): las métricas tienen más ruido que en las ventanas grandes.")

# Indicadores de la ventana
proba = metricas[(metricas["ventana"] == ventana) & (metricas["variable"] == META["variable_proba"])].iloc[0]
diferencia_rechazo = (fila["tasa_rechazo"] - referencia["tasa_rechazo"]) * 100
tarjetas = st.columns(5)
tarjetas[0].metric("Estado", f"{ICONO_ESTADO[fila['estado']]} {fila['estado']}")
tarjetas[1].metric("Variables en rojo", f"{fila['rojos']} de {fila['rojos'] + fila['amarillos'] + fila['verdes']}")
tarjetas[2].metric("PSI de la probabilidad", f"{ETIQUETA_SEMAFORO[proba['semaforo']].split()[0]} {proba['psi']:.2f}")
tarjetas[3].metric("Tasa de rechazo", f"{fila['tasa_rechazo']:.1%}", f"{diferencia_rechazo:+.1f} pp vs. referencia",
                   delta_color="inverse")
tarjetas[4].metric("Créditos en la ventana", f"{fila['n_creditos']:,}", f"{fila['n_muestra']:,} en la muestra",
                   delta_color="off")

tab_estado, tab_evolucion, tab_recomendaciones = st.tabs(["Estado actual", "Evolución", "Recomendaciones"])

with tab_estado:
    col_barras, col_distribucion = st.columns(2)
    with col_barras:
        st.subheader("Riesgo por variable")
        st.plotly_chart(grafico_barras_riesgo(actual, META), width="stretch")
    with col_distribucion:
        st.subheader(f"Histórico vs. actual: {variable}")
        detalle = distribuciones[(distribuciones["ventana"] == ventana) & (distribuciones["variable"] == variable)]
        st.plotly_chart(grafico_distribucion(detalle), width="stretch")
    st.subheader("Métricas por variable")
    mostrar_tabla_metricas(actual)
    st.caption(f"El semáforo usa solo el PSI: verde menor a {META['psi_amarillo']:.2f}, amarillo hasta "
               f"{META['psi_rojo']:.2f} y rojo desde ese valor. Con muchos datos, KS y chi-cuadrado marcan como significativas diferencias mínimas.")

with tab_evolucion:
    st.subheader("Semáforo por variable y ventana")
    st.plotly_chart(grafico_mapa_calor(en_alcance), width="stretch")
    col_linea, col_rechazo = st.columns(2)
    with col_linea:
        st.subheader(f"Evolución del PSI: {variable}")
        st.plotly_chart(grafico_evolucion(en_alcance, variable, META), width="stretch")
    with col_rechazo:
        st.subheader("Tasa de rechazo del modelo")
        st.plotly_chart(grafico_rechazo(resumen, referencia["tasa_rechazo"]), width="stretch")
        st.caption("La referencia sale de validación cruzada, para que sea comparable con créditos que el modelo no vio.")
    st.subheader("Tendencias y cambios bruscos")
    mostrar_tabla_tendencias(tendencias[tendencias["importante"]] if solo_importantes else tendencias)

with tab_recomendaciones:
    st.subheader(f"Recomendación para {ventana}")
    alertas_ventana = alertas[alertas["ventana"] == ventana]
    criticas = alertas_ventana[alertas_ventana["nivel"] == "critica"]
    if fila["estado"] == "Reentrenar":
        nombres = ", ".join(f"`{v}`" for v in criticas["variable"].unique())
        st.error(f"**Evaluar reentrenar el modelo.** Siguen en rojo por segunda ventana o más: {nombres}. "
                 "Antes de reentrenar, confirmar que el cambio es real y no un problema en la captura del dato.", icon="🚨")
    elif fila["estado"] == "Revisar":
        st.warning("**Revisar las variables señaladas.** Hay variables importantes que entraron en rojo. "
                   "Si se repite en la próxima ventana, se sugerirá reentrenar.", icon="⚠️")
    elif fila["estado"] == "Observar":
        st.info("**Seguir observando.** Hay cambios moderados o en variables que pesan poco.", icon="ℹ️")
    else:
        st.success("**Sin alertas.** La población se mantiene parecida a la de entrenamiento.", icon="✅")

    prioritarias = alertas_ventana[alertas_ventana["nivel"] != "informativa"].sort_values(["nivel", "psi"],
                                                                                         ascending=False)
    for alerta in prioritarias.itertuples():   # "critica" va antes que "advertencia" en orden descendente
        mostrar_alerta(alerta)
    informativas = alertas_ventana[alertas_ventana["nivel"] == "informativa"]
    if len(informativas):
        with st.expander(f"Alertas informativas ({len(informativas)})"):
            for alerta in informativas.itertuples():
                mostrar_alerta(alerta)

    st.caption("El drift indica que los clientes cambiaron, no que el modelo haya empeorado: confirmarlo requiere "
               f"conocer si pagaron, y en test solo hay {META['no_paga_test']} malos pagadores. "
               f"Como el modelo ya es débil (AUC {META['auc_test']:.2f}), reentrenarlo podría no ser suficiente.")

    with st.expander(f"Historial de alertas del escenario ({len(alertas)})"):
        st.dataframe(alertas.sort_values(["orden", "nivel"])[["ventana", "nivel", "variable", "psi", "mensaje", "accion"]],
                     hide_index=True, width="stretch", column_config={
                         "ventana": "Ventana", "nivel": "Nivel", "variable": "Variable",
                         "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                         "mensaje": "Mensaje", "accion": "Acción sugerida",
                     })
