import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from eventos_utils import meses_es

# Función para verificar columnas
def columnas_existen(df, columnas):
    return set(columnas).issubset(df.columns)

# Función para obtener meses de un trimestre o semestre
def obtener_meses_periodo(inicio, meses_por_periodo):
    return [inicio + i for i in range(meses_por_periodo) if inicio + i <= 12]

def main():
    st.set_page_config(page_title="Gráficos", layout="wide")
    st.title("📈 Visualización de Datos Clínicos")

    CARPETA_DATOS = "data_subida"
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    # Listar archivos Excel
    archivos = sorted(
        [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")],
        key=lambda x: os.path.getmtime(os.path.join(CARPETA_DATOS, x)),
        reverse=True
    )

    archivos_por_anio_mes = {}
    for archivo in archivos:
        nombre = archivo.lower().replace("_", " ").replace("-", " ")
        match_mes = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", nombre)
        match_anio = re.search(r"(20\d{2})", nombre)
        if match_mes and match_anio:
            mes_nombre = match_mes.group(1)
            anio = int(match_anio.group(1))
            mes_num = meses_es[mes_nombre]
            archivos_por_anio_mes.setdefault(anio, {})[mes_num] = archivo

    if not archivos_por_anio_mes:
        st.warning("⚠️ No hay archivos disponibles con formato reconocido.")
        st.stop()

    # Sidebar: año
    anios_disponibles = sorted(archivos_por_anio_mes.keys(), reverse=True)
    if "anio_seleccionado" not in st.session_state or st.session_state["anio_seleccionado"] not in anios_disponibles:
        st.session_state["anio_seleccionado"] = anios_disponibles[0]
    anio_seleccionado = st.sidebar.selectbox("Selecciona el año:", anios_disponibles,
                                              index=anios_disponibles.index(st.session_state["anio_seleccionado"]))
    st.session_state["anio_seleccionado"] = anio_seleccionado

    # Sidebar: mes
    meses_disponibles = sorted(archivos_por_anio_mes[anio_seleccionado].keys())
    meses_nombres = [k for k, v in meses_es.items() if v in meses_disponibles]

    if "mes_seleccionado" not in st.session_state or st.session_state["mes_seleccionado"] not in meses_disponibles:
        st.session_state["mes_seleccionado"] = meses_disponibles[0]

    nombre_mes_default = next((k for k, v in meses_es.items() if v == st.session_state["mes_seleccionado"]), meses_nombres[0])
    default_index = meses_nombres.index(nombre_mes_default) if nombre_mes_default in meses_nombres else 0

    mes_seleccionado_nombre = st.sidebar.selectbox("Selecciona el mes:", meses_nombres, index=default_index)
    mes_seleccionado = meses_es[mes_seleccionado_nombre]
    st.session_state["mes_seleccionado"] = mes_seleccionado

    archivo_actual = archivos_por_anio_mes[anio_seleccionado][mes_seleccionado]
    st.session_state["archivo_seleccionado"] = archivo_actual

    # Cargar archivo
    ruta_archivo = os.path.join(CARPETA_DATOS, archivo_actual)
    try:
        df = pd.read_excel(ruta_archivo)
        st.success(f"📊 Mostrando gráficos para: **{archivo_actual}**")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        st.stop()

    # Filtro por Clasificación
    if "Clasificación" in df.columns:
        clasificaciones = df["Clasificación"].dropna().unique().tolist()
        seleccion_clasificacion = st.sidebar.multiselect("Filtrar por Clasificación:", clasificaciones, default=clasificaciones)
        df = df[df["Clasificación"].isin(seleccion_clasificacion)]
    else:
        seleccion_clasificacion = []

    # Primer gráfico (mantener igual)
    if columnas_existen(df, ["Clasificación", "Evento"]):
        st.subheader("📊 Distribución de Clasificación y Conteo de Eventos")
        conteo_clasif = df["Clasificación"].value_counts()
        total_clasif = conteo_clasif.sum()
        porcentajes = (conteo_clasif / total_clasif) * 100
        conteo_eventos = df["Evento"].value_counts().head(10)

        st.markdown("""
        <style>
        .titulo-seccion {
            font-size: 18px;
            font-weight: bold;
            color: #003366;
            margin-bottom: 5px;
        }
        .alineado-derecha {
            margin-left: 600px;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown('<div class="titulo-seccion">Distribución % Clasificación</div>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(1.2, 3))
            bottom = 0
            colors = sns.color_palette("Set2", len(porcentajes))
            for i, (clasificacion, porcentaje) in enumerate(porcentajes.items()):
                ax1.bar(0, porcentaje, bottom=bottom, width=0.3, color=colors[i],
                        label=f"{clasificacion} ({porcentaje:.1f}%)")
                bottom += porcentaje
            ax1.set_ylim(0, 100)
            ax1.set_xlim(-0.4, 0.4)
            ax1.set_xticks([])
            ax1.set_ylabel("%", fontsize=8)
            ax1.set_title("Clasificación", fontsize=9)
            ax1.tick_params(axis='y', labelsize=7)
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="x-small", frameon=False)
            st.pyplot(fig1)

        with col2:
            st.markdown('<div class="titulo-seccion alineado-derecha">Top 10 Eventos</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            barras = sns.barplot(
                x=conteo_eventos.values,
                y=conteo_eventos.index,
                ax=ax2,
                palette="Blues_d"
            )
            ax2.set_xlabel("Cantidad", fontsize=9)
            ax2.set_ylabel("Evento", fontsize=9)
            ax2.tick_params(labelsize=8)
            for i, v in enumerate(conteo_eventos.values):
                ax2.text(v + 0.5, i, str(v), color='black', va='center', fontsize=8)
            st.pyplot(fig2)

    # Heatmap con selector jerárquico y meses disponibles
    if columnas_existen(df, ["Evento", "Servicio Ocurrencia", "Clasificación"]):
        st.subheader("🧊 Mapa de Calor: Evento vs Servicio")

        tipo_periodo = st.selectbox("Selecciona tipo de periodo:", ["Mes", "Trimestre", "Semestre"])

        meses_seleccion_periodo = []
        if tipo_periodo == "Mes":
            meses_seleccion_periodo = [mes_seleccionado]
        else:
            # Construir meses según tipo de periodo
            if tipo_periodo == "Trimestre":
                opcion = st.selectbox("Selecciona trimestre:",
                                      ["Primer trimestre","Segundo trimestre","Tercer trimestre","Cuarto trimestre"])
                inicio = {"Primer trimestre":1, "Segundo trimestre":4, "Tercer trimestre":7, "Cuarto trimestre":10}[opcion]
                meses_seleccion_periodo = [m for m in obtener_meses_periodo(inicio,3) if m in archivos_por_anio_mes[anio_seleccionado]]
            else:  # Semestre
                opcion = st.selectbox("Selecciona semestre:", ["Primer semestre", "Segundo semestre"])
                inicio = {"Primer semestre":1, "Segundo semestre":7}[opcion]
                meses_seleccion_periodo = [m for m in obtener_meses_periodo(inicio,6) if m in archivos_por_anio_mes[anio_seleccionado]]

        if not meses_seleccion_periodo:
            st.info(f"ℹ️ No hay datos disponibles para el {tipo_periodo.lower()} seleccionado.")
        else:
            archivos_periodo = [os.path.join(CARPETA_DATOS, archivos_por_anio_mes[anio_seleccionado][m])
                                for m in meses_seleccion_periodo]
            df_heatmap = pd.concat([pd.read_excel(a) for a in archivos_periodo], ignore_index=True)

            # 🔹 Filtrar usando la misma selección del sidebar
            df_filtrado = df_heatmap[df_heatmap["Clasificación"].isin(seleccion_clasificacion)]

            if df_filtrado.empty:
                st.info("ℹ️ No hay eventos para las clasificaciones seleccionadas en este periodo.")
            else:
                heatmap_data = pd.pivot_table(
                    df_filtrado,
                    index="Evento",
                    columns="Servicio Ocurrencia",
                    aggfunc="size",
                    fill_value=0
                )
                heatmap_data = heatmap_data.loc[:, (heatmap_data != 0).any(axis=0)]
                heatmap_data = heatmap_data[(heatmap_data.T != 0).any()]

                if not heatmap_data.empty:
                    total_eventos = heatmap_data.values.sum()
                    st.markdown(f"### 🔢 Total de eventos en Heatmap: **{total_eventos}**")
                    fig, ax = plt.subplots(figsize=(min(12, 0.7*len(heatmap_data.columns)),
                                                    min(10,0.3*len(heatmap_data))))
                    sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d",
                                linewidths=0.5, linecolor='lightgray', cbar=True, ax=ax)
                    ax.set_xlabel("Servicio")
                    ax.set_ylabel("Evento")
                    ax.set_title(f"Mapa de Calor: Clasificaciones seleccionadas ({tipo_periodo})")
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ No hay datos suficientes para generar el mapa de calor.")

if __name__ == "__main__":
    main()
