import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from eventos_utils import cargar_excel_con_origen, extraer_mes_anio, contar_eventos_adversos

def main():
    st.set_page_config(page_title="Resumen Eventos", layout="wide")
    st.title("📅 Resumen de Eventos por Archivo")

    CARPETA_DATOS = "data_subida"
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    # Listar archivos xlsx ordenados por fecha modificación (descendente)
    archivos = sorted(
        [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")],
        key=lambda x: os.path.getmtime(os.path.join(CARPETA_DATOS, x)),
        reverse=True
    )

    if not archivos:
        st.warning("⚠️ No hay archivos cargados aún.")
        st.stop()

    # Agrupar archivos por año extraído del nombre (buscando un número de 4 dígitos)
    archivos_por_anio = defaultdict(list)
    for archivo in archivos:
        for palabra in archivo.split():
            palabra_limpia = palabra.replace(".xlsx", "").strip()
            if palabra_limpia.isdigit() and len(palabra_limpia) == 4:
                archivos_por_anio[int(palabra_limpia)].append(archivo)
                break

    anios_disponibles = sorted(archivos_por_anio.keys(), reverse=True)
    anio_seleccionado = st.selectbox("Seleccione el año para ver resumen:", anios_disponibles, index=0)

    # ======================= #
    # Multiselect de Clasificación
    # ======================= #
    clasificaciones_disponibles = set()
    for archivo in archivos_por_anio[anio_seleccionado]:
        try:
            df_temp = pd.read_excel(os.path.join(CARPETA_DATOS, archivo))
            if "Clasificación" in df_temp.columns:
                clasificaciones_disponibles.update(df_temp["Clasificación"].dropna().unique().tolist())
        except Exception:
            continue

    if not clasificaciones_disponibles:
        st.warning("⚠️ No se encontraron clasificaciones en los archivos.")
        st.stop()

    seleccion_clasificacion = st.multiselect(
        "Seleccione clasificaciones a incluir:",
        sorted(clasificaciones_disponibles),
        default=["Adverso"] if "Adverso" in clasificaciones_disponibles else list(clasificaciones_disponibles)
    )

    # Crear resumen para el año seleccionado
    resumen = []
    for archivo in archivos_por_anio[anio_seleccionado]:
        ruta = os.path.join(CARPETA_DATOS, archivo)
        try:
            df_temp = pd.read_excel(ruta)
            if "Clasificación" in df_temp.columns:
                cantidad = df_temp[df_temp["Clasificación"].isin(seleccion_clasificacion)].shape[0]
            else:
                cantidad = None  # Indicamos error por falta de columna
        except Exception:
            cantidad = None  # Indicamos error en lectura

        resumen.append({
            "Archivo": archivo,
            "Eventos Seleccionados": cantidad
        })

    df_resumen_anio = pd.DataFrame(resumen)

    if df_resumen_anio.empty:
        st.warning(f"⚠️ No se encontraron archivos válidos para el año {anio_seleccionado}.")
    else:
        st.dataframe(df_resumen_anio)

        if "Eventos Seleccionados" in df_resumen_anio.columns and df_resumen_anio["Eventos Seleccionados"].notnull().all():
            total = df_resumen_anio["Eventos Seleccionados"].sum()
            st.markdown(f"### 🔢 Total de eventos en {anio_seleccionado} ({', '.join(seleccion_clasificacion)}): **{total}**")
        else:
            st.info("ℹ️ Algunos archivos no contienen datos válidos para las clasificaciones seleccionadas.")

    # ======================= #
    # Total de eventos seleccionados por mes para el año
    # ======================= #

    meses_es = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12
    }

    conteo_mensual = {mes: 0 for mes in range(1, 13)}

    for archivo in archivos_por_anio[anio_seleccionado]:
        nombre_archivo = archivo.lower()
        mes_detectado = None
        for nombre_mes, numero_mes in meses_es.items():
            if nombre_mes in nombre_archivo:
                mes_detectado = numero_mes
                break

        if mes_detectado is not None:
            try:
                df_temp = pd.read_excel(os.path.join(CARPETA_DATOS, archivo))
                if "Clasificación" in df_temp.columns:
                    cantidad = df_temp[df_temp["Clasificación"].isin(seleccion_clasificacion)].shape[0]
                    conteo_mensual[mes_detectado] += cantidad
            except Exception as e:
                st.warning(f"❌ Error leyendo {archivo}: {e}")

    meses_espanol = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    df_mensual = pd.DataFrame({
        "Mes": meses_espanol,
        "Cantidad": [conteo_mensual[m] for m in range(1, 13)]
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_mensual["Mes"], df_mensual["Cantidad"], marker='o', linestyle='-', color="#c50707")
    ax.set_title(f"Eventos por Mes - {anio_seleccionado} ({', '.join(seleccion_clasificacion)})")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad de Eventos")

    # Añadir etiquetas numéricas en los puntos
    for i, (mes, valor) in enumerate(zip(df_mensual["Mes"], df_mensual["Cantidad"])):
        ax.text(i, valor + 2, str(valor), ha="center", fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
