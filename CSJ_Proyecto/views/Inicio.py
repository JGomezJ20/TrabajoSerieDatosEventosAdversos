import streamlit as st
import os
import shutil
import re

def main():
    # Configuración general de la página
    st.set_page_config(page_title="Inicio - Selección de Archivo", layout="wide")
    st.title("🏠 Sistema de apoyo Gestión de Calidad")

    # Carpeta donde se almacenan los archivos subidos
    CARPETA_DATOS = "data_subida"
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    # Listado de archivos Excel existentes
    archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]

    # Diccionarios para conversión entre nombre y número de mes
    meses_es = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    meses_num_a_nombre = {v: k for k, v in meses_es.items()}

    # Organizar archivos por año y mes detectados en su nombre
    archivos_por_anio_mes = {}
    for archivo in archivos:
        nombre = archivo.lower()
        match_mes = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", nombre)
        match_anio = re.search(r"(20\d{2})", nombre)

        if match_mes and match_anio:
            mes_nombre = match_mes.group(1)
            anio = int(match_anio.group(1))
            mes_num = meses_es[mes_nombre]
            archivos_por_anio_mes.setdefault(anio, {})[mes_num] = archivo
        else:
            st.warning(f"⚠️ No se encontró mes o año en el nombre del archivo: {archivo}")

    # Si existen archivos válidos, mostrar selectboxes para Año y Mes
    if archivos_por_anio_mes:
        # Año seleccionado por defecto
        if "sel_anio" not in st.session_state:
            st.session_state["sel_anio"] = sorted(archivos_por_anio_mes.keys(), reverse=True)[0]

        anios = sorted(archivos_por_anio_mes.keys(), reverse=True)
        nuevo_anio = st.selectbox(
            "Selecciona el año",
            anios,
            index=anios.index(st.session_state["sel_anio"]),
            key="anio_selector"
        )

        if nuevo_anio != st.session_state["sel_anio"]:
            st.session_state["sel_anio"] = nuevo_anio
            st.session_state["pending_nav_change"] = "informes_1"
            st.rerun()

        # Meses disponibles según el año seleccionado
        meses_disponibles = sorted(archivos_por_anio_mes[st.session_state["sel_anio"]].keys())
        meses_nombres = [meses_num_a_nombre[m] for m in meses_disponibles]

        if "sel_mes_num" not in st.session_state or st.session_state["sel_mes_num"] not in meses_disponibles:
            st.session_state["sel_mes_num"] = meses_disponibles[0]

        nombre_mes_default = meses_num_a_nombre[st.session_state["sel_mes_num"]]

        nuevo_mes_nombre = st.selectbox(
            "Selecciona el mes",
            meses_nombres,
            index=meses_nombres.index(nombre_mes_default),
            key="mes_selector"
        )

        nuevo_mes_num = meses_es[nuevo_mes_nombre]
        if nuevo_mes_num != st.session_state["sel_mes_num"]:
            st.session_state["sel_mes_num"] = nuevo_mes_num
            st.session_state["pending_nav_change"] = "informes_1"
            st.rerun()

        # Archivo correspondiente al año y mes seleccionados
        archivo_seleccionado = archivos_por_anio_mes[st.session_state["sel_anio"]][st.session_state["sel_mes_num"]]

        # Guardar selección en session_state
        st.session_state["archivo_seleccionado"] = archivo_seleccionado
        st.session_state["anio_seleccionado"] = st.session_state["sel_anio"]
        st.session_state["mes_seleccionado"] = st.session_state["sel_mes_num"]

        st.write(f"Has seleccionado: Año {st.session_state['sel_anio']}, Mes {nuevo_mes_nombre.capitalize()}")

    else:
        st.info("ℹ️ No hay archivos válidos con año y mes reconocibles en la carpeta `data_subida`.")

    # Sección para subir un archivo Excel
    st.subheader("📁 Subir archivo Excel")
    archivo_subido = st.file_uploader("Selecciona un archivo Excel (.xlsx)", type=["xlsx"])

    if archivo_subido:
        ruta_guardado = os.path.join(CARPETA_DATOS, archivo_subido.name)
        with open(ruta_guardado, "wb") as f:
            shutil.copyfileobj(archivo_subido, f)

        st.success(f"✅ Archivo cargado: {archivo_subido.name}")

        # Detectar mes y año del archivo subido
        nombre = archivo_subido.name.lower()
        match_mes = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", nombre)
        match_anio = re.search(r"(20\d{2})", nombre)

        if match_mes and match_anio:
            mes_nombre = match_mes.group(1)
            anio = int(match_anio.group(1))
            mes_num = meses_es[mes_nombre]

            st.write(f"Archivo contiene datos de: **{mes_nombre.capitalize()} {anio}**")

            st.session_state["archivo_seleccionado"] = archivo_subido.name
            st.session_state["anio_seleccionado"] = anio
            st.session_state["mes_seleccionado"] = mes_num
            st.session_state["pending_nav_change"] = "informes_1"
            st.rerun()
        else:
            st.warning("⚠️ El archivo no contiene un mes o año reconocible en su nombre.")

if __name__ == "__main__":
    main()
