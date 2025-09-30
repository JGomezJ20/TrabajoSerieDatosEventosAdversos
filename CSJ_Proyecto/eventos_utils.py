import os
import pandas as pd
import re
import streamlit as st

CARPETA_DATOS = "data_subida"

def obtener_archivo_actual():
    """
    Devuelve el nombre del archivo Excel seleccionado guardado en session_state.
    Si no hay ninguno o no existe, selecciona el más reciente y lo guarda.
    """
    if "archivo_seleccionado" in st.session_state:
        archivo = st.session_state["archivo_seleccionado"]
        ruta = os.path.join(CARPETA_DATOS, archivo)
        if os.path.exists(ruta):
            return archivo

    archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
    if not archivos:
        return None

    archivo_mas_reciente = sorted(
        archivos,
        key=lambda x: os.path.getmtime(os.path.join(CARPETA_DATOS, x)),
        reverse=True
    )[0]

    st.session_state["archivo_seleccionado"] = archivo_mas_reciente
    return archivo_mas_reciente


# Diccionario meses en español a número
meses_es = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}

# Detecta mes y año desde nombre de archivo
def extraer_mes_anio(nombre_archivo):
    match = re.search(r'(?i)(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)[\s_-]+(\d{4})', nombre_archivo)
    if match:
        mes = match.group(1).lower()
        anio = int(match.group(2))
        if mes in meses_es:
            return mes.capitalize(), anio
    return "Desconocido", None

# Carga DataFrame desde archivo Excel con columna extra de origen
def cargar_excel_con_origen(ruta, nombre_archivo):
    df = pd.read_excel(ruta)
    mes, año = extraer_mes_anio(nombre_archivo)
    df["Origen"] = nombre_archivo
    df["Mes"] = mes
    df["Año"] = año
    return df

# Devuelve conteo de eventos adversos en un DataFrame
def contar_eventos_adversos(df):
    if "Clasificación" in df.columns:
        return df[df["Clasificación"].str.lower().str.strip() == "adverso"].shape[0]
    return 0

# Devuelve cantidad de eventos adversos en un archivo específico, mes y año dados
def get_eventos_adversos_por_archivo_y_mes(carpeta, archivo, mes, anio):
    ruta = os.path.join(carpeta, archivo)
    try:
        df = pd.read_excel(ruta)
        if {'Mes', 'Año', 'Clasificación'}.issubset(df.columns):
            df_filtrado = df[
                (df['Mes'].str.lower() == mes.lower()) &
                (df['Año'] == anio) &
                (df['Clasificación'].str.lower().str.strip() == 'adverso')
            ]
            return len(df_filtrado)
        else:
            return 0
    except Exception as e:
        print(f"Error leyendo archivo {archivo}: {e}")
        return 0

# Obtiene diccionario {anio: {mes_num: archivo}} con archivos ordenados por fecha (más recientes primero)
def obtener_archivos_por_anio_y_mes(carpeta):
    archivos = sorted(
        [f for f in os.listdir(carpeta) if f.endswith(".xlsx")],
        key=lambda x: os.path.getmtime(os.path.join(carpeta, x)),
        reverse=True
    )

    archivos_por_anio_mes = {}
    for archivo in archivos:
        nombre = archivo.lower()
        match_mes = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", nombre)
        match_anio = re.search(r"(20\d{2})", nombre)

        if match_mes and match_anio:
            mes_nombre = match_mes.group(1)
            anio = int(match_anio.group(1))
            if mes_nombre in meses_es:
                mes_num = meses_es[mes_nombre]
                archivos_por_anio_mes.setdefault(anio, {})[mes_num] = archivo

    return archivos_por_anio_mes
