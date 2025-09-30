import streamlit as st 
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import warnings
import statistics as stats
import math
import hashlib

warnings.filterwarnings("ignore")

# ==================== Generador de números determinista ====================
_semilla = 1  # valor inicial por defecto

def inicializar_semilla_determinista(eventos_hist):
    """Inicializa la semilla basada en el contenido de los eventos históricos."""
    eventos_str = "_".join(map(str, eventos_hist))
    hash_bytes = hashlib.md5(eventos_str.encode()).digest()
    semilla = int.from_bytes(hash_bytes[:4], "big")  # 32 bits
    return semilla

def rand_uniform():
    """Generador de números uniformes entre 0 y 1 sin usar random."""
    global _semilla
    _semilla = (1664525 * _semilla + 1013904223) % 2**32
    return _semilla / 2**32

def rand_normal(mu=0, sigma=1):
    """Generador normal usando Box–Muller y rand_uniform."""
    u1 = rand_uniform()
    u2 = rand_uniform()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z0

# ==================== Funciones ====================
def predecir_eventos_suavizado(eventos_hist, meses_futuros):
    """Predicción basada en tendencia lineal + ruido controlado, determinista."""
    global _semilla
    _semilla = inicializar_semilla_determinista(eventos_hist)  # semilla determinista

    predicciones = []

    if len(eventos_hist) == 0:
        ultimo_valor = 100
        predicciones = [ultimo_valor]*meses_futuros
        return predicciones

    # Calcular sigma basado en desviación poblacional de las diferencias
    deltas = [eventos_hist[i] - eventos_hist[i-1] for i in range(1, len(eventos_hist))]
    sigma = stats.pstdev(deltas) if len(deltas) > 1 else max(5, stats.mean(eventos_hist)/10)
    sigma = max(sigma, 1)

    # Tendencia histórica con regresión lineal
    if len(eventos_hist) >= 2:
        x = list(range(len(eventos_hist)))
        slope, intercept = stats.linear_regression(x, eventos_hist)
    else:
        slope, intercept = 0, eventos_hist[-1]

    # Generar predicciones siguiendo la tendencia + ruido
    for i in range(len(eventos_hist), len(eventos_hist)+meses_futuros):
        valor_base = slope*i + intercept
        ruido = rand_normal(0, sigma)
        siguiente_valor = max(0, int(round(valor_base + ruido)))
        predicciones.append(siguiente_valor)

    return predicciones

def extraer_fecha_archivo(nombre_archivo):
    """Extrae (año, mes) desde el nombre del archivo."""
    meses_es = {
        "enero":1, "febrero":2, "marzo":3, "abril":4,
        "mayo":5, "junio":6, "julio":7, "agosto":8,
        "septiembre":9, "octubre":10, "noviembre":11, "diciembre":12
    }
    nombre_lower = nombre_archivo.lower()

    match_anio = re.search(r"(20\d{2})", nombre_lower)
    anio = int(match_anio.group(1)) if match_anio else 2025

    mes = None
    for nombre_mes, numero_mes in meses_es.items():
        if nombre_mes in nombre_lower:
            mes = numero_mes
            break
    if mes is None:
        mes = 1  # fallback

    return pd.Timestamp(year=anio, month=mes, day=1)

# ==================== Streamlit ====================
def main():
    st.set_page_config(page_title="📊 Predictor vs Real", layout="wide")
    st.title("📊 Comparación: Real vs Predicción de Eventos Adversos")

    CARPETA_DATOS = "data_subida"
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
    if not archivos:
        st.warning("⚠️ No hay archivos cargados aún.")
        st.stop()

    archivos_fechas = [(archivo, extraer_fecha_archivo(archivo)) for archivo in archivos]
    archivos_ordenados = sorted(archivos_fechas, key=lambda x: x[1])
    archivos = [a[0] for a in archivos_ordenados]

    clasificaciones_disponibles = set()
    for archivo in archivos:
        try:
            df_temp = pd.read_excel(os.path.join(CARPETA_DATOS, archivo))
            if "Clasificación" in df_temp.columns:
                clasificaciones_disponibles.update(df_temp["Clasificación"].dropna().unique().tolist())
        except:
            continue

    if not clasificaciones_disponibles:
        st.warning("⚠️ No se encontraron clasificaciones en los archivos.")
        st.stop()

    seleccion_clasificacion = st.multiselect(
        "Seleccione clasificaciones a incluir:",
        sorted(clasificaciones_disponibles),
        default=["Adverso"] if "Adverso" in clasificaciones_disponibles else list(clasificaciones_disponibles)
    )

    eventos_hist = []
    etiquetas_hist = []
    fechas_hist = []

    for archivo, fecha in archivos_ordenados:
        ruta = os.path.join(CARPETA_DATOS, archivo)
        try:
            df_temp = pd.read_excel(ruta)
            if "Clasificación" in df_temp.columns:
                cantidad = df_temp[df_temp["Clasificación"].isin(seleccion_clasificacion)].shape[0]
            else:
                cantidad = 0
        except:
            cantidad = 0

        eventos_hist.append(cantidad)
        etiquetas_hist.append(archivo)
        fechas_hist.append(fecha)

    eventos_hist = pd.Series(eventos_hist, index=fechas_hist)

    meses_futuros = st.slider("Número de meses a predecir:", 1, 60, 12)

    if st.button("🔮 Ejecutar predicción"):
        pred_final = predecir_eventos_suavizado(eventos_hist.tolist(), meses_futuros)

        fechas_futuras = pd.date_range(
            start=eventos_hist.index[-1] + pd.offsets.MonthBegin(),
            periods=meses_futuros, freq="MS"
        )

        df_predicciones = pd.DataFrame({
            "Periodo": fechas_futuras.strftime("%b-%Y"),
            "Predicción_eventos": pred_final
        })
        st.subheader("📋 Predicciones Generadas")
        st.dataframe(df_predicciones)

        fechas_todas = list(eventos_hist.index.strftime("%b-%Y")) + list(fechas_futuras.strftime("%b-%Y"))
        valores_reales = list(eventos_hist.values) + [None] * meses_futuros
        valores_predichos = [None] * len(eventos_hist) + pred_final

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(fechas_todas, valores_reales, marker='o', label="Real (Histórico)", color='blue')
        ax.plot(fechas_todas, valores_predichos, marker='x', linestyle='--', label="Predicción", color='red')

        ax.set_ylabel("Número de eventos")
        ax.set_xlabel("Mes/Año")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
