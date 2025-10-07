import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import unicodedata
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

MES_ABREV = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# ---------------------- Utils ----------------------
def normalizar(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return texto

def label_quincena_range(start_dt):
    end_dt = start_dt + timedelta(days=14)
    return f"{start_dt.day:02d}-{end_dt.day:02d} {MES_ABREV[start_dt.month-1]} {start_dt.year}"

# ---------------------- Métricas ----------------------
def custom_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"MA": np.nan, "MRCE": np.nan, "R2": np.nan}
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ma = np.mean(np.abs(y_true - y_pred))
    mrce = np.mean(np.abs(y_true - y_pred) / np.where(y_true == 0, 1, y_true))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    return {"MA": ma, "MRCE": mrce, "R2": r2}

# ---------------------- RF Prediction Helper ----------------------
def rf_predict_series(series, periods, max_lag=6):
    df_rf = pd.DataFrame({'y': series.values})
    for lag in range(1, max_lag+1):
        df_rf[f'lag_{lag}'] = df_rf['y'].shift(lag)
    df_rf = df_rf.dropna()
    if df_rf.empty:
        last_val = series.values[-1] if len(series.values) > 0 else 0.0
        return np.repeat(last_val, periods), np.full(len(series.values), np.nan)

    X_rf = df_rf[[f'lag_{lag}' for lag in range(1, max_lag+1)]].values
    y_rf = df_rf['y'].values

    # Normalización
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_rf_scaled = scaler_X.fit_transform(X_rf)
    y_rf_scaled = scaler_y.fit_transform(y_rf.reshape(-1,1)).ravel()

    rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
    rf_model.fit(X_rf_scaled, y_rf_scaled)

    preds_hist_scaled = rf_model.predict(X_rf_scaled)
    preds_hist = scaler_y.inverse_transform(preds_hist_scaled.reshape(-1,1)).ravel()

    preds_hist_full = np.full(len(series.values), np.nan)
    preds_hist_full[max_lag:] = preds_hist

    last_values = list(series.values[-max_lag:])
    if len(last_values) < max_lag:
        last_values = [series.mean()]*(max_lag - len(last_values)) + last_values

    future_preds = []
    for _ in range(periods):
        X_input = np.array(last_values[-max_lag:]).reshape(1,-1)
        X_input_scaled = scaler_X.transform(X_input)
        pred_scaled = rf_model.predict(X_input_scaled)[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        future_preds.append(pred)
        last_values.append(pred)

    return np.array(future_preds), preds_hist_full

# ---------------------- Gráfico + Tabla Quincenal ----------------------
def plot_quincenal(df, periods=4):
    date_col = "Fecha de Ocurrencia"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        st.warning("No hay fechas válidas en el dataset para quincenal.")
        return

    fecha_min = df[date_col].min()
    def quincena_inicio(dt: pd.Timestamp):
        delta = (dt - fecha_min).days
        bloque = delta // 15
        return fecha_min + timedelta(days=bloque*15)

    df["Quincena"] = df[date_col].apply(quincena_inicio)
    df["Clasificacion_norm"] = df["Clasificación"].apply(lambda x: normalizar(x))
    df["Servicio_norm"] = df["Servicio Ocurrencia"].apply(lambda x: normalizar(x))
    df["Evento_norm"] = df["Evento"].apply(lambda x: normalizar(x))

    linea_total = df.groupby("Quincena").size().sort_index()
    mask_bar = (
        (df["Servicio_norm"] == "pabellon quirurgico") &
        (df["Evento_norm"] == "registros clinicos incompletos / omision")
    )
    barras = df[mask_bar].groupby("Quincena").size().sort_index()

    all_q = sorted(list(set(linea_total.index).union(barras.index)))
    linea_total = linea_total.reindex(all_q, fill_value=0)
    barras = barras.reindex(all_q, fill_value=0)

    n = len(all_q)
    X = np.arange(n).reshape(-1,1)

    # Lineal
    if n >= 1:
        lin_total_model = LinearRegression().fit(X, linea_total.values)
        y_lin_hist = lin_total_model.predict(X)
        y_lin_future = lin_total_model.predict(np.arange(n, n+periods).reshape(-1,1))
        lin_bar_model = LinearRegression().fit(X, barras.values)
        y_bar_hist = lin_bar_model.predict(X)
        y_bar_future = lin_bar_model.predict(np.arange(n, n+periods).reshape(-1,1))
    else:
        y_lin_hist, y_lin_future, y_bar_hist, y_bar_future = np.array([]), [], np.array([]), []

    # RF
    y_rf_future_total, y_rf_hist_total = rf_predict_series(linea_total, periods)
    y_rf_future_bar, y_rf_hist_bar = rf_predict_series(barras, periods)

    y_rf_hist_total_safe = np.where(np.isnan(y_rf_hist_total), linea_total.values, y_rf_hist_total)
    y_rf_hist_bar_safe = np.where(np.isnan(y_rf_hist_bar), barras.values, y_rf_hist_bar)

    # Gráfico
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.bar(np.arange(len(all_q)), barras.values, width=0.4, alpha=0.7, color="steelblue", label="Registros incompletos Real")
    ax1.bar(np.arange(len(all_q), len(all_q)+periods), y_rf_future_bar, width=0.4, alpha=0.4, color="steelblue", label="Registros incompletos RF Futuro")
    ax1.set_ylabel("Eventos específicos", color="steelblue")

    ax2 = ax1.twinx()
    idx = np.arange(len(all_q)+periods)
    ax2.plot(np.arange(len(all_q)), linea_total.values, color="red", marker="o", linewidth=2, label="Total Real")
    ax2.plot(np.arange(len(all_q)), y_lin_hist, color="#1f77b4", linewidth=2, marker="o", label="Lineal Histórico")
    ax2.plot(np.arange(len(all_q)), y_rf_hist_total_safe, color="#2ca02c", linewidth=2, marker="s", label="RF Histórico")
    ax2.plot(np.arange(len(all_q), len(all_q)+periods), y_lin_future, linestyle="--", marker="o", color="#1f77b4", alpha=0.7, label="Lineal Futuro")
    ax2.plot(np.arange(len(all_q), len(all_q)+periods), y_rf_future_total, linestyle="--", marker="s", color="#2ca02c", alpha=0.7, label="RF Futuro")
    ax2.set_ylabel("Eventos Totales", color="red")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("📊 Gráfico Quincenal con Predicción Normalizada")
    fig.tight_layout()
    st.pyplot(fig)

    # Tabla
    tabla_df = pd.DataFrame({
        "Quincena": [label_quincena_range(d) for d in all_q],
        "Total Real": linea_total.values,
        "Lineal": np.round(y_lin_hist,2),
        "RF": np.round(y_rf_hist_total_safe,2),
        "Error Abs Lineal": np.round(np.abs(linea_total.values - y_lin_hist),2),
        "Error Abs RF": np.round(np.abs(linea_total.values - y_rf_hist_total_safe),2)
    })
    st.subheader("📑 Datos Quincenales con predicción")
    st.dataframe(tabla_df, use_container_width=True)

    # Métricas
    st.subheader("📊 Métricas Quincenales")
    met_tot_lin = custom_metrics(linea_total.values, y_lin_hist)
    met_tot_rf = custom_metrics(linea_total.values, y_rf_hist_total_safe)
    df_metrics = pd.DataFrame([
        {"Serie": "Totales - Lineal", **met_tot_lin},
        {"Serie": "Totales - RF", **met_tot_rf},
    ])
    st.dataframe(df_metrics, use_container_width=True)

# ---------------------- Semanal ----------------------
def generar_predicciones_semanal(df, periods=8, rf_max_lag=6):
    date_col = "Fecha de Ocurrencia"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df_adverso = df[df["Clasificación"].str.lower()=="adverso"].copy()
    if df_adverso.empty:
        st.warning("No hay registros de 'Adverso' para semanal.")
        return None, None

    df_adverso["__period_start"] = df_adverso[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    series = df_adverso.groupby("__period_start").size().sort_index()
    all_w = sorted(series.index)
    series = series.reindex(all_w, fill_value=0)

    n = len(all_w)
    X = np.arange(n).reshape(-1,1)
    y = series.values

    # Lineal
    lin_reg = LinearRegression().fit(X, y)
    y_lin_hist = lin_reg.predict(X)
    y_lin_future = lin_reg.predict(np.arange(n, n+periods).reshape(-1,1))

    # RF
    y_rf_future, y_rf_hist = rf_predict_series(series, periods, max_lag=rf_max_lag)
    y_rf_hist_safe = np.where(np.isnan(y_rf_hist), series.values, y_rf_hist)

    # Tabla
    future_dates = pd.date_range(start=all_w[-1], periods=periods+1, freq="W")[1:]
    all_dates = list(all_w) + list(future_dates)

    tabla_df = pd.DataFrame({
        "Semana inicio": [str(d.date()) for d in all_dates[:n]],
        "Real": series.values,
        "Lineal": np.round(y_lin_hist,2),
        "RF": np.round(y_rf_hist_safe,2)
    })
    st.subheader("📑 Datos Semanales")
    st.dataframe(tabla_df, use_container_width=True)

    # Métricas
    st.subheader("📊 Métricas Semanales")
    met_lin = custom_metrics(series.values, y_lin_hist)
    met_rf = custom_metrics(series.values, y_rf_hist_safe)
    df_metrics = pd.DataFrame([
        {"Serie": "Semanal - Lineal", **met_lin},
        {"Serie": "Semanal - RF", **met_rf},
    ])
    st.dataframe(df_metrics, use_container_width=True)

    # Gráfico
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(np.arange(len(all_w)), series.values, 'k-o', label="Real")
    ax.plot(np.arange(len(all_w)), y_lin_hist, 'r--', marker='o', label="Lineal Histórico")
    ax.plot(np.arange(len(all_w)), y_rf_hist_safe, 'g--', marker='s', label="RF Histórico")
    ax.plot(np.arange(len(all_w), len(all_w)+periods), y_lin_future, 'r:', marker='o', alpha=0.7, label="Lineal Futuro")
    ax.plot(np.arange(len(all_w), len(all_w)+periods), y_rf_future, 'g:', marker='s', alpha=0.7, label="RF Futuro")
    ax.legend()
    plt.title("📈 Predicciones Semanales Normalizadas")
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------- Main App ----------------------
def main():
    st.title("📊 Dashboard de Eventos Adversos (Quincenal + Semanal)")
    uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    if uploaded_file is None:
        st.info("👆 Esperando que subas un archivo Excel...")
        return

    df = pd.read_excel(uploaded_file)
    required_cols = ["Fecha de Ocurrencia", "Clasificación", "Servicio Ocurrencia", "Evento"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"⚠️ Falta columna '{col}' en el Excel.")
            return

    st.header("📈 Análisis Quincenal")
    plot_quincenal(df.copy(), periods=4)

    st.header("📈 Análisis Semanal (Adversos)")
    generar_predicciones_semanal(df.copy(), periods=8, rf_max_lag=6)

if __name__ == "__main__":
    main()
