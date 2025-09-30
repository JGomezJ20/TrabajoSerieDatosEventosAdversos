import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import unicodedata
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

def bimestre_inicio(dt: pd.Timestamp):
    m = dt.month
    start_month = m if (m % 2 == 1) else m - 1
    return datetime(dt.year, start_month, 1)

def label_bimestre(dt: pd.Timestamp):
    m1 = dt.month
    m2 = m1 + 1
    return f"{MES_ABREV[m1-1]}-{MES_ABREV[m2-1]} {dt.year}"

# ---------------------- Gráfico Quincenal con Predicción ----------------------
def plot_quincenal(df, periods=4):
    date_col = "Fecha de Ocurrencia"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Crear quincenas de 15 días consecutivos desde el primer día
    fecha_min = df[date_col].min()
    def quincena_inicio(dt: pd.Timestamp):
        delta = (dt - fecha_min).days
        bloque = delta // 15
        return fecha_min + timedelta(days=bloque*15)

    df["Quincena"] = df[date_col].apply(quincena_inicio)

    # Normalizar columnas
    df["Clasificacion_norm"] = df["Clasificación"].apply(normalizar)
    df["Servicio_norm"] = df["Servicio Ocurrencia"].apply(normalizar)
    df["Evento_norm"] = df["Evento"].apply(normalizar)

    # Línea roja: todos los eventos
    linea_total = df.groupby("Quincena").size()

    # Barras azules: registros incompletos en pabellón
    mask_bar = (
        df["Servicio_norm"] == "pabellon quirurgico"
    ) & (
        df["Evento_norm"] == "registros clinicos incompletos / omision"
    )
    barras = df[mask_bar].groupby("Quincena").size()

    # Alinear periodos
    all_q = sorted(set(df["Quincena"]))
    linea_total = linea_total.reindex(all_q, fill_value=0)
    barras = barras.reindex(all_q, fill_value=0)

    # --- Predicciones Lineal ---
    n = len(all_q)
    X = np.arange(n).reshape(-1,1)
    lin_total_model = LinearRegression().fit(X, linea_total.values)
    y_lin_future = lin_total_model.predict(np.arange(n, n+periods).reshape(-1,1))
    lin_bar_model = LinearRegression().fit(X, barras.values)
    y_bar_future = lin_bar_model.predict(np.arange(n, n+periods).reshape(-1,1))

    # --- Predicciones RF con lags ---
    def rf_predict_series(series, periods, max_lag=6):
        df_rf = pd.DataFrame({'y': series.values})
        for lag in range(1, max_lag+1):
            df_rf[f'lag_{lag}'] = df_rf['y'].shift(lag)
        df_rf = df_rf.dropna()
        if df_rf.empty:
            return np.repeat(series.values[-1], periods)
        X_rf = df_rf[[f'lag_{lag}' for lag in range(1, max_lag+1)]].values
        y_rf = df_rf['y'].values
        rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
        rf_model.fit(X_rf, y_rf)
        last_values = list(series.values[-max_lag:])
        if len(last_values) < max_lag:
            last_values = [series.mean() for _ in range(max_lag - len(last_values))] + last_values
        future_preds = []
        for _ in range(periods):
            pred = rf_model.predict([last_values[-max_lag:]])[0]
            future_preds.append(pred)
            last_values.append(pred)
        return np.array(future_preds)

    y_rf_future_total = rf_predict_series(linea_total, periods)
    y_rf_future_bar = rf_predict_series(barras, periods)

    # Fechas futuras
    future_dates = [all_q[-1] + timedelta(days=15*(i+1)) for i in range(periods)]
    all_dates = list(all_q) + future_dates

    # --- Gráfico ---
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.bar(all_q, barras.values, width=12, alpha=0.7, color="steelblue",
            label="Registros clínicos incompletos (Pabellón) Real")
    ax1.bar(future_dates, y_rf_future_bar, width=12, alpha=0.4, color="steelblue",
            label="Registros incompletos RF Futuro")
    ax1.set_ylabel("Eventos específicos", color="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(all_q, linea_total.values, color="red", marker="o", linewidth=2,
             label="Total Real")
    ax2.plot(future_dates, y_lin_future, linestyle="--", marker="o", color="#1f77b4",
             alpha=0.7, label="Lineal Futuro")
    ax2.plot(future_dates, y_rf_future_total, linestyle="--", marker="s", color="#2ca02c",
             alpha=0.7, label="Total RF Futuro")
    ax2.set_ylabel("Eventos Totales", color="red")

    def label_quincena_range(start_dt):
        end_dt = start_dt + timedelta(days=14)
        return f"{start_dt.day:02d}-{end_dt.day:02d} {MES_ABREV[start_dt.month-1]} {start_dt.year}"

    ax1.set_xticks(all_dates)
    ax1.set_xticklabels([label_quincena_range(d) for d in all_dates], rotation=45, ha="right")

    plt.title("📊 Gráfico Quincenal con Predicción: Todos los eventos vs Registros incompletos (Pabellón)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)

    # --- Tabla con datos ---
    tabla_df = pd.DataFrame({
        "Quincena": [label_quincena_range(d) for d in all_dates],
        "Fecha inicio": all_dates,
        "Total eventos": list(linea_total.values)+[np.nan]*periods,
        "Total eventos Lineal": list(linea_total.values)+list(np.round(y_lin_future,1)),
        "Total eventos RF": list(linea_total.values)+list(np.round(y_rf_future_total,1)),
        "Registros incompletos (Pabellón)": list(barras.values)+[np.nan]*periods,
        "Registros incompletos Lineal": list(barras.values)+list(np.round(y_bar_future,1)),
        "Registros incompletos RF": list(barras.values)+list(np.round(y_rf_future_bar,1))
    })
    st.subheader("📑 Datos del gráfico quincenal con predicción")
    st.dataframe(tabla_df)

# ---------------------- Predicciones generales ----------------------
def generar_predicciones(df, periodo_type, title, periods=6, rf_max_lag=27,
                         rf_n_estimators=400, rf_max_depth=4, rf_min_samples_leaf=2):
    date_col = "Fecha de Ocurrencia"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if "Clasificación" not in df.columns:
        st.error("No se encontró la columna 'Clasificación' en el Excel.")
        return None, None
    df = df[df["Clasificación"].str.lower()=="adverso"]

    if df.empty:
        st.warning(f"No hay registros de 'Adverso' para {title}")
        return None, None

    if periodo_type == "2M":
        df["__period_start"] = df[date_col].apply(lambda dt: pd.Timestamp(bimestre_inicio(dt)))
        freq = "2MS"
    else:
        df["__period_start"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
        freq = "W-MON"

    df["__count"] = 1
    min_start = df["__period_start"].min()
    max_start = df["__period_start"].max()
    full_range = pd.date_range(start=min_start, end=max_start, freq=freq)

    series = df.groupby("__period_start")["__count"].sum().reindex(full_range, fill_value=0).sort_index()
    series_total = series.reset_index()
    series_total.columns = ["__period_start", "__count"]
    series_total["idx"] = np.arange(len(series_total))

    n = len(series_total)
    X = series_total[["idx"]].values
    y = series_total["__count"].values

    if n < 3:
        last_val = float(y[-1]) if n>=1 else 0.0
        future_lin = np.repeat(last_val, periods)
        future_rf = np.repeat(last_val, periods)
        y_pred_lin = y.copy()
        y_pred_rf_hist = y.copy()
    else:
        y_smooth = pd.Series(y).rolling(3, center=True, min_periods=1).median().values

        lin_reg = LinearRegression().fit(X, y_smooth)
        y_pred_lin = lin_reg.predict(X)
        future_idx = np.arange(len(X), len(X)+periods).reshape(-1,1)
        future_lin = lin_reg.predict(future_idx)

        # RF por lags
        max_lag = min(rf_max_lag, n)
        df_rf = series_total.copy()
        lag_cols = []
        for lag in range(1, max_lag+1):
            col = f'lag{lag}'
            df_rf[col] = df_rf['__count'].shift(lag)
            lag_cols.append(col)
        for h in range(1, periods+1):
            df_rf[f'target_h{h}'] = df_rf['__count'].shift(-h)

        rf_models = {}
        for h in range(1, periods+1):
            sub = df_rf.dropna(subset=lag_cols + [f'target_h{h}'])
            if len(sub) >= max(3, max_lag+1):
                X_train = sub[lag_cols].values
                y_train = sub[f'target_h{h}'].values
                rf = RandomForestRegressor(
                    random_state=42,
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_leaf=rf_min_samples_leaf,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                rf_models[h] = rf

        y_pred_rf_hist = np.full(n, np.nan)
        if 1 in rf_models:
            sub1 = df_rf.dropna(subset=lag_cols + ['target_h1'])
            if len(sub1) > 0:
                X_sub = sub1[lag_cols].values
                preds = rf_models[1].predict(X_sub)
                for idx_row, p in zip(sub1.index, preds):
                    target_idx = idx_row + 1
                    if 0 <= target_idx < n:
                        y_pred_rf_hist[target_idx] = p
        for i in range(n):
            if np.isnan(y_pred_rf_hist[i]):
                y_pred_rf_hist[i] = series_total['__count'].iloc[i]

        last_lags = [series_total['__count'].iloc[-l] for l in range(1, max_lag+1)]
        if len(last_lags) < max_lag:
            last_lags += [np.mean(last_lags)]*(max_lag - len(last_lags))

        future_rf = []
        for h in range(1, periods+1):
            if h in rf_models:
                future_rf.append(rf_models[h].predict([last_lags])[0])
            else:
                future_rf.append(float(np.mean(series_total['__count'].iloc[-max_lag:])))

        future_lin = np.maximum(future_lin, 0)
        future_rf = np.maximum(np.array(future_rf), 0)
        y_pred_rf_hist = np.maximum(y_pred_rf_hist, 0)

    future_dates = pd.date_range(start=series_total["__period_start"].iloc[-1], periods=periods+1, freq=freq)[1:]

    fig, ax = plt.subplots(figsize=(12,5))
    all_dates = series_total["__period_start"].tolist() + list(future_dates)
    lin_total = np.concatenate([y_pred_lin, future_lin])
    rf_total = np.concatenate([y_pred_rf_hist, future_rf])
    hist_len = len(y_pred_lin)

    ax.plot(series_total["__period_start"], series_total["__count"], marker="o", color="black", linewidth=2, label="Histórico Real")
    ax.plot(all_dates[:hist_len], lin_total[:hist_len], marker="o", color="#1f77b4", label="Lineal Histórico", linewidth=2)
    ax.plot(all_dates[hist_len:], lin_total[hist_len:], linestyle="--", marker="o", color="#1f77b4", alpha=0.7, label="Lineal Futuro", linewidth=2)
    ax.plot(all_dates[:hist_len], rf_total[:hist_len], marker="s", color="#2ca02c", label="RF Histórico (1-step)", linewidth=2)
    ax.plot(all_dates[hist_len:], rf_total[hist_len:], linestyle="--", marker="s", color="#2ca02c", alpha=0.7, label="RF Futuro", linewidth=2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Eventos Adversos")
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    if periodo_type == "2M":
        labels_hist = [label_bimestre(d) for d in series_total["__period_start"]]
        labels_future = [label_bimestre(d) for d in future_dates]
    else:
        labels_hist = [d.strftime("%Y-%m-%d") for d in series_total["__period_start"]]
        labels_future = [d.strftime("%Y-%m-%d") for d in future_dates]

    hist_df = pd.DataFrame({
        "Periodo inicio (fecha)": series_total["__period_start"],
        "Periodo": labels_hist,
        "Histórico Real": series_total["__count"],
        "Lineal": np.round(y_pred_lin,1),
        "Random Forest": np.round(y_pred_rf_hist,1)
    })
    future_df = pd.DataFrame({
        "Periodo inicio (fecha)": future_dates,
        "Periodo": labels_future,
        "Histórico Real": [np.nan]*len(future_dates),
        "Lineal": np.round(future_lin,1),
        "Random Forest": np.round(future_rf,1)
    })
    result_df = pd.concat([hist_df, future_df], ignore_index=True)
    st.subheader(f"Tabla: {title}")
    st.dataframe(result_df)
    return series_total, result_df

# ---------------------- Main app ----------------------
def main():
    st.title("📊 Dashboard de Eventos Adversos")
    uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx)", type=["xlsx"])
    if uploaded_file is None:
        st.info("👆 Esperando que subas un archivo Excel...")
        return
    df = pd.read_excel(uploaded_file)
    date_col = "Fecha de Ocurrencia"
    required_cols = [date_col, "Clasificación", "Servicio Ocurrencia", "Evento"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"⚠️ No se encontró la columna '{col}' en el Excel.")
            return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    st.subheader("📊 Gráfico Quincenal: Todos los eventos vs Registros incompletos (Pabellón)")
    plot_quincenal(df.copy(), periods=4)

    st.subheader("📈 Predicciones próximas 8 semanas (Adverso)")
    generar_predicciones(df.copy(), "W", "Predicciones Semanales - Adverso", periods=8, rf_max_lag=6)

if __name__ == "__main__":
    main()
