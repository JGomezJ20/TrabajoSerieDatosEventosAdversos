import streamlit as st 
import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import warnings
import unicodedata
from pandas.tseries.offsets import MonthBegin

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

def main():
    
    # ----------------------
    # Model imports
    # ----------------------
    try:
        from prophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        STATSMODELS_AVAILABLE = True
    except Exception:
        STATSMODELS_AVAILABLE = False

    # ----------------------
    # Helpers
    # ----------------------
    def normalize_col(c):
        c = str(c).strip().lower()
        c = unicodedata.normalize('NFD', c)
        c = ''.join([ch for ch in c if unicodedata.category(ch) != 'Mn'])
        return c

    def find_service_col(df_cols):
        df_norm = [normalize_col(c) for c in df_cols]
        for i, c in enumerate(df_norm):
            if 'servicio' in c and 'ocurrencia' in c:
                return df_cols[i]
        for i, c in enumerate(df_norm):
            if 'servicio' in c:
                return df_cols[i]
        return None

    def find_event_col(df_cols):
        df_norm = [normalize_col(c) for c in df_cols]
        for i, c in enumerate(df_norm):
            if c == 'evento':
                return df_cols[i]
        for i, c in enumerate(df_norm):
            if 'evento' in c:
                return df_cols[i]
        return None

    def month_from_filename(filename):
        meses_map = {
            "enero":1,"ene":1,"febrero":2,"feb":2,"marzo":3,"mar":3,
            "abril":4,"abr":4,"mayo":5,"may":5,"junio":6,"jun":6,
            "julio":7,"jul":7,"agosto":8,"ago":8,"septiembre":9,"setiembre":9,"sep":9,"set":9,
            "octubre":10,"oct":10,"noviembre":11,"nov":11,"diciembre":12,"dic":12
        }
        nombre = filename.lower()
        match_anio = re.search(r"(20\d{2})", nombre)
        anio = int(match_anio.group(1)) if match_anio else None
        mes = None
        for palabra,num in meses_map.items():
            if palabra in nombre:
                mes = num
                break
        if mes and anio:
            return pd.Timestamp(year=anio, month=mes, day=1)
        else:
            return None

    # ----------------------
    # Streamlit inputs
    # ----------------------
    st.title("Predictor de eventos adversos — lectura automática de Excel")

    DATA_FOLDER = st.text_input("Carpeta donde están los archivos Excel", value="./data_subida")
    PRED_MONTHS = st.number_input("Meses a predecir", min_value=1, max_value=24, value=6)
    SHOW_COMBINED = st.checkbox("Mostrar serie combinada (Servicio top + Evento top en mismo gráfico, eje secundario)", value=True)

    p = Path(DATA_FOLDER)
    files = sorted(p.glob("*.xls*"))

    st.write("Ruta buscada:", p.resolve())
    st.write("Archivos encontrados:", [f.name for f in files])

    if not files:
        st.warning(f"No encontré archivos .xls/.xlsx en {DATA_FOLDER}. Coloca los archivos y recarga.")
        st.stop()

    # ----------------------
    # Lectura y agregación
    # ----------------------
    all_rows = []
    file_summaries = []

    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception as e:
            st.error(f"No pude leer {f.name}: {e}")
            continue

        service_col = find_service_col(df.columns)
        event_col = find_event_col(df.columns)
        if service_col is None or event_col is None:
            st.warning(f"Archivo {f.name}: no pude detectar columnas Servicio/Event correctamente. Columnas: {df.columns.tolist()}")
            continue

        df_proc = df.copy()
        df_proc[service_col] = df_proc[service_col].astype(str).str.strip()
        df_proc[event_col] = df_proc[event_col].astype(str).str.strip()
        df_proc['__service'] = df_proc[service_col]
        df_proc['__event']   = df_proc[event_col]
        df_proc['__count'] = 1

        # Fecha original (si existe)
        date_col = None
        for c in df.columns:
            if 'fecha' in normalize_col(c):
                date_col = c
                break
        if date_col:
            df_proc['__date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        else:
            df_proc['__date'] = pd.NaT

        # Generar __month basado en nombre del archivo
        month_from_file = month_from_filename(f.name)
        if month_from_file is not None:
            df_proc['__month'] = month_from_file.to_period('M')
        else:
            df_proc['__month'] = df_proc['__date'].dt.to_period('M')

        # Agrupación
        agg = df_proc.groupby(['__month','__service','__event'])['__count'].sum().reset_index()
        agg['__file'] = f.name

        if not agg.empty:
            service_tot = agg.groupby('__service')['__count'].sum().sort_values(ascending=False)
            event_tot = agg.groupby('__event')['__count'].sum().sort_values(ascending=False)
            top_service = service_tot.idxmax() if not service_tot.empty else 'UNKNOWN'
            top_service_count = int(service_tot.max()) if not service_tot.empty else 0
            top_event = event_tot.idxmax() if not event_tot.empty else 'UNKNOWN'
            top_event_count = int(event_tot.max()) if not event_tot.empty else 0

            file_summaries.append({
                'file': f.name,
                'top_service': top_service,
                'top_service_count': top_service_count,
                'top_event': top_event,
                'top_event_count': top_event_count
            })

        all_rows.append(agg)

    if not all_rows:
        st.error("No se pudieron procesar archivos correctamente.")
        st.stop()

    df_all = pd.concat(all_rows, ignore_index=True)

    if df_all['__month'].isna().all():
        st.error("No se pudo inferir ninguna fecha/mes de los archivos.")
        st.stop()

    df_all['month_ts'] = df_all['__month'].dt.to_timestamp()

    # ----------------------
    # Top global
    # ----------------------
    global_service = df_all.groupby('__service')['__count'].sum().sort_values(ascending=False)
    global_event = df_all.groupby('__event')['__count'].sum().sort_values(ascending=False)

    top_service_global = global_service.idxmax()
    top_event_global = global_event.idxmax()

    st.subheader("Resumen por archivo (top servicio / top evento)")
    st.dataframe(pd.DataFrame(file_summaries))

    st.subheader("Top global")
    st.write(f"Servicio más frecuente: **{top_service_global}** — total eventos: {int(global_service.max())}")
    st.write(f"Evento más frecuente: **{top_event_global}** — total eventos: {int(global_event.max())}")

    # ----------------------
    # Serie temporal
    # ----------------------
    start_period = df_all['__month'].min()
    end_period = df_all['__month'].max()
    full_index = pd.period_range(start=start_period, end=end_period, freq='M').to_timestamp(how='start')

    service_series = df_all[df_all['__service']==top_service_global].groupby('month_ts')['__count'].sum().reindex(full_index, fill_value=0)
    event_series   = df_all[df_all['__event']==top_event_global].groupby('month_ts')['__count'].sum().reindex(full_index, fill_value=0)

    ts_df = pd.concat([service_series, event_series], axis=1)
    ts_df.columns = ['service_count','event_count']

    st.subheader("Serie temporal — meses")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts_df.index, ts_df['service_count'], marker='o', label=top_service_global)
    ax.plot(ts_df.index, ts_df['event_count'], marker='s', label=top_event_global)
    ax.set_title("Serie temporal de eventos y servicios")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    if SHOW_COMBINED:
        fig, ax1 = plt.subplots(figsize=(10,4))
        ax1.plot(ts_df.index, ts_df['service_count'], marker='o', label=top_service_global, color='blue')
        ax1.set_xlabel('Mes')
        ax1.set_ylabel(f"{top_service_global} — conteo")
        ax2 = ax1.twinx()
        ax2.plot(ts_df.index, ts_df['event_count'], marker='s', label=top_event_global, color='orange')
        ax2.set_ylabel(f"{top_event_global} — conteo")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        plt.title(f"{top_service_global} (izq) vs {top_event_global} (der)")
        plt.tight_layout()
        st.pyplot(fig)

    # ----------------------
    # Forecast function mejorada con año completo
    # ----------------------
    def forecast_series(series, periods=PRED_MONTHS, SMOOTHING=3):
        if series.empty: return None

        if isinstance(series.index, pd.PeriodIndex):
            series = series.copy()
            series.index = series.index.to_timestamp()

        df = series.reset_index()
        df.columns = ['ds','y']
        df['ds'] = pd.to_datetime(df['ds'])
        n_months = series.shape[0]

        # Suavizado adaptativo
        rolling_window = max(SMOOTHING, n_months//4)
        df['y'] = df['y'].rolling(rolling_window, min_periods=1).mean()

        # Limite máximo relativo al histórico
        max_limit = max(df['y'].max() * 1.5, series.max() * 1.5, 10)

        fc = None
        if PROPHET_AVAILABLE:
            try:
                yearly_seasonality = True if n_months >= 12 else False
                m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False, daily_seasonality=False, growth="linear")
                df['cap'] = max_limit
                df['floor'] = 0
                m.fit(df)
                future = m.make_future_dataframe(periods=periods, freq='M')
                future['cap'] = max_limit
                future['floor'] = 0
                fc = m.predict(future)[['ds','yhat']].set_index('ds')
                fc = fc.rename(columns={'yhat':'pred'})
            except:
                st.warning("Prophet falló, usando Holt-Winters.")

        if fc is None and STATSMODELS_AVAILABLE:
            try:
                sp = 12 if n_months>=12 else (3 if n_months>=6 else None)
                model = ExponentialSmoothing(series.values, trend="add", seasonal="add" if sp else None, seasonal_periods=sp, initialization_method="estimated")
                fit = model.fit()
                pred = fit.forecast(periods)
                start = series.index[-1]+MonthBegin(1)
                idx = pd.date_range(start=start, periods=periods, freq='MS')
                fc = pd.Series(pred,index=idx,name='pred').to_frame()
            except:
                st.error("No pude ajustar Holt-Winters")
                return None

        if fc is not None:
            # Mantener históricos intactos
            fc.loc[fc.index <= series.index[-1], 'pred'] = series.values
            # Limitar solo predicciones futuras
            fc.loc[fc.index > series.index[-1], 'pred'] = (
                fc.loc[fc.index > series.index[-1], 'pred']
                .clip(upper=max_limit, lower=0)
                .round()
                .astype(int)
            )

        return fc

    # ----------------------
    # Forecast
    # ----------------------
    service_fc = forecast_series(ts_df['service_count'])
    event_fc   = forecast_series(ts_df['event_count'])

    # ----------------------
    # Evaluación (RMSE compatible)
    # ----------------------
    for fc, name, hist in [(service_fc, top_service_global, ts_df['service_count']), (event_fc, top_event_global, ts_df['event_count'])]:
        if fc is not None:
            st.write(f"Predicción (Holt/Prophet) para {name}")
            st.dataframe(fc.tail(PRED_MONTHS))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(hist.index, hist, marker='o', color='blue', label="Histórico")
            ax.plot(fc.index, fc['pred'], marker='o', color='red', linestyle='--', label="Predicción")
            ax.axvline(hist.index[-1], linestyle='--', color='gray', alpha=0.6)
            ax.set_title(f"Forecast {name} (Holt/Prophet)")
            ax.set_ylabel("Número de eventos")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # RMSE
            y_true = hist.values
            y_pred = fc.loc[hist.index, 'pred'].values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            st.write(f"RMSE para {name}: {rmse:.2f}")

    # ----------------------
    # Resumen mensual y forecast ejecutivo
    # ----------------------
    monthly_totals = df_all.groupby('__month')['__count'].sum().rename("total_events")
    monthly_top_service = df_all[df_all['__service']==top_service_global].groupby('__month')['__count'].sum().rename("top_service_events")
    monthly_df = pd.concat([monthly_totals, monthly_top_service], axis=1).fillna(0)
    monthly_df['otros'] = monthly_df['total_events']-monthly_df['top_service_events']

    if isinstance(monthly_df.index, pd.PeriodIndex):
        monthly_df.index = monthly_df.index.to_timestamp()

    fc_total = forecast_series(monthly_df['total_events'], periods=PRED_MONTHS)
    fc_top   = forecast_series(monthly_df['top_service_events'], periods=PRED_MONTHS)

    if fc_total is not None:
        preds = fc_total.copy()
        preds = preds.rename(columns={'pred':'pred_total'})
        preds['pred_top'] = fc_top['pred'] if fc_top is not None else 0
        preds['pred_otros'] = preds['pred_total'] - preds['pred_top']
        preds = preds.fillna(0)

        preds_display = preds.reset_index()
        preds_display['Mes'] = pd.to_datetime(preds_display.iloc[:,0]).dt.to_period('M').dt.to_timestamp().dt.strftime('%Y-%m')

        st.subheader("Predicción ejecutiva — próximos meses")
        st.dataframe(preds_display[['Mes','pred_total','pred_top','pred_otros']].style.format({
            'pred_total':'{:.0f}', 'pred_top':'{:.0f}', 'pred_otros':'{:.0f}'
        }))

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(monthly_df.index, monthly_df['top_service_events'], color='blue', marker='o', label=f"{top_service_global} (hist)")
        ax.plot(monthly_df.index, monthly_df['otros'], color='gray', marker='o', label="Otros servicios (hist)")
        ax.plot(monthly_df.index, monthly_df['total_events'], color='black', linestyle='-', alpha=0.5, label="Total (hist)")

        ax.plot(preds.index, preds['pred_top'], color='blue', linestyle='--', marker='x', label=f"{top_service_global} (pred)")
        ax.plot(preds.index, preds['pred_otros'], color='gray', linestyle='--', marker='x', label="Otros servicios (pred)")
        ax.plot(preds.index, preds['pred_total'], color='black', linestyle='--', alpha=0.7, label="Total (pred)")

        ax.axvspan(preds.index[0], preds.index[-1], color='orange', alpha=0.1)
        ax.axvline(monthly_df.index[-1]+MonthBegin(1), color='red', linestyle='--', linewidth=1)

        ax.set_ylabel("Número de eventos")
        ax.set_title("Predicción ejecutiva mensual — histórico vs forecast")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
