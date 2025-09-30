import os
import re
import pickle
import requests
import pandas as pd
import json
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

CARPETA_DATOS = "data_subida"
INDEX_PATH = "faiss_index.pkl"
RUTA_MEMORIA = "memoria_respuestas.json"

# ============================
# Utilidades de memoria Mistral
# ============================
def _asegurar_archivo_memoria():
    if not os.path.exists(RUTA_MEMORIA):
        with open(RUTA_MEMORIA, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def cargar_memoria(limit=None):
    _asegurar_archivo_memoria()
    with open(RUTA_MEMORIA, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        return data[-limit:]
    return data

def guardar_respuesta_en_memoria(pregunta, respuesta, usuario="default", meta=None):
    _asegurar_archivo_memoria()
    entrada = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "usuario": usuario,
        "pregunta": pregunta,
        "respuesta": respuesta,
        "meta": meta or {}
    }
    with open(RUTA_MEMORIA, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(entrada)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

# ============================
# Control de tokens y resumen automático
# ============================
UMBRAL_TOKENS_PROMPT = 30000
RESERVA_TOKENS_RESPUESTA = 8000

def approx_tokens(texto):
    return max(1, int(len(texto)/4))

def construir_contexto_memoria_relevante(pregunta, n_recientes=5, api_key=None):
    memoria = cargar_memoria()
    recientes = memoria[-n_recientes:] if len(memoria) >= n_recientes else memoria[:]
    contexto_recientes = "\n\n".join([f"Pregunta: {m['pregunta']}\nRespuesta: {m['respuesta']}" for m in recientes])

    full_context = "\n\n".join([f"Pregunta: {m['pregunta']}\nRespuesta: {m['respuesta']}" for m in memoria])
    tokens_est = approx_tokens(full_context)

    if tokens_est + RESERVA_TOKENS_RESPUESTA < UMBRAL_TOKENS_PROMPT:
        return full_context, False
    else:
        antiguas = memoria[:-n_recientes] if len(memoria) > n_recientes else []
        resumen_antiguas = ""
        if antiguas and api_key:
            prompt_resumen = (
                "Resume brevemente (máx 200 palabras) las siguientes entradas de memoria. "
                "Mantén bullets con datos relevantes (periodo, hallazgos, acciones propuestas):\n\n"
                + "\n\n".join([f"Pregunta: {m['pregunta']}\nRespuesta: {m['respuesta']}" for m in antiguas])
            )
            resumen_antiguas = consultar_mistral(prompt_resumen, api_key)
        contexto_final = "Resumen de memoria antigua:\n" + resumen_antiguas + "\n\nEntradas recientes:\n" + contexto_recientes
        return contexto_final, True

# ============================
# Cargar datos desde archivos Excel
# ============================
def cargar_datos():
    print("🔹 Cargando datos desde Excel...")
    archivos = [f for f in os.listdir(CARPETA_DATOS) if f.endswith(".xlsx")]
    dataframes = []

    for archivo in archivos:
        ruta = os.path.join(CARPETA_DATOS, archivo)
        try:
            df = pd.read_excel(ruta)
            if df.empty:
                print(f"⚠️ Archivo vacío: {archivo}")
                continue
            mes, anio = extraer_mes_anio(archivo)
            if mes is None or anio is None:
                print(f"⚠️ No se pudo extraer mes/año de: {archivo}")
                continue
            df["Origen"] = archivo
            df["Mes"] = mes
            df["Año"] = anio
            dataframes.append(df)
            print(f"✅ Cargado: {archivo} ({len(df)} filas)")
        except Exception as e:
            print(f"❌ Error al leer {archivo}: {e}")

    if not dataframes:
        print("⚠️ No se encontraron datos válidos.")
        return pd.DataFrame()

    df_completo = pd.concat(dataframes, ignore_index=True)
    meses_orden = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    df_completo["Mes_Num"] = df_completo["Mes"].map(meses_orden)
    df_completo.sort_values(by=["Año", "Mes_Num"], inplace=True)
    print(f"🔹 Total de registros cargados: {len(df_completo)}")
    return df_completo

def extraer_mes_anio(nombre_archivo):
    match = re.search(r'(?i)(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(\d{4})', nombre_archivo)
    if match:
        return match.group(1).capitalize(), int(match.group(2))
    return None, None

# ============================
# Crear documentos para LangChain
# ============================
def crear_documentos(df):
    print("🔹 Creando documentos para LangChain...")
    return [Document(page_content=fila.to_string(), metadata={"fila": i}) for i, fila in df.iterrows()]

# ============================
# Cargar o crear índice FAISS
# ============================
def obtener_vectorstore(docs_divididos):
    if os.path.exists(INDEX_PATH):
        print("🔹 Cargando índice FAISS existente...")
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print("🔹 Creando nuevo índice FAISS...")
        embeddings = FakeEmbeddings(size=256)
        vectorstore = FAISS.from_documents(docs_divididos, embeddings)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(vectorstore, f)
        print("✅ Índice FAISS creado y guardado.")
        return vectorstore

# ============================
# Consultar modelo Mistral
# ============================
def consultar_mistral(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 20000
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            resultado = response.json()
            print("✅ Tokens usados:", resultado.get("usage", {}))
            return resultado["choices"][0]["message"]["content"]
        return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Error en conexión: {e}"

# ============================
# Detectar periodo en la pregunta
# ============================
def detectar_periodo(pregunta):
    pregunta = pregunta.lower()
    meses = {
        "enero":1, "febrero":2, "marzo":3, "abril":4, "mayo":5, "junio":6,
        "julio":7, "agosto":8, "septiembre":9, "octubre":10, "noviembre":11, "diciembre":12
    }

    mes = next((meses[m] for m in meses if m in pregunta), None)
    semestre = 1 if "primer semestre" in pregunta else 2 if "segundo semestre" in pregunta else None
    trimestre_map = {"primer trimestre":1,"segundo trimestre":2,"tercer trimestre":3,"cuarto trimestre":4}
    trimestre = next((v for k,v in trimestre_map.items() if k in pregunta), None)

    return {"mes": mes, "semestre": semestre, "trimestre": trimestre}

# ============================
# Extraer elementos clave desde la pregunta
# ============================
def extraer_elementos_pregunta(pregunta):
    pregunta = pregunta.lower()
    print(f"🔹 Analizando pregunta: {pregunta}")

    clasificaciones = {
        "adverso": "Adverso", "adversos": "Adverso",
        "incidente": "Incidente", "incidentes": "Incidente",
        "centinela": "Centinela", "centinelas": "Centinela",
        "descarte": "Descarte", "descartes": "Descarte",
        "leve": "Leve", "leves": "Leve",
        "grave": "Grave", "graves": "Grave",
        "moderado": "Moderado", "moderados": "Moderado",
        "general": "General"
    }

    servicios = {
        "urgencia": "Urgencia", "pediatria": "Pediatría",
        "upc": "UPC Adulto", "uci": "UPC Adulto",
        "maternidad": "Maternidad", "hospitalizacion": "Hospitalización",
        "hospitalización": "Hospitalización", "medicina interna": "Medicina Interna"
    }

    anios = list(map(int, re.findall(r"\b(20[2-3][0-9])\b", pregunta)))
    periodo = detectar_periodo(pregunta)
    mes, semestre, trimestre = periodo["mes"], periodo["semestre"], periodo["trimestre"]

    clasificaciones_detectadas = [clasificaciones[c] for c in clasificaciones if re.search(rf"\b{c}\b", pregunta)]
    servicios_detectados = [servicios[s] for s in servicios if re.search(rf"\b{s}\b", pregunta)]

    print(f"🔹 Clasificaciones detectadas: {clasificaciones_detectadas}")
    print(f"🔹 Servicios detectados: {servicios_detectados}")
    print(f"🔹 Años detectados: {anios}")
    print(f"🔹 Periodo detectado: mes={mes}, trimestre={trimestre}, semestre={semestre}")

    return {
        "años": anios if anios else None,
        "año": anios[0] if len(anios) == 1 else None,
        "mes": mes,
        "semestre": semestre,
        "trimestre": trimestre,
        "clasificacion": clasificaciones_detectadas if clasificaciones_detectadas else None,
        "servicio": servicios_detectados if servicios_detectados else None
    }

# ============================
# Responder con datos del DataFrame + Memoria
# ============================
def responder_mistral_automatico(df, pregunta, api_key,
                                año=None, años=None, mes=None, semestre=None,
                                trimestre=None, clasificacion=None, servicio=None,
                                usuario="default"):
    df_filtrado = df.copy()
    print("🔹 Filtrando datos según criterios de la pregunta...")

    if años:
        df_filtrado = df_filtrado[df_filtrado["Año"].isin([int(a) for a in años])]
    elif año:
        df_filtrado = df_filtrado[df_filtrado["Año"] == int(año)]

    if mes:
        df_filtrado = df_filtrado[df_filtrado["Mes_Num"] == mes]

    if semestre:
        rango = (1,6) if semestre==1 else (7,12)
        df_filtrado = df_filtrado[df_filtrado["Mes_Num"].between(rango[0], rango[1])]

    if trimestre:
        rangos_trimestre = {1:(1,3),2:(4,6),3:(7,9),4:(10,12)}
        df_filtrado = df_filtrado[df_filtrado["Mes_Num"].between(*rangos_trimestre[trimestre])]

    if clasificacion and "Clasificación" in df_filtrado.columns:
        if "General" in clasificacion:
            filtro_clasif = ["Adverso","Incidente","Centinela","Descarte"]
        else:
            filtro_clasif = clasificacion
        df_filtrado = df_filtrado[df_filtrado["Clasificación"].str.capitalize().isin(filtro_clasif)]

    if servicio and "Servicio Ocurrencia" in df_filtrado.columns:
        serv_list = [s.lower() for s in servicio] if isinstance(servicio, list) else [servicio.lower()]
        df_filtrado = df_filtrado[df_filtrado["Servicio Ocurrencia"].str.lower().isin(serv_list)]

    cantidad = len(df_filtrado)
    if cantidad == 0:
        return "No se encontraron registros que coincidan con los criterios indicados."

    resumen_datos = f"Cantidad de registros encontrados: {cantidad}.\nServicios más frecuentes:\n"
    servicios_top = df_filtrado["Servicio Ocurrencia"].value_counts().head(5)
    for serv, cnt in servicios_top.items():
        resumen_datos += f"- {serv}: {cnt} eventos\n"

    clasificaciones_top = df_filtrado["Clasificación"].value_counts().head(5)
    resumen_datos += "Clasificaciones más frecuentes:\n"
    for clasif, cnt in clasificaciones_top.items():
        resumen_datos += f"- {clasif}: {cnt} eventos\n"

    texto_tabla = df_filtrado.head(5).to_string(index=False)

    # --- Memoria con resumen automático ---
    contexto, resumido = construir_contexto_memoria_relevante(pregunta, n_recientes=5, api_key=api_key)
    if resumido:
        print("⚠️ Se resumió la memoria antigua para evitar sobrepasar tokens.")

    prompt = (
        f"{contexto}\n\n"
        "Eres un asistente médico experto que responde en español.\n"
        "El usuario hizo la siguiente pregunta:\n"
        f"{pregunta}\n\n"
        "A continuación tienes datos relevantes encontrados:\n"
        f"{resumen_datos}\n\n"
        "Aquí un extracto de los datos:\n"
        f"{texto_tabla}\n\n"
        "Por favor, responde de forma clara y detallada, brindando análisis, sugerencias, posibles mejoras y orientaciones que el usuario podría aplicar en base a esta información."
    )

    respuesta_ia = consultar_mistral(prompt, api_key)

    try:
        guardar_respuesta_en_memoria(pregunta, respuesta_ia, usuario=usuario, meta={"filtros": {
            "año": año, "años": años, "mes": mes, "semestre": semestre,
            "trimestre": trimestre, "clasificacion": clasificacion, "servicio": servicio
        }})
    except Exception as e:
        print("⚠️ Error guardando memoria:", e)

    return respuesta_ia

# ============================
# Borrar índice FAISS
# ============================
def borrar_indice_faiss():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        print("✅ Índice FAISS eliminado.")
