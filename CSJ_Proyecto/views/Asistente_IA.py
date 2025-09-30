import streamlit as st
import time
from asistente import cargar_datos, extraer_elementos_pregunta, responder_mistral_automatico

def main():
    # ========== Configuración de página ==========
    st.set_page_config(page_title="🧠 Asistente Clínico IA", layout="wide")
    st.title("🧠 Asistente Clínico")
    st.info(
        "⚠️La información entregada por este asistente es solo con fines de sugerencia "
        "y análisis clínico referencial. No debe ser utilizada como base única para la toma de decisiones "
        "estratégicas, Clínicas o administrativas."
    )
    st.write("📁 Haz preguntas sobre los archivos cargados")

    # ========== Cargar datos solo una vez ==========
    if "df" not in st.session_state:
        st.session_state.df = cargar_datos()

    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    # ========== Función para agregar mensajes ==========
    def agregar_mensaje(usuario: str, mensaje: str):
        st.session_state.chat_historial.append({"usuario": usuario, "mensaje": mensaje})

    # ========== Procesar la pregunta ==========
    def procesar_pregunta(pregunta):
        pregunta = pregunta.strip()
        if not pregunta:
            return

        agregar_mensaje("usuario", pregunta)
        tiempo_inicio = time.time()

        try:
            elementos = extraer_elementos_pregunta(pregunta)
        except Exception as e:
            agregar_mensaje("asistente", f"⚠️ No pude interpretar la pregunta correctamente: {e}")
            return

        with st.spinner("🧠 Generando respuesta..."):
            try:
                tiempo_modelo_ini = time.time()
                respuesta = responder_mistral_automatico(
                    df=st.session_state.df,
                    pregunta=pregunta,
                    api_key=st.secrets["MISTRAL_API_KEY"],
                    año=elementos.get("año"),
                    años=elementos.get("años"),
                    mes=elementos.get("mes"),
                    clasificacion=elementos.get("clasificacion"),
                    servicio=elementos.get("servicio"),
                )
                tiempo_modelo_fin = time.time()
            except Exception as e:
                agregar_mensaje("asistente", f"❌ Error al consultar a la IA: {e}")
                return

        agregar_mensaje("asistente", respuesta)

        duracion_total = round(time.time() - tiempo_inicio, 2)
        duracion_modelo = round(tiempo_modelo_fin - tiempo_modelo_ini, 2)

        if duracion_modelo > 10:
            st.warning(f"⚠️ La IA tardó {duracion_modelo} segundos en responder. Puede haber congestión en el modelo o tu red.")
        else:
            st.caption(f"🕒 Tiempo IA: {duracion_modelo} s | Tiempo total: {duracion_total} s")

    # ========== Botón para limpiar conversación ==========
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🧹 Limpiar conversación"):
            st.session_state.chat_historial = []
            st.rerun()

    # ========== Mostrar historial de chat ==========
    st.markdown("---")
    st.markdown("<h4 style='margin-bottom: 15px;'>💬 Conversación</h4>", unsafe_allow_html=True)

    for mensaje in st.session_state.chat_historial:
        if mensaje["usuario"] == "usuario":
            st.markdown(
                f'<div style="background:#DCF8C6; padding:10px; border-radius:10px; max-width:60%; margin-left:auto; margin-bottom:8px;">'
                f'<b>Tú:</b> {mensaje["mensaje"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background:#F1F0F0; padding:10px; border-radius:10px; max-width:60%; margin-right:auto; margin-bottom:8px;">'
                f'<b>Asistente:</b> {mensaje["mensaje"]}</div>',
                unsafe_allow_html=True
            )

    # ========== Formulario de entrada ==========
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.form(key="formulario_pregunta", clear_on_submit=True):
        texto_pregunta = st.text_input("📝 Escribe tu pregunta:", key="input_usuario")
        enviar = st.form_submit_button("Enviar")
        if enviar:
            procesar_pregunta(texto_pregunta)
            st.rerun()  # 🔁 Para que se actualice el historial al instante

if __name__ == "__main__":
    main()