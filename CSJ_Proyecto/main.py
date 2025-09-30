import streamlit as st
from pathlib import Path
import importlib.util
from streamlit_option_menu import option_menu

# ==============================
# Estado inicial de sesión
# ==============================
if "menu_option" not in st.session_state:
    st.session_state["menu_option"] = "🏠 Inicio"

# ==============================
# Ejecutar vistas
# ==============================
def run_page(path: Path):
    """Carga y ejecuta dinámicamente el archivo de la vista."""
    if not path.exists():
        st.error(f"⚠️ No se encontró el archivo: {path}")
        return
    spec = importlib.util.spec_from_file_location("modulo", path)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if hasattr(modulo, "main"):
        modulo.main()
    else:
        st.warning(f"⚠️ El archivo `{path.name}` no contiene una función main().")

# ==============================
# Main
# ==============================
def main():
    # Páginas disponibles
    paginas = {
        "🏠 Inicio": "Inicio",
        "📄 Informes Clínicos": "Informes_1",
        "📊 Gráficos": "Graficos",
        "📋 Eventos Adversos": "eventos_adversos",
        "📈 Predictor Eventos": "Predictiva3",
        "🤖 Asistente IA": "Asistente_IA"
        
    }
    opciones_menu = list(paginas.keys())

    # ================== Sidebar ==================
    with st.sidebar:
        st.write("### 📁 Navegación")
        seleccion = option_menu(
            menu_title="",
            options=opciones_menu,
            icons=["house", "file-earmark-text", "bar-chart", "clipboard-data", "robot"],
            default_index=opciones_menu.index(st.session_state["menu_option"]),
            styles={
                "container": {"padding": "5px", "background-color": "#f8f9fa"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
                "nav-link-selected": {"background-color": "#c50707", "color": "white"},
            },
            key="menu_selector"
        )

    # ================== Actualizar selección ==================
    if seleccion != st.session_state["menu_option"]:
        st.session_state["menu_option"] = seleccion
        st.rerun()

    # ================== Ejecutar vista ==================
    nombre_modulo = paginas[st.session_state["menu_option"]]
    archivo_path = Path("views") / f"{nombre_modulo}.py"
    run_page(archivo_path)

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()