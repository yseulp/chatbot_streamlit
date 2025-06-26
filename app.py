import streamlit as st
from ui_chat import render_chat_ui
from ui_kapitel import render_kapitel_ui
from ui_game import render_game_ui

if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"

# Sidebar
st.sidebar.image("PTW_CiP_Logo.svg", width=300)
st.sidebar.header("Modus wählen")
mode = st.sidebar.radio("Wähle einen Modus", ["Normal Chat", "Kapitel-Modus"])

# Main
if mode == "Normal Chat":
    render_chat_ui()
elif mode == "Kapitel-Modus":
    render_kapitel_ui()



