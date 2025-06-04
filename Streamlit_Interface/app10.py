import streamlit as st
import subprocess
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
import streamlit.components.v1 as components
import pickle

def get_installed_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:
            model_name = line.split()[0]
            models.append(model_name)
        return models
    except Exception as e:
        st.warning(f"⚠️ Fehler beim Abrufen der Ollama-Modelle: {e}")
        return []

AVAILABLE_MODELS = get_installed_ollama_models()

# Seitenspalte für Auswahl
col1, col2 = st.columns([3, 1])
with col1:
    with st.container():
        #st.image("Bilder/cip_logo.jpg", width=100)
        # Titel darunter anzeigen
        st.markdown("""<h1 style="font-size: 4em; color: #333;">Chatbot</h1>""", unsafe_allow_html=True)
        
with col2:
    selected_model = st.selectbox("🔧 Modell wählen", AVAILABLE_MODELS, key="model_select")
    
st.session_state.model = selected_model

# Session State für gespeicherte Chats und Kapitel
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.saved_chats = []  
    st.session_state.current_chapter = None
    st.session_state.current_step = None
if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"

# SIDEBAR: Mode Selection 
st.sidebar.header("Modus wählen")
mode = st.sidebar.radio("Wähle einen Modus", ["Normal Chat", "Kapitel-Modus"])

# Speichern des Chats 
def save_chat():
    chat_number = len(st.session_state.saved_chats) + 1
    st.session_state.saved_chats.append(st.session_state.messages.copy())
    st.session_state.messages = []  # Leert den aktuellen Chatverlauf

#  Neuer Chat Button 
def start_new_chat():
    st.session_state.messages = []

# Moduswechsel und Chat speichern
def handle_mode_change(new_mode):
    if new_mode != st.session_state.mode:
        if len(st.session_state.messages) > 0:
            save_chat()
        start_new_chat()
        st.session_state.mode = new_mode

handle_mode_change(mode)

# Normaler Chat Mode 
if mode == "Normal Chat":
    st.subheader("💬 Normaler Chat")

    # Initialisiere Input-Puffer (für saubere Darstellung)
    if "user_input_buffer" not in st.session_state:
        st.session_state.user_input_buffer = None

    # Chatverlauf anzeigen (immer oben)
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Buttons (Chat speichern, Neuer Chat) 
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("💾 Chat speichern"):
            save_chat()
    with col_btn2:
        if st.button("🆕 Neuer Chat"):
            start_new_chat()

    # Mikrofon-Button unter den Buttons
    mic_button = mic_recorder(
        start_prompt="🎤 Aufnahme starten",
        stop_prompt="⏹️ Aufnahme stoppen",
        just_once=True,
        key="mic"
    )

    # Eingabefeld (unten) 
    user_input = st.chat_input("Stelle eine Frage oder nimm Audio auf")


    # Falls Texteingabe vorhanden: puffern & rerun
    if user_input:
        st.session_state.user_input_buffer = user_input
        st.rerun()

    # Falls Puffer vorhanden → Nachricht verarbeiten
    if st.session_state.user_input_buffer:
        input_text = st.session_state.user_input_buffer
        st.session_state.user_input_buffer = None  # zurücksetzen

        # Nutzer-Nachricht speichern
        st.session_state.messages.append(HumanMessage(input_text))

        # Antwort generieren
        llm = ChatOllama(model=st.session_state.model)
        result = llm.invoke(st.session_state.messages).content

        # Antwort speichern
        st.session_state.messages.append(AIMessage(result))

        # Alles neu rendern
        st.rerun()

#  Kapitel-Modus 
elif mode == "Kapitel-Modus":
    st.subheader("📘 Kapitel-Modus")

    if "chapter_titles" not in st.session_state:
        st.session_state.chapter_titles = ["Kapitel 1: Einführung", "Kapitel 2: Thema X", "Kapitel 3: Thema Y"]
        st.session_state.current_chapter = 0

    chapter = st.selectbox("Wähle ein Kapitel", st.session_state.chapter_titles, key="chapter_select")
    
    steps = {
        "Kapitel 1: Einführung": ["Schritt 1", "Schritt 2", "Schritt 3"],
        "Kapitel 2: Thema X": ["Schritt 1", "Schritt 2"],
        "Kapitel 3: Thema Y": ["Schritt 1", "Schritt 2", "Schritt 3"]
    }

    step = st.selectbox("Wähle einen Schritt", steps[chapter], key="step_select")
    st.session_state.current_chapter = chapter
    st.session_state.current_step = step

    learn_mode = st.radio("Wähle Lernmodus", ["Fragen", "Test", "Zusammenfassen"], horizontal=True)
    # Chatverlauf anzeigen
    if learn_mode == "Fragen":
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            if isinstance(message, AIMessage):  
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        
        prompt = st.chat_input(f"Frage zu {chapter} - {step}")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append(HumanMessage(prompt))

            llm = ChatOllama(model=st.session_state.model)
            result = llm.invoke(st.session_state.messages).content

            with st.chat_message("assistant"):
                st.markdown(result)
                st.session_state.messages.append(AIMessage(result))

    elif learn_mode == "Test":
        try:
            with open("quiz_catalog.pkl", "rb") as f:
                quiz_catalog = pickle.load(f)
        except Exception as e:
            st.error(f"❌ Quiz-Datei konnte nicht geladen werden: {e}")
            st.stop()
        
        chapter_to_quiz_key = {
            "Kapitel 1: Einführung": "Waste Reduction",
            "Kapitel 2: Thema X": "Types of Waste",
            "Kapitel 3: Thema Y": "Spaghetti Diagram Method"
            # weitere Mappings bei Bedarf ergänzen
        }

        mapped_key = chapter_to_quiz_key.get(chapter)
        quiz_data = quiz_catalog.get(mapped_key, [])

        if not quiz_data:
            st.warning("Für dieses Kapitel sind keine Quizfragen vorhanden.")
        else:
            st.markdown("### Kapitelquiz")
            for idx, q in enumerate(quiz_data, 1):
                st.markdown(f"**Frage {idx}:** {q['question']}")
                user_answer = st.radio("Deine Antwort:", q['options'], key=f"answer_{idx}")
                if st.button(f"Antwort prüfen für Frage {idx}", key=f"check_{idx}"):
                    selected_letter = user_answer.split(":")[0].strip()
                    if selected_letter == q["correct_answer"]:
                        st.success("Richtig!")
                    else:
                        correct_option = next(opt for opt in q["options"] if opt.startswith(q["correct_answer"] + ":"))
                        st.error(f"Falsch. Richtige Antwort: {q['correct_answer']}")


    
    # Chat speichern oder neuen Chat starten
    #if st.button("Chat speichern"):
    #    save_chat()

    #if st.button("Neuer Chat"):
    #    start_new_chat()

# Sidebar für gespeicherte Chats 
st.sidebar.header("Gespeicherte Chats")
for i, chat in enumerate(st.session_state.saved_chats, 0):
    chat_summary = f"Chat {i}: {len(chat)} Nachrichten, {mode}"
    if st.sidebar.button(f"🗣️ {chat_summary}", key=f"chat_{i}"):
        st.session_state.messages = chat  
    if st.sidebar.button(f"🗑️ Löschen {i}", key=f"delete_chat_{i}"):
        st.session_state.saved_chats.pop(i)
