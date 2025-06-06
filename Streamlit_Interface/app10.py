import streamlit as st
import subprocess
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from langchain_community.document_loaders import PyPDFLoader
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



# Session State für gespeicherte Chats und Kapitel
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.saved_chats = []  
    st.session_state.current_chapter = None
    st.session_state.current_step = None
if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"

# SIDEBAR: Mode Selection 
st.sidebar.image("PTW_CiP_Logo.svg", width=300)
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
    col_left, col_right = st.columns([3,1])

    with col_left:
        st.subheader("💬 Normaler Chat")
    with col_right:
        selected_model = st.selectbox("🔧 Modell wählen", AVAILABLE_MODELS, key="model_select")

    st.session_state.model = selected_model

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
    col_left, col_right = st.columns([3,1])

    with col_left:
        st.subheader(":closed_book: Kapitel-Modus")
    with col_right:
        selected_model = st.selectbox("🔧 Modell wählen", AVAILABLE_MODELS, key="model_select")

    st.session_state.model = selected_model

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
            st.error(f" Quiz-Datei konnte nicht geladen werden: {e}")
            st.stop()
        
        if "test_results" not in st.session_state:
            st.session_state.test_results = {}
        
        st.markdown("### Kapitelquiz")

        for key, questions in quiz_catalog.items():
            st.markdown(f"## Thema: {key}")

        # Frageformat bereinigen
            if isinstance(questions, dict):
                questions = [questions]
            questions = questions[:2]  # Nur zwei Fragen pro Key

            for idx, q in enumerate(questions, 1):
                st.markdown(f"**Frage {idx}:** {q['question']}")
                user_answer = st.radio("Deine Antwort:", q["options"], key=f"{key}_q{idx}")

                if st.button(f"Antwort prüfen für {key} Frage {idx}", key=f"check_{key}_{idx}"):
                    selected_letter = user_answer.split(":")[0].strip()
                    is_correct = selected_letter == q["correct_answer"]

                    if is_correct:
                        st.success("✅ Richtig!")
                    else:
                        correct_option = next(
                            opt for opt in q["options"] if opt.startswith(q["correct_answer"] + ":")
                        )
                        st.error(f"❌ Falsch. Richtige Antwort: {correct_option}")

                    if key not in st.session_state.test_results:
                        st.session_state.test_results[key] = []
                    st.session_state.test_results[key].append(is_correct)
        
        if st.button("📊 Analyse anzeigen"):
            st.markdown("### 🧠 Schwächenanalyse")
            for key, results in st.session_state.test_results.items():
                total = len(results)
                falsch = results.count(False)
                if falsch > 0:
                    st.warning(f"🔍 **{key}**: {falsch} von {total} falsch beantwortet.")
                else:
                    st.success(f"✅ **{key}**: Alles richtig beantwortet!")


    elif learn_mode == "Zusammenfassen":
        
        chapter_pdfs = {
            "Kapitel 1: Einführung": "pdfs/Chapter 3_Waste_Reduction.pdf", 
        } 
        
        pdf_path = chapter_pdfs.get(chapter)

        if not pdf_path:
            st.warning("Keine PDF-Datei für dieses Kapitel gefunden.")
        else:
            try:
                with st.spinner("Lade und analysiere PDF..."):
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load()
                    full_text = "\n\n".join([page.page_content for page in pages])
            except Exception as e:
                st.error(f"Fehler beim Laden der PDF: {e}")
                st.stop()

            summary_style = st.selectbox(
                "Wie soll die Zusammenfassung aussehen?",
                ["Stichpunkte", "Ausführlich", "Einfach erklärt"]
            )

            if st.button("Kapitel zusammenfassen"):
                llm = ChatOllama(model=st.session_state.model)

                prompt = f"Fasse den folgenden Text aus der Vorlesung im Stil '{summary_style}'zusammen:\n\n{full_text}"
                summary = llm.invoke([HumanMessage(prompt)]).content

                st.markdown("### Zusammenfassung")
                st.markdown(summary)
                
                st.download_button(
                    label=" Zusammenfassung herunterladen",
                    data=summary,
                    file_name=f"Zusammenfassung_{chapter.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

        

    
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
