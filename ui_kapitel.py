# ui_kapitel.py
import streamlit as st
import pickle
import random
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from ui_game import render_game_ui
from ui_chat import render_chat_ui
import os
from utils.topic_selector import get_available_topics

def show_themes():
    available_topics = get_available_topics()
    selected_topic = st.selectbox("Wähle ein Thema", available_topics, key="kapitel_topic_select")

    # Wenn Thema sich geändert hat, reset
    if "current_topic" in st.session_state and st.session_state.current_topic != selected_topic:
        # Reset für Detective/Game-Modi
        if "detective_state" in st.session_state:
            del st.session_state.detective_state
        if "detective_chat" in st.session_state:
            del st.session_state.detective_chat

        st.session_state.current_topic = selected_topic
        st.rerun()  # Thema geändert → neu laden
    else:
        st.session_state.current_topic = selected_topic


def render_kapitel_ui():
    col1, col2 = st.columns([2,0.25])
    with col1: 
        st.subheader(":closed_book: Kapitel-Modus")
    with col2:
        if st.button("📃", help="pdf upload"):
            st.toast("Noch nicht implementiert")
    
    #Learn Mode
    learn_mode = st.radio("Wähle Lernmodus", ["Chat", "Game-Modus", "Trainer-Unterstützung"], horizontal=True)
    
    if "chapter_titles" not in st.session_state:
        st.session_state.chapter_titles = ["Kapitel 1: Waste_Reduction", "Kapitel 2: Thema X", "Kapitel 3: Thema Y"]
        st.session_state.current_chapter = 0

    chapter = st.selectbox("Wähle ein Kapitel", st.session_state.chapter_titles, key="chapter_select")
    steps = {
        "Kapitel 1: Waste_Reduction",
        "Kapitel 2: Thema X",
        "Kapitel 3: Thema Y"
    }


    model = st.session_state.get("model")
    if not model:
        st.warning("Bitte zuerst ein Modell im Normalmodus auswählen")
        return

 
    # === CHAT MODUS === #
    if learn_mode == "Chat":
        show_themes()
        for msg in st.session_state.messages:
            with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                st.markdown(msg.content)

        prompt = st.chat_input(f"Frage zu {st.session_state.current_topic}")
        if prompt:
            st.session_state.messages.append(HumanMessage(prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
            llm = ChatOllama(model=model)
            response = llm.invoke(st.session_state.messages).content
            st.session_state.messages.append(AIMessage(response))
            with st.chat_message("assistant"):
                st.markdown(response)

    # === GAME MODUS === #
    elif learn_mode == "Game-Modus":
        show_themes()
        if "current_topic" not in st.session_state:
            st.warning("Bitte wähle zuerst ein Thema aus.")
            return
        render_game_ui()


    # === TRAINER MODUS === #
    elif learn_mode == "Trainer-Unterstützung":
        quiz_catalog = {}
        quiz_dir = "quiz_catalogs"
        summary_dir = "topic_summaries"

        for file in os.listdir(quiz_dir):
            if file.endswith(".pkl"):
                path = os.path.join(quiz_dir, file)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        topic = file.replace(".pkl", "")
                        quiz_catalog[topic] = data
                except Exception as e:
                    st.warning(f"⚠️ Konnte {file} nicht laden: {e}")

        if not quiz_catalog:
            st.error("❌ Keine Quiz-Dateien gefunden oder fehlerhaft.")
            return

        if "trainer_selected" not in st.session_state:
            st.session_state.trainer_selected = set()
        if "trainer_questions" not in st.session_state:
            st.session_state.trainer_questions = {}
        if "checked_answers" not in st.session_state:
            st.session_state.checked_answers = set()

        selected_topics = st.multiselect("Themen auswählen", list(quiz_catalog.keys()))

        for topic in selected_topics:
            if topic not in st.session_state.trainer_selected:
                st.session_state.trainer_questions[topic] = random.sample(
                    quiz_catalog[topic], min(2, len(quiz_catalog[topic]))
                )
                st.session_state.trainer_selected.add(topic)

            st.markdown(f"### Thema: {topic}")
            questions = st.session_state.trainer_questions[topic]

            for idx, q in enumerate(questions):
                st.markdown(f"**Frage {idx + 1}:** {q['question']}")

                input_key = f"{topic}_{idx}_input"
                check_key = f"{topic}_{idx}_checked"

                if q.get("difficulty") == "easy" and "options" in q:
                    answer = st.radio("Antwort:", q["options"], key=input_key)
                else:
                    answer = st.text_input("Antwort eingeben:", key=input_key)

                if st.button(f"Antwort prüfen {topic}-{idx}", key=f"check_{topic}_{idx}"):
                    correct = q["correct_answer"].strip().lower()
                    given = answer.strip().lower()
                    if given == correct:
                        st.success("✅ Richtig!")
                    else:
                        st.error(f"❌ Falsch. Richtige Antwort: {q['correct_answer']}")

            summary_file = os.path.join(summary_dir, topic + ".pkl")
            if os.path.exists(summary_file):
                with open(summary_file, "rb") as f:
                    summary_data = pickle.load(f)
                if st.button(f"📄 Zeige Zusammenfassung zu {topic}", key=f"summary_{topic}"):
                    st.markdown("### Zusammenfassung")
                    st.markdown(summary_data["short"])
                    st.download_button("Herunterladen", summary_data["long"], file_name=f"Zusammenfassung_{topic}.txt")
