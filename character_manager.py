import streamlit as st
import random
import uuid
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from utils.load_quiz_data import load_quiz_catalog
from utils.feedback_tools import get_feedback, get_progressive_hint, get_topic_summary, get_question_context
from utils.chat_history_memory import save_message, retrieve_similar_history

# === Prompt Template ===
manager_prompt = PromptTemplate.from_template("""
You are a helpful and experienced Lean Production shopfloor manager.
You're having a conversation with a colleague about the topic: "{topic}".

This is the current user question:
"{question}"

Here are some related past chat segments to inform your response:
{history}

Now, answer informally, supportively, and practically – like a manager teaching on the shopfloor.
""")

@st.cache_resource
def get_llm():
    return OllamaLLM(model="openhermes")

def run_manager_mode_streamlit():
    st.markdown("## 👷 Shopfloor Manager")

    current_topic = st.session_state.get("current_topic", "")
    quiz_catalog = load_quiz_catalog()

    if "manager_state" not in st.session_state:
        st.session_state.manager_state = {
            "step": "intro",
            "log": [],
            "difficulty_index": 0,
            "question": None,
            "session_id": str(uuid.uuid4()),
            "topic": current_topic
        }
    else:
        # Thema hat sich geändert → reset
        if st.session_state.manager_state.get("topic") != current_topic:
            del st.session_state.manager_state
            st.rerun()

    state = st.session_state.manager_state
    session_id = state["session_id"]
    topic = state["topic"]
    llm = get_llm()

    # Thema anzeigen
    st.markdown(f"### 📘 Aktuelles Thema: *{topic}*")

    # Intro erzeugen
    if state["step"] == "intro":
        intro = get_topic_summary(topic)
        state["log"].append(("assistant", f"Welcome to the Gemba! Today's topic is *{topic}*\n\n{intro}"))
        state["log"].append(("assistant", "Type `start quiz` to begin or ask any question about the topic."))
        state["step"] = "chat"
        st.rerun()

    # Chatverlauf anzeigen
    for speaker, msg in state["log"]:
        align = "right" if speaker == "user" else "left"
        bgcolor = "#dcf8c6" if speaker == "user" else "#f1f0f0"
        name = "You" if speaker == "user" else "Assistant"
        st.markdown(
            f"""
            <div style='text-align: {align}; background-color: {bgcolor}; padding: 10px; border-radius: 12px; margin-bottom: 5px;'>
                <b>{name}:</b><br>{msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Eingabe verarbeiten
    user_input = st.chat_input("Frage stellen oder `start quiz` eingeben")
    if user_input:
        state["log"].append(("user", user_input))
        save_message("user", user_input, session_id=session_id)

        if user_input.lower() == "quit":
            state["log"].append(("assistant", "👋 Session ended. See you next time on the shopfloor!"))
            state["step"] = "finished"
            st.rerun()

        elif user_input.lower() == "start quiz":
            state["log"].append(("assistant", "🧠 Great! Let's test your knowledge now."))
            state["step"] = "quiz"
            difficulty_levels = ["easy", "medium", "hard"]
            questions = [q for q in quiz_catalog[topic] if q.get("difficulty") == difficulty_levels[state["difficulty_index"]]]

            if not questions:
                state["log"].append(("assistant", f"⚠️ Keine Fragen für {difficulty_levels[state['difficulty_index']]} vorhanden."))
                state["step"] = "finished"
            else:
                q = random.choice(questions)
                state["question"] = q
                state["log"].append(("assistant", f"📚 Kontext: {get_question_context(q['question'], difficulty_levels[state['difficulty_index']])}"))
                state["log"].append(("assistant", f"🔧 Frage: {q['question']}"))
                if q.get("options"):
                    state["log"].append(("assistant", "\n".join(q["options"])))
            st.rerun()

        elif user_input.lower() == "chat":
            state["step"] = "chat"
            state["log"].append(("assistant", "💬 Back to discussion mode!"))
            state["question"] = None
            st.rerun()

        elif user_input.lower() == "hint" and state.get("question"):
            q = state["question"]
            hint = get_progressive_hint(q["question"], q["correct_answer"])
            state["log"].append(("assistant", f"💡 Tipp: {hint}"))
            st.rerun()

        elif state["step"] == "chat":
            history = retrieve_similar_history(user_input, role="manager", session_id=session_id)
            context = "\n".join(f"- {h}" for h in history) if history else "No previous context available."
            reply = llm.invoke(manager_prompt.format(topic=topic, question=user_input, history=context))
            state["log"].append(("assistant", reply))
            save_message("manager", reply, session_id=session_id)
            st.rerun()

        elif state["step"] == "quiz":
            q = state.get("question")
            difficulty_levels = ["easy", "medium", "hard"]

            if q is None:
                questions = [q for q in quiz_catalog[topic] if q.get("difficulty") == difficulty_levels[state["difficulty_index"]]]
                if not questions:
                    state["log"].append(("assistant", f"⚠️ Keine Fragen für {difficulty_levels[state['difficulty_index']]} vorhanden."))
                    state["step"] = "finished"
                    st.rerun()
                q = random.choice(questions)
                state["question"] = q
                state["log"].append(("assistant", f"📚 Kontext: {get_question_context(q['question'], difficulty_levels[state['difficulty_index']])}"))
                state["log"].append(("assistant", f"🔧 Frage: {q['question']}"))
                if q.get("options"):
                    state["log"].append(("assistant", "\n".join(q["options"])))
                st.rerun()

            else:
                is_correct, feedback = get_feedback(q["question"], q["correct_answer"], user_input)
                state["log"].append(("assistant", f"📜 {feedback}"))
                save_message("manager", feedback, session_id=session_id)

                if is_correct:
                    state["log"].append(("assistant", "✅ Nice work! Leveling up your knowledge."))
                    state["difficulty_index"] = min(state["difficulty_index"] + 1, 2)
                else:
                    state["log"].append(("assistant", "📉 Not quite. Let's stay on this level."))
                state["question"] = None
                state["log"].append(("assistant", "Type 'chat' to switch or 'continue' answering."))
                st.rerun()

    if state["step"] == "finished":
        if st.button("🔁 Neustart"):
            del st.session_state.manager_state
            st.rerun()
