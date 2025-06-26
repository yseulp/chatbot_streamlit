import streamlit as st
import random
import uuid
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from utils.load_quiz_data import load_quiz_catalog
from utils.feedback_tools import get_feedback, get_progressive_hint, get_lecture_context
from utils.chat_history_memory import save_message, retrieve_similar_history

llm = OllamaLLM(model="openhermes")

lecture_intro_prompt = PromptTemplate.from_template("""
You are a university professor for Lean Production and operations management.
Your tone is academic, structured, and engaging. You speak to master's students.

Give a short introductory lecture (max 6 sentences) about the topic: "{topic}" and the difficulty level: "{level}".

Include:
- Real-world relevance
- Typical challenges
- What learners should focus on

Here is what you have already said in previous lectures:
{history}

Your response should resemble a short lecture excerpt. Return only the lecture.
""")

reflection_prompt = PromptTemplate.from_template("""
You are a Lean Production university professor summarizing a completed learning unit.

Based on this:
- Topic: "{topic}"
- Difficulty: "{level}"
- User correctness: {correct_count} out of {total_count}
- User answers and professor feedback:
{feedback_list}

Create a short summary lecture (4–6 sentences), reflecting on what went well, what students typically struggle with, and what the student should focus on next.

Your tone is professional but encouraging. End with a motivating sentence.
""")

def run_professor_mode_streamlit():
    current_topic = st.session_state.get("current_topic")
    if "professor_session_id" not in st.session_state:
        st.session_state.professor_session_id = str(uuid.uuid4())

    if "professor_state" not in st.session_state:
        st.session_state.professor_state = {
            "step": "select_mode",
            "mode": None,
            "topic": current_topic,
            "catalog": load_quiz_catalog(),
            "question_queue": [],
            "current_question": None,
            "hint_count": 0,
            "correct_total": 0,
            "feedback_log": []
        }

    state = st.session_state.professor_state
    session_id = st.session_state.professor_session_id
    st.markdown("## 👨‍🏫 Professor Mode")

    # Handle topic change
    if state.get("topic") != current_topic:
        del st.session_state.professor_state
        st.rerun()

    topic = current_topic or "Mixed Topics"
    state["topic"] = topic

    # Show current topic
    st.markdown(f"### 📘 Aktuelles Thema: *{topic}*")

    # Step: Mode Selection
    if state["step"] == "select_mode":
        st.markdown("📘 **Choose your learning mode:**")
        mode = st.radio("Select mode:", ["📘 Basic", "📗 Advanced", "📕 Expert", "🎓 Exam"])
        if st.button("➡️ Continue"):
            mode_map = {
                "📘 Basic": "easy",
                "📗 Advanced": "medium",
                "📕 Expert": "hard",
                "🎓 Exam": "exam"
            }
            state["mode"] = mode_map[mode]
            state["step"] = "lecture"
            st.rerun()

    # Step: Lecture
    elif state["step"] == "lecture":
        history = "\n".join(retrieve_similar_history(topic, k=2, role="professor", session_id=session_id))
        intro = llm.invoke(lecture_intro_prompt.format(topic=topic, level=state["mode"], history=history))
        st.markdown(f"### 🎙️ Lecture Introduction:\n{intro}")
        if topic != "Mixed Topics":
            st.info(get_lecture_context(topic, state["mode"]))
        if st.button("🧠 Start Quiz"):
            catalog = state["catalog"]
            if state["mode"] == "exam":
                all_qs = [q for qs in catalog.values() for q in qs]
                state["question_queue"] = random.sample(all_qs, min(15, len(all_qs)))
            else:
                filtered = [q for q in catalog.get(topic, []) if q.get("difficulty") == state["mode"]]
                state["question_queue"] = random.sample(filtered, min(5, len(filtered)))
            state["step"] = "ask_question"
            st.rerun()

    elif state["step"] == "ask_question":
        if state["current_question"] is None and state["question_queue"]:
            state["current_question"] = state["question_queue"].pop(0)

        q = state["current_question"]
        st.markdown(f"### ❓ Question: {q['question']}")

        if "options" in q:
            selected_option = st.radio("Choose an answer:", q["options"], key="prof_answer_radio")
        else:
            selected_option = st.text_input("Your answer:", key="prof_answer_input")

        if st.button("💡 Hint"):
            state["hint_count"] += 1
            hint_text = get_progressive_hint(q["question"], q["correct_answer"], state["hint_count"])
            st.warning(hint_text)

        if st.button("✅ Check Answer"):
            is_correct, feedback = get_feedback(q["question"], q["correct_answer"], selected_option, q.get("options"))
            state["correct_total"] += int(is_correct)
            state["feedback_log"].append(f"Q: {q['question']} → {selected_option} → {feedback}")
            save_message("user", selected_option, session_id)
            save_message("professor", feedback, session_id)
            state["last_feedback"] = (is_correct, feedback)
            state["step"] = "show_feedback"
            st.rerun()

    elif state["step"] == "show_feedback":
        is_correct, feedback = state.get("last_feedback", (False, "Kein Feedback."))
        st.markdown(f"### 📝 Feedback")
        st.success("✅ Correct!") if is_correct else st.error(f"{feedback}")

        if st.button("➡️ Nächste Frage"):
            state["current_question"] = None
            state["last_feedback"] = None
            state["step"] = "ask_question" if state["question_queue"] else "reflect"
            st.rerun()

    # Step: Reflection
    elif state["step"] == "reflect":
        summary = reflection_prompt.format(
            topic=topic,
            level=state["mode"],
            correct_count=state["correct_total"],
            total_count=len(state["feedback_log"]),
            feedback_list="\n".join(state["feedback_log"])
        )
        reflection = llm.invoke(summary)
        st.markdown(f"### 📘 Professor's Reflection:\n{reflection}")
        if st.button("🔁 New Session"):
            del st.session_state.professor_state
            st.rerun()