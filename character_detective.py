import streamlit as st
import random
import uuid
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from utils.load_quiz_data import load_quiz_catalog
from utils.feedback_tools import get_feedback, get_progressive_hint
from utils.topic_selector import get_available_topics
from utils.chat_history_memory import save_message, retrieve_similar_history

# === Prompts ===
story_prompt = PromptTemplate.from_template("""
You are Inspector Kaizen – a legendary Lean Production detective summoned to investigate hidden inefficiencies.

Based on the following selected clues, generate a story-style case briefing for the inspector:
- Each clue is a Lean-related challenge described as a question.
- Mention the domains/topics if available.
- Use storytelling style, grounded in industrial reality.
- Include one reference to a prior case if relevant: {history_snippets}
- End with a call to action that the inspector must investigate 4 specific areas.

Selected clues:
{clues}

Return ONLY the case narrative.
""")

summary_prompt = PromptTemplate.from_template("""
You are Inspector Kaizen, wrapping up a Lean Production case.

Based on the following clue outcomes, write a detailed case summary:
- Reflect on what went well and what didn’t.
- Mention what the inspector learned and what to improve.
- Optionally connect to past investigations if helpful: {history_snippets}
- Keep it immersive and story-like.

Clue results:
{results}

Return only the summary text.
""")

# === Main Streamlit Function ===
def run_detective_mode_streamlit():
    st.markdown("## 🕵️ Lean Detective")
    llm = OllamaLLM(model="openhermes")
    session_id = st.session_state.get("detective_session_id", str(uuid.uuid4()))
    st.session_state.detective_session_id = session_id

    difficulty_levels = ["easy", "medium", "hard"]

    if "detective_state" not in st.session_state:
        quiz_catalog = load_quiz_catalog()
        selected_topic = st.session_state.get("current_topic")

        if selected_topic and selected_topic in quiz_catalog and len(quiz_catalog[selected_topic]) >= 4:
            sorted_questions = sorted(
                quiz_catalog[selected_topic],
                key=lambda q: ["easy", "medium", "hard"].index(q.get("difficulty", "medium"))
            )
            story_clues = "\n".join([f"- ({q['difficulty'].title()}) {q['question']} [{selected_topic}]" for q in sorted_questions[:4]])
            history_snippets = "\n".join(retrieve_similar_history("lean detective case", k=2, role="detective"))

            with st.spinner("🧠 Generiere Fallbeschreibung..."):
                story = llm.invoke(story_prompt.format(
                    clues=story_clues,
                    history_snippets=history_snippets
                ))

            st.session_state.detective_state = {
                "step": "show_clue",
                "quiz_catalog": quiz_catalog,
                "topics": [selected_topic],
                "topic": selected_topic,
                "questions": sorted_questions,
                "difficulty_index": 0,
                "log": [("detective", story)],
                "clue_results": [],
                "question_index": 0,
                "hint_used": False,
                "correct_answers": 0,
                "last_result": None
            }

            save_message("detective", story)
            st.rerun()
        else:
            topics = [t for t, qs in quiz_catalog.items() if qs and len(qs) >= 4]
            st.session_state.detective_state = {
                "step": "select_topic",
                "quiz_catalog": quiz_catalog,
                "topics": topics,
                "topic": None,
                "questions": [],
                "difficulty_index": 0,
                "log": [],
                "clue_results": [],
                "question_index": 0,
                "hint_used": False,
                "correct_answers": 0,
                "last_result": None
            }

    state = st.session_state.detective_state

    if state.get("topic"):
        st.markdown(f"🧩 **Fallthema:** _{state['topic']}_")

    for role, message in state["log"]:
        align = "left" if role == "detective" else "right"
        st.markdown(
            f"<div style='text-align: {align}; padding: 10px; border-radius: 8px; background-color: #f1f1f1; margin: 5px;'>{message}</div>",
            unsafe_allow_html=True
        )

    if state["question_index"] >= 4:
        clue_summary = "\n".join(state["clue_results"])
        history_snippets = "\n".join(retrieve_similar_history("detective summary", k=2, role="detective", session_id=session_id))
        summary = llm.invoke(summary_prompt.format(results=clue_summary, history_snippets=history_snippets))
        save_message("detective", summary, session_id=session_id)
        state["log"].append(("detective", summary))
        st.success("✅ Fall abgeschlossen!")
        if st.button("🔁 Neuer Fall starten"):
            del st.session_state.detective_state
            st.rerun()
        return

    q = state["questions"][state["question_index"]]
    st.markdown(f"### 🧩 Clue {state['question_index'] + 1} ({q['difficulty'].title()}):\n> {q['question']}")

    if q.get("options"):
        st.markdown("**Antwortmöglichkeiten:**")
        for opt in q["options"]:
            st.markdown(f"- {opt}")

    answer = st.text_input("Deine Deduktion:", key=f"answer_{state['question_index']}")

    if not state["hint_used"] and st.button("💡 Hinweis anzeigen"):
        tip = get_progressive_hint(q["question"], q["correct_answer"], level=3)
        if tip:
            st.info(f"🧠 Hinweis: {tip}")
            state["hint_used"] = True

    if st.button("✅ Antwort prüfen"):
        is_correct, feedback = get_feedback(q["question"], q["correct_answer"], answer)

        result_text = f"- Clue {state['question_index'] + 1} ({q['difficulty'].title()}): {'✔️ Correct' if is_correct else '❌ Incorrect'} — {q['question']}"
        state["clue_results"].append(result_text)
        state["log"].append(("detective", f"🔍 Frage: {q['question']}\n📝 Deine Antwort: {answer}\n📘 Feedback: {feedback}"))
        save_message("detective", result_text, session_id=session_id)

        state["last_result"] = (is_correct, feedback, q["question"], answer)
        state["step"] = "show_feedback"
        st.rerun()

    if state.get("step") == "show_feedback" and state.get("last_result"):
        is_correct, feedback, question, user_answer = state["last_result"]
        st.markdown("---")
        st.markdown("### 🧾 Ergebnis & Feedback")
        st.markdown(f"**Frage:** {question}")
        st.markdown(f"**Deine Antwort:** {user_answer}")
        if is_correct:
            st.success("✔️ Deine Antwort ist korrekt!")
            state["correct_answers"] += 1

            # Schwierigkeitsgrad erhöhen (max = 2)
            state["difficulty_index"] = min(state["difficulty_index"] + 1, 2)
            next_difficulty = difficulty_levels[state["difficulty_index"]]

            # Neue Frage mit höherem Schwierigkeitsgrad hinzufügen
            all_questions = state["quiz_catalog"][state["topic"]]
            candidates = [
                q for q in all_questions
                if q.get("difficulty") == next_difficulty and q not in state["questions"]
            ]
            if candidates:
                next_q = random.choice(candidates)
                state["questions"].append(next_q)
        else:
            st.error(f"❌ Falsch: {feedback}")
            same_level = [q2 for q2 in state["questions"] if q2["difficulty"] == q["difficulty"] and q2 != q]
            if same_level:
                state["questions"].insert(state["question_index"] + 1, random.choice(same_level))
            state["difficulty_index"] = max(state["difficulty_index"] - 1, 0)

        if st.button("➡️ Weiter zum nächsten Hinweis"):
            state["question_index"] += 1
            state["hint_used"] = False
            state["step"] = "show_clue"
            state["last_result"] = None
            st.rerun()
