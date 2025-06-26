import streamlit as st
import random
import uuid
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from utils.load_quiz_data import load_quiz_catalog
from utils.feedback_tools import get_feedback
from utils.chat_history_memory import save_message, retrieve_similar_history

# === Prompts ===
intro_prompt = PromptTemplate.from_template("""
You're a friendly and knowledgeable colleague at a Lean Production company.
Your tone is supportive, inspiring, and informal – you're here to help your coworker learn through discussion.

Provide a brief, welcoming introduction to the topic: "{topic}".
Mention why this topic matters in real-world operations, and what kinds of practical issues arise around it.

Keep it short (max 5 sentences), engaging, and personal – like you'd say over a coffee break.
""")

chat_prompt_template = PromptTemplate.from_template("""
You are a helpful and supportive Lean Production colleague.
You're discussing the topic "{topic}" with a teammate in an informal tone.

Here’s some context from your previous chat:
{history}

Now respond to this user message:
"{message}"

Keep your tone natural, clear, and helpful – like a friendly, experienced coworker.
Don't ask a quiz question here – just respond conversationally.
""")

quiz_chat_prompt_template = PromptTemplate.from_template("""
You are a Lean Production colleague having an informal chat.
You're discussing the topic "{topic}" with a teammate.

Here’s some context from your previous chat:
{history}

Now respond to the following user message in a friendly and informative way:
"{message}"

Then ask a related question (difficulty: {difficulty}) to keep the learning engaging.
The question is:
{question}

End with the question only.
""")

def run_colleague_mode_streamlit():
    current_topic = st.session_state.get("current_topic")
    if not current_topic:
        st.error("❗ Kein Thema ausgewählt. Bitte zuerst ein Thema wählen.")
        return

    llm = OllamaLLM(model="openhermes")
    session_id = st.session_state.get("colleague_session_id", str(uuid.uuid4()))
    st.session_state.colleague_session_id = session_id

    # Reset bei Themenwechsel
    if st.session_state.get("colleague_topic") != current_topic:
        for k in ["colleague_topic", "colleague_messages", "colleague_remaining_questions", "colleague_used_questions", "colleague_response_count", "colleague_correct_streak", "colleague_difficulty_index"]:
            st.session_state.pop(k, None)
        st.session_state.colleague_topic = current_topic
        intro = llm.invoke(intro_prompt.format(topic=current_topic))
        st.session_state.colleague_messages = [f"💬 Colleague: {intro}"]
        save_message("colleague", intro, session_id=session_id)

    topic = st.session_state.colleague_topic

    if "colleague_messages" not in st.session_state:
        st.session_state.colleague_messages = []
    if "colleague_remaining_questions" not in st.session_state:
        quiz_catalog = load_quiz_catalog()
        st.session_state.colleague_remaining_questions = quiz_catalog.get(topic, []).copy()
        st.session_state.colleague_used_questions = set()
        st.session_state.colleague_response_count = 0
        st.session_state.colleague_correct_streak = 0
        st.session_state.colleague_difficulty_index = 0

    st.markdown("## 🤝 Colleague Mode")
    st.markdown(f"### 📘 Aktuelles Thema: *{topic}*")

    for msg in st.session_state.colleague_messages:
        st.markdown(msg)

    user_input = st.chat_input("Was möchtest du deinem Kollegen sagen?")
    if user_input:
        save_message("user", user_input, session_id=session_id)
        st.session_state.colleague_response_count += 1
        context = "\n".join(retrieve_similar_history(user_input, role="colleague", session_id=session_id))
        st.session_state.colleague_messages.append(f"👤 You: {user_input}")

        if st.session_state.colleague_response_count % 2 == 0:
            difficulty_levels = ["easy", "medium", "hard"]
            difficulty = difficulty_levels[st.session_state.colleague_difficulty_index]
            remaining = st.session_state.colleague_remaining_questions
            questions = [q for q in remaining if q["difficulty"] == difficulty] or remaining

            if questions:
                q = random.choice(questions)
                remaining.remove(q)
                st.session_state.colleague_used_questions.add(q["question"])
                response = llm.invoke(quiz_chat_prompt_template.format(
                    topic=topic,
                    message=user_input,
                    difficulty=difficulty,
                    history=context,
                    question=q["question"]
                ))
                st.session_state.colleague_messages.append(f"💬 Colleague: {response}")
                save_message("colleague", q["question"], session_id=session_id)

                answer = st.text_input("Deine Antwort:", key=f"answer_{st.session_state.colleague_response_count}")
                if answer:
                    correct, feedback = get_feedback(q["question"], q["correct_answer"], answer)
                    if correct:
                        st.success("✅ Richtig! Frag mich was!")
                        st.session_state.colleague_correct_streak += 1
                        if st.session_state.colleague_correct_streak >= 2:
                            st.session_state.colleague_difficulty_index = min(
                                st.session_state.colleague_difficulty_index + 1, 2
                            )
                    else:
                        st.error(f"❌ Nicht ganz richtig. Feedback: {feedback}")
                        st.session_state.colleague_correct_streak = 0
                        st.session_state.colleague_difficulty_index = max(
                            st.session_state.colleague_difficulty_index - 1, 0
                        )
            else:
                st.info("🎉 Alle Fragen für dieses Thema sind beantwortet!")
        else:
            reply = llm.invoke(chat_prompt_template.format(
                topic=topic,
                message=user_input,
                history=context
            ))
            st.session_state.colleague_messages.append(f"💬 Colleague: {reply}")
            save_message("colleague", reply, session_id=session_id)

        st.rerun()