# utils/feedback_tools.py

import os
import json
import pickle
from difflib import SequenceMatcher
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === LLM Setup ===
llm = OllamaLLM(model="llama3.2")

# === FEEDBACK ===
_feedback_prompt = ChatPromptTemplate.from_template("""
You are a Lean Production tutor. Analyze the user's answer and provide a clear judgment. The user's answer should only be evaluated semantically. The correct spelling and punctuation is not important.

Question: {question}
Correct Answer: {correct_answer}
User Answer: {user_answer}

Respond in this exact JSON format:
{{
  "is_correct": true or false,
  "feedback": "Brief explanation why the answer is correct or incorrect. If incorrect, always include the correct answer."
}}
""")
_feedback_chain = _feedback_prompt | llm | StrOutputParser()

def get_feedback(question: str, correct_answer: str, user_answer: str, options: list = None) -> tuple[bool, str]:
    user_input_clean = user_answer.strip().lower()
    correct_answer_clean = correct_answer.strip().lower()

    # === MULTIPLE CHOICE HANDLING ===
    if options:
        option_map = {}
        values = []

        for opt in options:
            if ":" in opt:
                letter, text = opt.split(":", 1)
                letter = letter.strip().lower()
                text = text.strip()
                option_map[letter] = text
                values.append(text.lower())

        # 1. User entered a letter (A/B/C/...)
        if user_input_clean in option_map:
            selected_text = option_map[user_input_clean].strip().lower()
            if selected_text == correct_answer_clean:
                return True, "✅ Correct!"
            else:
                return False, f"❌ Incorrect. The correct answer is: {correct_answer}"

        # 2. User entered an option text → check best match
        best_match_text = max(values, key=lambda v: SequenceMatcher(None, v, user_input_clean).ratio())
        match_ratio = SequenceMatcher(None, best_match_text, correct_answer_clean).ratio()

        if match_ratio > 0.85:
            return True, "✅ Correct (text match)!"
        else:
            # Wrong answer → Detailed explanation from LLM
            feedback_prompt = ChatPromptTemplate.from_template("""
            You are a Lean Production professor. A student gave a wrong answer to the following multiple choice question.

            Question: {question}
            Correct Answer: {correct_answer}
            User Answer: {user_answer}

            Please explain why the user's answer is incorrect and what the correct answer is. Use a helpful and didactic tone.
            """)
            chain = feedback_prompt | llm | StrOutputParser()
            try:
                explanation = chain.invoke({
                    "question": question,
                    "correct_answer": correct_answer,
                    "user_answer": user_answer
                }).strip()
                return False, f"❌ Incorrect.\n\n📘 {explanation}"
            except Exception as e:
                return False, f"❌ Incorrect. The correct answer is: {correct_answer}"

    # === FREITEXT-HANDLING (für Shopfloor & Detective) ===
    raw_output = _feedback_chain.invoke({
        "question": question,
        "correct_answer": correct_answer,
        "user_answer": user_answer
    })
    try:
        parsed = json.loads(raw_output)
        return parsed.get("is_correct", False), parsed.get("feedback", "No feedback provided.")
    except Exception:
        return False, "⚠️ Error parsing feedback. Please try again."



# === HINTS ===
_hint_prompt = ChatPromptTemplate.from_template("""
You are a Lean Production tutor. Provide a helpful hint for the following quiz question without revealing the answer.

Question: {question}
Correct Answer: {correct_answer}

Output only the hint.
""")
_hint_chain = _hint_prompt | llm | StrOutputParser()

def get_progressive_hint(question: str, correct_answer: str, level: int = 1) -> str:
    level = max(1, min(level, 3))
    hint_variants = {
        1: f"{question}",
        2: f"{question}\n\nGive a deeper hint related to typical Lean mistakes or tools.",
        3: f"{question}\n\nGive a very detailed hint that almost reveals the answer but not entirely."
    }
    return _hint_chain.invoke({
        "question": hint_variants[level],
        "correct_answer": correct_answer
    })


# === LOAD TOPIC SUMMARIES FROM FILE ===
def load_topic_summaries(summary_dir="topic_summaries"):
    summaries = {}
    if not os.path.exists(summary_dir):
        return summaries

    for file in os.listdir(summary_dir):
        if file.endswith(".pkl"):
            topic_name = file[:-4].replace("_", " ")
            with open(os.path.join(summary_dir, file), "rb") as f:
                summaries[topic_name] = pickle.load(f)
    return summaries

topic_summaries = load_topic_summaries()


# === SHORT SUMMARY ===
def get_topic_summary(topic):
    summary = topic_summaries.get(topic)
    if summary:
        return summary.get("short", "Lean focuses on improving flow and reducing waste.")
    return "This Lean topic is crucial for improving efficiency and reducing waste."


# === LLM-POWERED RESPONSE TO USER QUESTION BASED ON LONG SUMMARY ===
_topic_qa_prompt = ChatPromptTemplate.from_template("""
You are a Lean Production coach. Use the following topic summary to answer the user's question informatively and clearly.

Topic Summary:
{summary}

User Question:
{question}

Answer:
""")
_topic_qa_chain = _topic_qa_prompt | llm | StrOutputParser()

def get_topic_response(topic: str, user_question: str) -> str:
    context = topic_summaries.get(topic, {})
    long_text = context.get("long", "")

    if not long_text:
        return "🧠 I’m still gathering insights on that. But here's what I know so far: Lean focuses on eliminating waste and increasing value."

    try:
        answer = _topic_qa_chain.invoke({
            "summary": long_text,
            "question": user_question
        }).strip()
        return f"🧠 {answer}"
    except Exception as e:
        return f"❗ Sorry, I couldn't process that right now. ({str(e)})"


# === CONTEXT INTRO BASED ON DIFFICULTY ===
def get_question_context(question, difficulty):
    if difficulty == "easy":
        return "Let’s begin with the basics of Lean thinking. Remember: eliminating the 7 wastes is key."
    elif difficulty == "medium":
        return "We’re now applying Lean tools more directly to real-world problems. Focus on identifying the right approach."
    elif difficulty == "hard":
        return "This is where deeper insights matter. Think about the strategic impact of Lean in complex systems."
    return ""

# === LECTURE CONTEXT FOR PROFESSOR ===
from difflib import get_close_matches

def match_topic_name(query: str, available_topics: list[str]) -> str:
    matches = get_close_matches(query, available_topics, n=1, cutoff=0.6)
    return matches[0] if matches else query

def get_lecture_context(topic: str, difficulty: str = "medium") -> str:
    matched_topic = match_topic_name(topic, list(topic_summaries.keys()))
    summary_data = topic_summaries.get(matched_topic)

    if not summary_data:
        return "This topic is a key part of Lean Production, focusing on waste reduction and process efficiency."

    base_text = summary_data.get("long") or summary_data.get("short")
    if not base_text:
        return "This topic is essential in understanding Lean principles."

    # Kürzen je nach Schwierigkeitsgrad
    if difficulty == "easy":
        return base_text.strip()
    elif difficulty == "medium":
        return base_text[:400].rsplit(".", 1)[0] + "."
    elif difficulty == "hard":
        return base_text[:250].rsplit(".", 1)[0] + "."
    else:
        return base_text[:350] + "..."
