### --- agent_quiz_executor.py --- ###

import pickle
import random
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load quiz catalog
with open("quiz_catalog.pkl", "rb") as f:
    quiz_catalog = pickle.load(f)

# Setup model
model = OllamaLLM(model="llama3.2")

# Select topic
print("Available topics:")
topics = list(quiz_catalog.keys())
for i, topic in enumerate(topics, 1):
    print(f"{i}. {topic}")
topic_choice = input("\nSelect a topic by number: ").strip()
try:
    topic = topics[int(topic_choice) - 1]
except (IndexError, ValueError):
    print("Invalid selection. Defaulting to the first topic.")
    topic = topics[0]

# Select difficulty
difficulty_level = input("Please choose a difficulty level (easy, medium, hard): ").strip().lower()
if difficulty_level not in ["easy", "medium", "hard"]:
    print("Invalid difficulty level. Defaulting to 'medium'.")
    difficulty_level = "medium"

# Retrieve questions for selected topic and difficulty
topic_questions = quiz_catalog.get(topic, [])
if isinstance(topic_questions, dict):
    topic_questions = [topic_questions]  # wrap in list if single dict
questions = [q for q in topic_questions if isinstance(q, dict) and q.get("difficulty", "medium") == difficulty_level]

# Shuffle questions for randomness
random.shuffle(questions)

# Setup feedback evaluation prompt
feedback_prompt = ChatPromptTemplate.from_template("""
You are a Lean Production expert. Evaluate the user's answer for the quiz question below.

Question: {question}
Correct Answer: {correct_answer}
User Answer: {user_answer}

Provide semantic feedback:
1. Was the user's answer semantically correct?
2. If incorrect, explain briefly why and provide the correct answer.
3. Avoid commenting on formatting, punctuation, or casing.

Output only the feedback message.
""")
feedback_chain = LLMChain(prompt=feedback_prompt, llm=model)

print("\nInteractive quiz session started!\n")

# Run the quiz
for idx, q in enumerate(questions, 1):
    print(f"**Question {idx}:**\n{q['question']}")
    for opt in q.get("options", []):
        print(opt)
    answer = input("\nPlease respond with your answer (e.g., A, B, or C): ").strip()

    feedback = feedback_chain.run({
        "question": q['question'],
        "correct_answer": q.get("correct_answer", ""),
        "user_answer": answer
    })
    print(f"\nFeedback: {feedback}\n")

    next_action = input("Enter 'q' to quit, or press Enter for the next question: ").strip().lower()
    if next_action == 'q':
        print("\nQuiz ended. Thank you for participating!")
        break
