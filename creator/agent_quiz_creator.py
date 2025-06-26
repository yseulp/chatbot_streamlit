import os
import re
import json
import pickle

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

# === Einstellungen ===
vectorstore_base_dir = "/Users/simonhenschel/Desktop/UNI/M.Sc/SoSe 25/Masterthesis/Versuch 2/vectorstores_topic_summaries"
quiz_catalogs_dir = "/Users/simonhenschel/Desktop/UNI/M.Sc/SoSe 25/Masterthesis/Versuch 2/quiz_catalogs"
os.makedirs(quiz_catalogs_dir, exist_ok=True)

# === Modelle ===
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
llm = OllamaLLM(model="openhermes")

# === Hilfsfunktion zur Bereinigung ===
def sanitize_topic_name(topic):
    return re.sub(r'[^a-zA-Z0-9._-]', '_', topic).strip('_')

# === Quiz Prompt Template ===
quiz_prompt = PromptTemplate.from_template("""
You are a JSON-only quiz generator for the topic Lean Production.
You must strictly respond with ONLY a valid JSON array – no explanations, no comments.

Create a quiz with 10 questions about the topic: "{topic}".

Instructions:
- Include:
  - 3 easy multiple-choice questions (each with 3 options labeled A, B, C)
  - 3 medium open-ended questions (short free-text answers, no options)
  - 4 hard open-ended questions (more detailed free-text answers, no options)

Strict Requirements:
- Each easy question must include an "options" field with exactly 3 strings labeled "A: ...", "B: ...", "C: ..."
- The "correct_answer" must match one of the options exactly.
- For medium and hard questions, do NOT include an "options" field – only "question", "correct_answer", and "difficulty".
- The output MUST be a valid JSON array. Do not include any other text or formatting.

Difficulty Levels:
- Easy: surface-level recall of concepts, definitions, simple examples
- Medium: require reasoning or short explanation
- Hard: require detailed, multi-aspect understanding or applied problem-solving
""")

# === JSON-Reparatur (einfach) ===
def naive_json_repair(text):
    text = text.strip()
    text = re.sub(r"“|”", '"', text)
    text = re.sub(r"‘|’", "'", text)
    text = text.replace("'", '"')
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text

# === JSON-Parser ===
def try_parse_json(raw_text, raw_path=None):
    try:
        json_start = raw_text.find("[")
        json_end = raw_text.rfind("]") + 1
        if json_start == -1 or json_end == -1:
            print("❌ Kein JSON-Array erkannt.")
            return None

        json_str = raw_text[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("⚠️  Repariere fehlerhaftes JSON...")
            repaired = naive_json_repair(json_str)
            if raw_path:
                with open(raw_path.replace(".raw.txt", ".repaired.json"), "w") as f:
                    f.write(repaired)
            return json.loads(repaired)
    except Exception as e:
        print(f"❌ Fehler beim Parsen: {str(e)[:200]}")
        return None

# === Validierung ===
def validate_mcq(q):
    if q.get("difficulty") != "easy":
        return True
    options = q.get("options", [])
    correct = q.get("correct_answer", "")
    if not options or not isinstance(options, list):
        return False
    return correct in options

def validate_open_ended(q):
    if q.get("difficulty") in {"medium", "hard"}:
        return "options" not in q
    return True

# === Hauptprozess ===
print("📚 Starte Quiz-Generierung basierend auf Vektor-Matching...\n")
all_topic_names = [f for f in os.listdir(vectorstore_base_dir) if os.path.isdir(os.path.join(vectorstore_base_dir, f))]

for topic_folder in sorted(all_topic_names):
    topic = topic_folder.replace("_", " ")
    topic_safe = sanitize_topic_name(topic_folder)
    topic_vector_dir = os.path.join(vectorstore_base_dir, topic_folder)

    try:
        vs = Chroma(
            persist_directory=topic_vector_dir,
            embedding_function=embedding_model,
            collection_name=topic_safe
        )
        docs = vs.similarity_search("Generate quiz questions", k=4)
        if not docs:
            print(f"⚠️  Kein Inhalt im Vektorstore für Topic: {topic}")
            continue

        context = "\n".join(d.page_content for d in docs[:4])
        prompt = quiz_prompt.format(topic=topic)
        print(f"📤 Generiere Quiz für: {topic}...")

        response = llm.invoke(prompt)

        # 📝 Rohantwort speichern
        raw_path = os.path.join(quiz_catalogs_dir, f"{topic_safe}.raw.txt")
        with open(raw_path, "w") as f:
            f.write(response)

        quiz_data = try_parse_json(response, raw_path=raw_path)
        if not quiz_data or not isinstance(quiz_data, list):
            print(f"❌ Fehlerhafte JSON-Antwort für Topic: {topic}")
            continue

        # 🔍 Validierung
        invalid = [q for q in quiz_data if not validate_mcq(q) or not validate_open_ended(q)]
        if invalid:
            print(f"⚠️  Ungültige Fragen für Topic: {topic} – werden nicht gespeichert.")
            continue

        # ✅ Speichern
        quiz_path = os.path.join(quiz_catalogs_dir, f"{topic_safe}.pkl")
        with open(quiz_path, "wb") as f:
            pickle.dump(quiz_data, f)

        print(f"✅ Gespeichert unter: {quiz_path}\n")

    except Exception as e:
        print(f"❌ Fehler beim Laden von Vectorstore für {topic}: {str(e)[:200]}\n")

print("🎓 Quizgenerierung abgeschlossen.")
