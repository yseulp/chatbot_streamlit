### --- agent_quiz_creator.py --- ###

import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import json

# Absolute path to the PDF file in the content folder
pdf_path = "/Users/simonhenschel/Desktop/UNI/M.Sc/SoSe 25/Masterthesis/Versuch 2/content/Chapter 3_Waste_Reduction.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at: {pdf_path}")

# Load the PDF
doc_loader = PyPDFLoader(pdf_path)
documents = doc_loader.load()

# Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Setup embeddings and vector store
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_model)

# Add new data to the vector database
vectorstore.add_documents(texts)

# Load language model
model = OllamaLLM(model="llama3.2")

# Prompt for extracting chunk topics
chunk_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a tutor for Lean Production. Analyze the following text excerpt and return a concise topic or concept that best summarizes the content. 
    Return only the topic name.

    Text: {text}
    """
)
llm_chain_chunk_splitter = LLMChain(prompt=chunk_prompt_template, llm=model)

# Prompt for generating quiz questions from topic
text_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a quiz generator for the topic Lean Production. Create three multiple-choice questions for the following topic:
"{topic}"

Each question should have three answer choices (A, B, C). Return **only the JSON block** as shown below – **no introduction, comments, or explanations**:

[
    {{
        "question": "Question 1...",
        "options": ["A: ...", "B: ...", "C: ..."],
        "correct_answer": "A",
        "difficulty": "easy"
    }},
    ...
]
"""
)

llm_chain_text = LLMChain(prompt=text_prompt, llm=model)

# Initialize quiz catalog
quiz_catalog = {}

for doc in texts:
    text = doc.page_content
    topic = llm_chain_chunk_splitter.predict(text=text).strip()
    if not topic:
        continue

    try:
        quiz_json = llm_chain_text.predict(topic=topic).strip()
        # Try to extract only the JSON part
        json_start = quiz_json.find("[")
        json_end = quiz_json.rfind("]") + 1
        json_str = quiz_json[json_start:json_end]

        quiz_data = json.loads(json_str)
        quiz_catalog[topic] = quiz_data
    except json.JSONDecodeError as e:
        print(f"Error parsing quiz for topic '{topic}': {e}")
        print("Invalid JSON response:\n", quiz_json)
    except Exception as e:
        print(f"General error for topic '{topic}': {e}")

# Save quiz catalog
with open("quiz_catalog.pkl", "wb") as f:
    pickle.dump(quiz_catalog, f)

print("Quiz catalog successfully created and saved.")
