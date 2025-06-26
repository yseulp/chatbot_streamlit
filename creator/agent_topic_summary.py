# Re-import necessary packages after code execution state reset
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Paths
pdf_path = "/Users/simonhenschel/Desktop/UNI/M.Sc/SoSe 25/Masterthesis/Versuch 2/content/Chapter 3_Waste_Reduction.pdf"
vectorstore_dir = "/Users/simonhenschel/Desktop/UNI/M.Sc/SoSe 25/Masterthesis/Versuch 2/vectorstores_topic_summaries"
os.makedirs(vectorstore_dir, exist_ok=True)

# Load and split PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Models
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
llm = OllamaLLM(model="openhermes")

# Topic extraction
topic_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a Lean Production expert. Based on the text, assign a concise topic category 
that reflects the main idea, principle, or method described. 
Return only the topic name.

Text:
{text}
"""
)
topic_chain = LLMChain(prompt=topic_prompt, llm=llm)

# Summary generation
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a Lean Production tutor. Write a clear and extensive summary (10-15 sentences) of the following content 
as if you're explaining it to a student. Focus on the key ideas and concepts.

Text:
{text}
"""
)
summary_chain = LLMChain(prompt=summary_prompt, llm=llm)

# Categorize and summarize
# --- Neue Struktur ---
from collections import defaultdict
topics = defaultdict(list)

# Zuweisung von Chunks zu Themen
for chunk in chunks:
    text = chunk.page_content.strip()
    topic = topic_chain.run({"text": text}).strip()
    topics[topic].append(chunk)

# Erstellung von Zusammenfassungen pro Topic
summaries = {}
for topic, topic_chunks in topics.items():
    combined_text = "\n".join([c.page_content for c in topic_chunks])
    summary_text = summary_chain.run({"text": combined_text}).strip()
    summaries[topic] = summary_text


# Save vectorstores per topic
stored_topics = []
for topic, docs in topics.items():
    topic_safe = topic.replace(" ", "_").replace("/", "_")
    store_path = os.path.join(vectorstore_dir, topic_safe)
    try:
        Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=store_path,
            collection_name=topic_safe
        )
        stored_topics.append(topic)
    except Exception as e:
        stored_topics.append(f"{topic} ❌ ({str(e)[:60]}...)")

(stored_topics, summaries)

summary_dir = "topic_summaries"
os.makedirs(summary_dir, exist_ok=True)

for topic, text in summaries.items():
    topic_safe = topic.replace(" ", "_").replace("/", "_")
    summary_data = {
        "short": text,
        "long": text
    }
    with open(os.path.join(summary_dir, f"{topic_safe}.pkl"), "wb") as f:
        pickle.dump(summary_data, f)

