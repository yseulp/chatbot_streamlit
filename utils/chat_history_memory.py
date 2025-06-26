# --- chat_history_memory.py ---

import os
import uuid
import shutil
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings

# === Konfiguration ===
PERSIST_DIR = "./chat_history_vectorstore"
COLLECTION_NAME = "chat_history"
os.makedirs(PERSIST_DIR, exist_ok=True)

# === Initialisiere Embedding-Modell ===
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# === Initialisiere Vektorstore (global)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)

# === Nachricht speichern (mit optionaler Session-ID) ===
def save_message(role: str, content: str, session_id: str = None):
    """
    Speichert eine Nachricht im Vektorstore.
    Metadaten enthalten: Rolle, UUID, optional Session-ID und role_session für kombinierte Filterung.
    """
    metadata = {
        "role": role,
        "id": str(uuid.uuid4())
    }
    if session_id:
        metadata["session_id"] = session_id
        metadata["role_session"] = f"{role}_{session_id}"

    doc = Document(
        page_content=content,
        metadata=metadata
    )
    vectorstore.add_documents([doc])

# === Ähnlichen Kontext abrufen (nach Rolle + optional Session-ID) ===
def retrieve_similar_history(query: str, k: int = 3, role: str = None, session_id: str = None):
    """
    Ruft ähnliche frühere Nachrichten ab.
    Optional kann nach Rolle und/oder Session-ID gefiltert werden.
    Verwendet role_session für kombiniertes Chroma-kompatibles Filterverhalten.
    """
    try:
        filter_dict = None
        if role and session_id:
            filter_dict = {"role_session": f"{role}_{session_id}"}
        elif role:
            filter_dict = {"role": role}
        elif session_id:
            filter_dict = {"session_id": session_id}

        results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        return [doc.page_content for doc in results]

    except Exception as e:
        print(f"❌ Fehler beim Abruf des Kontexts: {e}")
        return []

# === Kompletten Verlauf löschen ===
def reset_history():
    """Löscht alle gespeicherten Chatnachrichten (globaler Reset)."""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)
    print("🧹 Globaler Chatverlauf gelöscht.")

# === Verlauf für bestimmte Session-ID löschen ===
def reset_session_history(session_id: str):
    """
    Entfernt Nachrichten mit gegebener Session-ID.
    ⚠️ Chroma unterstützt aktuell kein gezieltes Löschen nach Filter.
    """
    print("⚠️ Session-spezifisches Reset noch nicht implementiert (limitiert durch Chroma).")

# === Rollenübersicht ===
# character_detective  -> role="detective"
# character_colleague  -> role="colleague"
# character_manager    -> role="manager"
# character_professor  -> role="professor"
