import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import OllamaEmbeddings  # Lokale Embeddings

def setup_retriever(texts, text_summaries, tables, table_summaries, images, image_summaries):
    # Lokales Embedding-Modell mit Ollama
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    # Vectorstore für Child Chunks
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embedding_model)

    # Storage für Parent Docs
    store = InMemoryStore()
    id_key = "doc_id"

    # Multi-Vector Retriever initialisieren
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )

    # --- Texte hinzufügen ---
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # --- Tabellen hinzufügen ---
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # --- Bildbeschreibungen hinzufügen ---
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))

    return retriever
