import uuid
from base64 import b64decode
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.messages import SystemMessage, HumanMessage
from PIL import Image
import io


### --- FUNKTION ZUM ANZEIGEN VON BASE64-BILDERN --- ###
def display_base64_image(base64_string):
    """Hilfsfunktion, um ein Base64-Bild anzuzeigen."""
    image_data = b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.show()


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return {"prompt": prompt_template, "prompt_content": prompt_content}


### --- RETRIEVER SETUP --- ###

# Embedding-Modell lokal via Ollama (ersetzen OpenAI durch Ollama)
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Vektorstore für Child-Dokumente
vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=embedding_model
)

# Speicher für Parent-Dokumente
store = InMemoryStore()
id_key = "doc_id"

# Multi-Vector Retriever initialisieren
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Lokales Sprachmodell für die Beantwortung von Fragen
model = OllamaLLM(model="llama3.2")


### --- KETTE AUFBAUEN --- ###

# Verwenden von RunnableLambda für den Modellaufruf, um das Hashbarkeit-Problem zu lösen
chain = (
    RunnableSequence(
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        },
        {
            "prompt": RunnableLambda(build_prompt),  # Anpassen der Übergabe
            "model": RunnableLambda(lambda kwargs: model.invoke(kwargs["prompt"])),  # Den 'prompt' explizit weitergeben
        }
    )
    | RunnablePassthrough()
)

# Beispielanfrage
response = chain.invoke(
    {"question": "What is the method?", "context": {}}
)

print(response)

response = chain.invoke(
    {"question": "What is waste?", "context": {}}
)

print("Response:", response['response'])

print("\n\nContext:")
for text in response['context']['texts']:
    print(text.text)
    print("Page number: ", text.metadata.page_number)
    print("\n" + "-"*50 + "\n")
for image in response['context']['images']:
    # Funktion zum Anzeigen von Base64-Bildern
    display_base64_image(image)
