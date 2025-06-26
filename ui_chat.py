import streamlit as st
import subprocess
import base64
import io
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from gtts import gTTS

def generate_audio(text):
    tts = gTTS(text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()


def get_installed_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        return [line.split()[0] for line in lines[1:]]
    except Exception as e:
        st.warning(f"⚠️ Fehler beim Abrufen der Ollama-Modelle: {e}")
        return []

def filter_chat_models(model_names): 
    banned = ["mxbai-embed-large:latest", "nomic-embed-text:latest", "nomic-embed:latest", "nomic-embed-base:latest"]
    return [name for name in model_names if name not in banned]

def render_chat_ui():
    # --- STYLES ---
    st.markdown("""
        <style>
        .top-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 1.5rem;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }
        .top-row .left,
        .top-row .right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .user-bubble, .assistant-bubble {
            display: flex;
            margin: 0.3rem 0;
            max-width: 100%;
        }
        .user-bubble {
            justify-content: flex-end;
        }
        .assistant-bubble {
            justify-content: flex-start;
        }
        .bubble {
            padding: 0.7rem 1rem;
            border-radius: 1rem;
            background-color: #f1f1f1;
            max-width: 75%;
            word-wrap: break-word;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .user-bubble .bubble {
            background-color: #dcfce7;
        }
        .assistant-bubble .bubble {
            background-color: #f9fafb;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("💬 Dialogmodus")

    # --- SESSION STATE INIT ---
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("saved_chats", [])
    st.session_state.setdefault("user_input_buffer", None)

    # --- HEADER TOOLS ---
    st.markdown('<div class="top-row">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 2])
    with col1:
        if st.button(":new:", help="Neuer Chat"):
            st.session_state.messages = []
    with col2:
        if st.button(":floppy_disk:", help="Chat Speichern"):
            st.session_state.saved_chats.append(st.session_state.messages.copy())
            st.session_state.messages = []
    with col3:
        mic_button = mic_recorder("🎤 ", "⏹️", just_once=True, key="mic", format="wav")
    with col4:
        available_models = filter_chat_models(get_installed_ollama_models())
        if not available_models:
            st.error("❌ Keine Ollama-Modelle gefunden.")
            return
        st.selectbox("🔧 Modell wählen", available_models, key="model_select")
        st.session_state.model = st.session_state.get("model", available_models[0])

    # --- MODEL INIT ---
    llm = ChatOllama(model=st.session_state.model)

    # --- CHAT DISPLAY ---
    for idx, message in enumerate(st.session_state.messages):
        is_user = isinstance(message, HumanMessage)
        bubble_class = "user-bubble" if is_user else "assistant-bubble"

        st.markdown(f"""
            <div class="{bubble_class}">
                <div class="bubble">{message.content}</div>
            </div>
        """, unsafe_allow_html=True)

        if not is_user:
            if st.button(":speaking_head:", help="Antwort vorlesen", key=f"play_{idx}"):
                audio_bytes = generate_audio(message.content)
                st.audio(audio_bytes, format="audio/mp3")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- CHAT INPUT ---
    user_input = st.chat_input("Was möchtest du wissen?")

    if mic_button and "bytes" in mic_button:
        try:
            transcript = transcribe_audio(mic_button["bytes"])
            st.write("📜 Transkription:", transcript)
            st.session_state.user_input_buffer = transcript
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Fehler bei Transkription: {e}")

    if user_input:
        st.session_state.user_input_buffer = user_input
        st.rerun()

    if st.session_state.user_input_buffer:
        input_text = st.session_state.user_input_buffer
        st.session_state.user_input_buffer = None
        st.session_state.messages.append(HumanMessage(input_text))
        response = llm.invoke(st.session_state.messages).content
        st.session_state.messages.append(AIMessage(response))
        st.rerun()