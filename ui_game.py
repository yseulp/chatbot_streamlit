import streamlit as st
import time
from utils.load_quiz_data import load_quiz_catalog
from character_detective import run_detective_mode_streamlit
from character_manager import run_manager_mode_streamlit
from character_professor import run_professor_mode_streamlit
from character_colleague import run_colleague_mode_streamlit
 
# Einmalige Initialisierung von States
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False
st.session_state.setdefault("game_started", False)
if "selected_guide" not in st.session_state:
    st.session_state.selected_guide = None


# Typewriter-Effekt für Texte
def typewriter_effect(text, delay=0.05, size="####"):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(f"{size} {typed}▌")
        time.sleep(delay)
    placeholder.markdown(f"{size} {typed}")

# Streamlit-native Kartenanzeige
def render_clickable_card(icon, title, description, guide_name):
    card_key = f"card_clicked_{guide_name}"  # Richtiger Variablenname
    hover_text = f"Start with {title}"

    # Klick erkannt → Modus aktivieren
    if st.session_state.get(card_key):
        st.session_state.selected_guide = guide_name
        st.session_state.game_started = True
        st.session_state[card_key] = False  # Reset für später
        st.rerun()

    # HTML + CSS-Karte mit Hover-Effekt
    st.markdown(f"""
        <style>
        .card-container {{
            border: 2px solid #e0e0e0;
            border-radius: 16px;
            padding: 16px;
            margin: 10px;
            background-color: #ffffff;
            box-shadow: 10px 10px 12px rgba(0,0,0,0.1);
            height: 650px;
            width: 100%;
            text-align: center;
            cursor: pointer;
            transition: 0.3s ease;
            position: relative;
        }}

        .card-container:hover {{
            background-color: #f0f2f6;
            box-shadow: 4px 4px 16px rgba(255,0,0,0.3);
            transform: scale(1.02);
        }}

        .card-container:hover .hover-text {{
            opacity: 1;
        }}
        </style>

        <div class="card-container" onclick="document.getElementById('{guide_name}_form').submit();">
            <div style="font-size: 32px;">{icon}</div>
            <div style="font-weight: bold; font-size: 20px; margin: 10px 0;">{title}</div>
            <div style="font-size: 15px; color: #444;">{description}</div>
        </div>

        <form id="{guide_name}_form" method="post">
            <input type="hidden" name="clicked" value="{guide_name}">
            <input type="submit" style="display: none;">
        </form>
    """, unsafe_allow_html=True)

    # Workaround, falls du kein echtes Form-Handling brauchst:
    if st.button("👉", key=f"hidden_{guide_name}"):  # Unsichtbarer Backup-Button
        st.session_state.selected_guide = guide_name
        st.session_state.game_started = True
        st.rerun()


def render_game_ui():
    # Ein Topic muss schon gewählt sein
    topic = st.session_state.get("current_topic")

    if not topic:
        st.warning("Bitte zuerst ein Thema wählen")
        return 

    st.session_state.setdefault("game_started", False)
    st.session_state.setdefault("intro_shown", False)
    st.session_state.setdefault("selected_guide", None)

    try:
        quiz_catalog = load_quiz_catalog()
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Quizdaten: {e}")
        return

    # Startbildschirm (Rollenauswahl)
    if not st.session_state.game_started:
        if not st.session_state.intro_shown:
            typewriter_effect("Your learning journey through Lean Production begins 🚀", delay=0.02, size="#####")
            typewriter_effect("Choose your learning partner to begin your journey!", delay=0.02, size="#####")
            st.session_state.intro_shown = True
        else:
            st.markdown("### Your learning journey through Lean Production begins 🚀")
            st.markdown("##### Choose your learning partner to begin your journey!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_clickable_card(
                icon="🕵️",
                title="Lean Detective",
                description="""This mode keeps things simple and jumps straight into a randomly chosen case.<br>
                You’ll start with an easy-level question to warm up.<br>
                Answer correctly, and you'll level up to the next challenge — with a total of four questions per case.<br><br>
                At the end, you’ll get to choose: tackle the next case or wrap up your investigation. 🕵️‍♂️✨""",
                guide_name="detective"
            )
        with col2:
            render_clickable_card(
                icon="👷",
                title="Shopfloor Manager",
                description="""Start with a topic overview and ask questions in the chat.<br>
                Switch to quiz mode whenever you're ready — it starts easy and gets harder.<br>
                No fixed question count: you decide when to quiz or chat again.<br><br>
                Perfect for learning at your own pace, with full flexibility!""",
                guide_name="manager"
            )
        with col3:
            render_clickable_card(
                icon="👨‍🏫",
                title="Lean Professor",
                description="""Pick your level: Basic, Advanced, or Expert.<br>
                Each topic begins with an intro, then questions follow.<br>
                Wrong answer? Get detailed feedback.<br><br><br>
                Feeling bold? Exam Mode offers 15 mixed-difficulty questions with no previews. 🧪📚""",
                guide_name="professor"
            )
        with col4: 
            render_clickable_card(
                icon="👥", 
                title="Colleague",
                description=""" Start with a short intro and chat freely about the topic.
                Every second reply from your colleague brings a question — open-ended and tailored to your level. ✍️
                Answer, get feedback, and keep the conversation going until you're done or type quit.
                Learn naturally, one chat at a time! 💬✨""",
                guide_name="colleague"
            )

    # Sobald eine Karte ausgewählt wurde
    else:
        col_back, _ = st.columns([1, 9])
        with col_back:
            if st.button("⬅", help="Zurück zur Partnerwahl"):
                st.session_state.game_started = False
                st.session_state.selected_guide = None
                st.session_state.intro_shown = False
                st.rerun()

        guide = st.session_state.selected_guide
        if guide == "detective":
            run_detective_mode_streamlit()
        elif guide == "manager":
            run_manager_mode_streamlit()
        elif guide == "professor":
            run_professor_mode_streamlit()
        elif guide == "colleague":
            run_colleague_mode_streamlit()
