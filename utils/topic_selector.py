from utils.load_quiz_data import load_quiz_catalog

def get_available_topics():
    quiz_catalog = load_quiz_catalog()
    return [t for t, qs in quiz_catalog.items() if qs]

def choose_topic_cli(available_topics, prompt_text="Choose a topic number: "):
    if not available_topics:
        print("⚠️ No topics available.")
        return None

    print("📚 Available Topics:")
    for i, topic in enumerate(available_topics, 1):
        print(f"{i}. {topic}")

    while True:
        selected = input(f"\n{prompt_text}").strip()
        if selected.isdigit() and 1 <= int(selected) <= len(available_topics):
            return available_topics[int(selected) - 1]
        print("❌ Invalid input. Try again.")
