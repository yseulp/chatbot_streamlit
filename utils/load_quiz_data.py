# --- utils/load_quiz_data.py ---

import os
import pickle

def load_quiz_catalog(catalog_dir="quiz_catalogs"):
    quiz_catalog = {}
    if not os.path.exists(catalog_dir):
        raise FileNotFoundError(f"Quiz directory not found: {catalog_dir}")
    
    for file in os.listdir(catalog_dir):
        if file.endswith(".pkl"):
            topic_name = file[:-4].replace("_", " ")
            with open(os.path.join(catalog_dir, file), "rb") as f:
                quiz_catalog[topic_name] = pickle.load(f)
    
    return quiz_catalog
