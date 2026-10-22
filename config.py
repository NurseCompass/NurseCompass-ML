import os

# Safe environment loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Settings & Constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

TRAINING_FOLDER = "training_data"
CHROMA_DB_DIR = "./chroma_db"