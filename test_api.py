import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the secret API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: No API key found. Check your .env file.")
    exit()

print("🔑 API Key Loaded successfully.")
print("🔍 Asking Google what models you have access to...\n")

genai.configure(api_key=API_KEY)

try:
    # Ask Google for a list of all models allowed for this key
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"✅ You can use: {m.name}")
    
    if not available_models:
        print("⚠️ Google says your API key has NO text models available!")
        
except Exception as e:
    print(f"❌ API Error: {e}")