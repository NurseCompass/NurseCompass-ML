import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

print("--- STARTING ULTIMATE BRUTE-FORCE TEST ---")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
response = requests.get(url)

if response.status_code != 200:
    print(f"💀 API Key Error: {response.text}")
    exit()

models = response.json().get('models', [])
working_model = None

print(f"🔍 Found {len(models)} total models. Testing them one by one...\n")

for m in models:
    methods = m.get('supportedGenerationMethods', [])
    name = m.get('name', '')
    
    # Only test text-generation models
    if 'generateContent' in methods and 'vision' not in name:
        print(f"🔄 Testing model: {name}...")
        
        test_url = f"https://generativelanguage.googleapis.com/v1beta/{name}:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": "If you can read this, reply with exactly one word: Banana."}]}]}
        headers = {"Content-Type": "application/json"}
        
        try:
            test_res = requests.post(test_url, json=payload, headers=headers)
            
            if test_res.status_code == 200:
                reply = test_res.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"   🎉 SUCCESS! Google replied: {reply}")
                working_model = name
                break # Stop testing! We found the golden ticket.
            else:
                print(f"   ❌ Rejected (Status {test_res.status_code})")
        except Exception as e:
            print(f"   ❌ Crashed: {e}")

print("\n" + "="*50)
if working_model:
    print(f"🚀 YOUR FIX: Open app.py and replace the model name with EXACTLY: '{working_model}'")
else:
    print("💀 ALL MODELS REJECTED. Google has completely locked text generation on this specific API key. You must generate a new API key from Google AI Studio.")
print("="*50 + "\n")