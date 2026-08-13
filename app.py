import os
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

TRAINING_FOLDER = "training_data"
os.makedirs(TRAINING_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({"message": "NurseCompass ML Engine is Live!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('user_message')
        
        print(f"Student asked: {user_message}")
        print("🧠 Calling Gemini API via REST...")
        
        # Use the verified working model endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": f"Provide concise nursing interventions for: {user_message}"}]
            }]
        }
        
        headers = {"Content-Type": "application/json"}
        
        # 60 seconds timeout to prevent dropping connections
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        res_data = response.json()
        
        if response.status_code != 200:
            print(f"❌ Google Error: {res_data}")
            return jsonify({"reply": "Google API Error occurred."}), 500

        # Extract text safely
        # Extract the raw text safely
        raw_text = res_data['candidates'][0]['content']['parts'][0]['text']
        
        # --- AGGRESSIVE CLEANING FILTER ---
        # Look for the last occurrence of common bullet formatting or section markers
        # because the model always puts its final clean output at the very end.
        if "* *" in raw_text:
            # Find all parts and grab the last chunk which contains the actual answer
            parts = raw_text.split("* *")
            clean_reply = "* *" + parts[-1]
        elif "* **" in raw_text:
            parts = raw_text.split("* **")
            clean_reply = "* **" + parts[-1]
        else:
            clean_reply = raw_text
            
        print("✅ Reply received successfully!")
        return jsonify({"reply": clean_reply}), 200

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)