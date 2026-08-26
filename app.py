import os
import requests
import re
from flask import Flask, jsonify, request
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

app = Flask(__name__)

TRAINING_FOLDER = "training_data"
os.makedirs(TRAINING_FOLDER, exist_ok=True)

vector_db = None

def initialize_rag():
    global vector_db
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Check if database already exists
    if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
        print("⚡ Loading existing study materials from the database...")
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print("✅ RAG Pipeline Ready! (Skipped scanning)")
        return

    print("📚 Scanning 'training_data' for study materials...")
    loader = PyPDFDirectoryLoader(TRAINING_FOLDER)
    docs = loader.load()
    
    if not docs:
        print("⚠️ No PDFs found. Llama will use its general knowledge.")
        return

    print(f"📄 Loaded {len(docs)} pages/documents. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} text chunks. Generating embeddings in batches...")

    # Initialize Chroma DB with the first batch to create the database structure
    batch_size = 50  # Process 50 chunks at a time to prevent RAM overload
    vector_db = Chroma.from_documents(
        documents=chunks[:batch_size], 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    # Loop through the rest of the chunks in safe batches
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_db.add_documents(batch)
        print(f"progress: Processed chunk {i} out of {len(chunks)}...")

    print("✅ RAG Pipeline Ready! The massive textbook has been fully indexed.")

initialize_rag()

@app.route('/')
def home():
    return jsonify({"message": "NurseCompass ML Engine is Live! (Local RAG Mode)"})

# --- THIS IS THE ROUTE LARAVEL IS LOOKING FOR ---
@app.route('/api/train-chatbot', methods=['POST'])
def receive_file():
    try:
        if 'document' not in request.files:
            return jsonify({"error": "No document provided"}), 400
            
        file = request.files['document']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        file_path = os.path.join(TRAINING_FOLDER, file.filename)
        file.save(file_path)
        print(f"📥 Successfully received and saved: {file.filename}")
        
        # Re-run the RAG pipeline so Llama reads the new file instantly
        initialize_rag()
        
        return jsonify({"message": "File received and AI trained successfully!"}), 200
        
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return jsonify({"error": str(e)}), 500
# ------------------------------------------------

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('user_message')
        
        context = ""
        if vector_db is not None:
            search_results = vector_db.similarity_search(user_message, k=3)
            for doc in search_results:
                context += doc.page_content + "\n\n"

        if context:
            system_prompt = f"You are a Nursing Study Assistant. Answer the student's question strictly using the provided STUDY MATERIAL below. If the answer is not in the material, say 'I cannot find the answer in the provided study materials.' Format your response clearly: use concise paragraphs for definitions, and use bullet points for lists.\n\nSTUDY MATERIAL:\n{context}"
        else:
            system_prompt = "You are a Nursing Study Assistant. Format your response clearly: use concise paragraphs for definitions, and use bullet points for lists. Do not include introductory greetings or concluding remarks."

        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "llama3.2",
            "system": system_prompt,
            "prompt": user_message,
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        res_data = response.json()
        
        if response.status_code != 200:
            return jsonify({"reply": "Local AI Error occurred."}), 500

        clean_reply = res_data.get('response', '')
        clean_reply = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_reply)
        clean_reply = clean_reply.replace('\n', '<br>')
            
        return jsonify({"reply": clean_reply}), 200

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)