import os
import requests
import re
from flask import Flask, jsonify, request
from langchain_community.document_loaders import PyMuPDFLoader
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
        vector_db = Chroma(
            persist_directory="./chroma_db", embedding_function=embeddings
        )
        print("✅ RAG Pipeline Ready! (Skipped scanning)")
        return

    print("📚 Scanning 'training_data' for study materials using PyMuPDF...")

    # 1. NEW EXTRACTION: Loop through directory and use PyMuPDF to extract tables cleanly
    docs = []
    for filename in os.listdir(TRAINING_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(TRAINING_FOLDER, filename)
            loader = PyMuPDFLoader(file_path)
            docs.extend(loader.load())

    if not docs:
        print("⚠️ No PDFs found. Llama will use its general knowledge.")
        return

    # 2. NEW CHUNKING: Reduced chunk_size to 300 to prevent context collision
    print(f"📄 Loaded {len(docs)} pages/documents. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} text chunks. Generating embeddings in batches...")

    # Initialize Chroma DB with the first batch to create the database structure
    batch_size = 50  # Process 50 chunks at a time to prevent RAM overload
    vector_db = Chroma.from_documents(
        documents=chunks[:batch_size],
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

    # Loop through the rest of the chunks in safe batches
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_db.add_documents(batch)
        print(f"progress: Processed chunk {i} out of {len(chunks)}...")

    print("✅ RAG Pipeline Ready! The massive textbook has been fully indexed.")


initialize_rag()


@app.route("/")
def home():
    return jsonify({"message": "NurseCompass ML Engine is Live! (Local RAG Mode)"})


# --- THIS IS THE ROUTE LARAVEL IS LOOKING FOR ---
@app.route("/api/train-chatbot", methods=["POST"])
def receive_file():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document provided"}), 400

        file = request.files["document"]

        if file.filename == "":
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


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("user_message")

        context = ""
        # Increase k from 3 to 8 to give the LLM a wider reading window
        if vector_db is not None:
            search_results = vector_db.similarity_search(user_message, k=8)
            for doc in search_results:
                context += doc.page_content + "\n\n"

        # 3. HYBRID PROMPT: Prioritize context, but allow general knowledge as a fallback
        if context:
            system_point = (
                "You are a Nursing Study Assistant for Universidad de Manila.\n"
                "First, try to answer the student's question using ONLY the provided Context.\n"
                "If the Context does not contain the answer, you may use your general medical and nursing knowledge to answer the question, but briefly mention that this information is outside of the uploaded reviewer.\n\n"
                f"Context:\n{context}"
            )
        else:
            system_point = "You are a Nursing Study Assistant for Universidad de Manila. Please ask a question about nursing."

        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "llama3.2",
            "system": system_point,
            "prompt": user_message,
            "stream": False,
            "options": {
                "temperature": 0.3  # Raised from 0.0 to allow a little creativity
            },
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        res_data = response.json()

        if response.status_code != 200:
            return jsonify({"reply": "Local AI Error occurred."}), 500

        clean_reply = res_data.get("response", "")
        clean_reply = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_reply)
        clean_reply = clean_reply.replace("\n", "<br>")

        return jsonify({"reply": clean_reply}), 200

    except Exception as e:
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500


# --- ENSURE THIS IS OUTSIDE AND ALIGNED TO THE LEFT ---
@app.route("/api/predict-readiness", methods=["POST"])
def predict_readiness():
    try:
        data = request.get_json()
        score = data.get("score", 0)
        total = data.get("total_questions", 1)
        category_scores = data.get("category_scores", {})
        detailed_answers = data.get("detailed_answers", [])

        percentage = (score / total) * 100 if total > 0 else 0
        prediction = "Workforce Ready" if percentage >= 75 else "Needs Improvement"

        # 1. GUARANTEE all 4 categories exist to prevent ranking crashes
        core_categories = [
            "Academic Performance",
            "Self-Efficacy",
            "Initiative",
            "Motivation",
        ]
        cat_sorted = []

        for cat in core_categories:
            stats = category_scores.get(cat, {"score": 0, "total": 0})
            cat_score = stats.get("score", 0)
            cat_total = stats.get("total", 0)
            cat_pct = round((cat_score / cat_total) * 100) if cat_total > 0 else 0
            cat_sorted.append({"name": cat, "percentage": cat_pct})

        cat_sorted = sorted(cat_sorted, key=lambda x: x["percentage"], reverse=True)

        top_category = cat_sorted[0]["name"]
        top_pct = cat_sorted[0]["percentage"]
        second_category = cat_sorted[1]["name"]
        second_pct = cat_sorted[1]["percentage"]
        lowest_category = cat_sorted[-1]["name"]
        lowest_pct = cat_sorted[-1]["percentage"]

        category_percentages = {item["name"]: item["percentage"] for item in cat_sorted}

        # 2. Format itemized answers for Llama
        incorrect_items = []
        correct_items = []

        for item in detailed_answers:
            q_text = item.get("question")
            section = item.get("section")
            is_correct = item.get("is_correct")
            student_ans = item.get("student_answer")
            correct_ans = item.get("correct_answer")

            if is_correct:
                correct_items.append(f"- [{section}] {q_text}")
            else:
                incorrect_items.append(
                    f"- [{section}] {q_text} | You chose: '{student_ans}' | Correct: '{correct_ans}'"
                )

        incorrect_text = "\n".join(incorrect_items[:10]) if incorrect_items else "None"
        correct_text = "\n".join(correct_items[:10]) if correct_items else "None"

        prompt = (
            f"Evaluate this nursing student who scored {score}/{total} ({percentage:.0f}%).\n\n"
            f"MISTAKES:\n{incorrect_text}\n\n"
            f"CORRECT:\n{correct_text}\n\n"
            f"Write a direct, 3-sentence feedback paragraph mentioning specific clinical topics they missed and what they got right. DO NOT use formal letter openings or bracketed placeholders."
        )

        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": "llama3.2", "prompt": prompt, "stream": False}

        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}, timeout=120
        )
        note = response.json().get(
            "response", "Review core nursing principles and clinical protocols."
        )

        return (
            jsonify(
                {
                    "prediction": prediction,
                    "personalized_note": note,
                    "category_percentages": category_percentages,
                    "top_category": top_category,
                    "top_pct": top_pct,
                    "second_category": second_category,
                    "second_pct": second_pct,
                    "lowest_category": lowest_category,
                    "lowest_pct": lowest_pct,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"❌ Error generating prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
