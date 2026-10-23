from flask import Blueprint, request, jsonify
from services.rag_service import rag_service
from services.llm_service import generate_chat_reply

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("user_message", "")

        if not user_message.strip():
            return jsonify({"reply": "Please enter a question to ask."}), 400

        # Search for matching study material chunks in ChromaDB
        context = rag_service.get_context(user_message, k=6)

        # Generate 100% study-material-bound response
        reply = generate_chat_reply(user_message, context)

        return jsonify({"reply": reply}), 200
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return jsonify({"reply": f"Server Error: {str(e)}"}), 500

@chatbot_bp.route("/api/train-chatbot", methods=["POST"])
def train_chatbot():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No document uploaded."}), 400

        file = request.files["document"]
        if file.filename == "":
            return jsonify({"error": "No selected file."}), 400

        file_path = f"training_data/{file.filename}"
        file.save(file_path)
        print(f"📥 Saved study material: {file.filename}")

        # Re-index RAG vector store
        rag_service.initialize_rag()

        return jsonify({"message": "Study material indexed successfully!"}), 200
    except Exception as e:
        print(f"❌ Error in /api/train-chatbot: {e}")
        return jsonify({"error": str(e)}), 500