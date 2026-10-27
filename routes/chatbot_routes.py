from flask import Blueprint, request, jsonify
from services.rag_service import rag_service
from services.llm_service import generate_chat_reply
import os

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("user_message", "")
        
        user_id = data.get("user_id", "unknown")
        section_id = data.get("section_id", "none")

        if not user_message.strip():
            return jsonify({"reply": "Please enter a question to ask."}), 400

        # Retrieve up to 12 chunks to ensure complete coverage
        context = rag_service.get_context(user_message, user_id=user_id, section_id=section_id, k=12)

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
            
        owner_id = request.form.get("owner_id", "unknown")
        section_id = request.form.get("section_id", "none")

        os.makedirs("training_data", exist_ok=True)
        file_path = f"training_data/{file.filename}"
        file.save(file_path)
        print(f"📥 Saved study material: {file.filename}")

        success = rag_service.add_document(file_path, file.filename, owner_id, section_id)

        if success:
            return jsonify({"message": "Study material indexed successfully with privacy controls!"}), 200
        else:
            return jsonify({"error": "Failed to extract text or index the document."}), 500

    except Exception as e:
        print(f"❌ Error in /api/train-chatbot: {e}")
        return jsonify({"error": str(e)}), 500
    
@chatbot_bp.route("/api/delete-document", methods=["POST"])
def delete_document():
    try:
        data = request.get_json()
        filename = data.get("filename")
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400
            
        success = rag_service.delete_document(filename)
        
        if success:
            return jsonify({"message": f"Document {filename} successfully purged from vector database."}), 200
        else:
            return jsonify({"error": "Failed to delete document from vector database."}), 500
            
    except Exception as e:
        print(f"❌ Error in /api/delete-document: {e}")
        return jsonify({"error": str(e)}), 500