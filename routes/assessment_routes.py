from flask import Blueprint, jsonify, request
from services.pdf_parser_service import PDFParserService

assessment_bp = Blueprint("assessment", __name__)

@assessment_bp.route("/api/extract-questions", methods=["POST"])
def extract_questions():
    try:
        if "document" not in request.files:
            return jsonify({"error": "No PDF provided"}), 400

        file = request.files["document"]
        all_questions = PDFParserService.extract_questions_from_pdf(file)

        if not all_questions:
            return (
                jsonify({"error": "No questions could be parsed. Check PDF format."}),
                400,
            )

        print(
            f"⚡ Ultimate Pre-Processed Parse Complete! Total questions found: {len(all_questions)}"
        )
        return jsonify({"questions": all_questions}), 200

    except Exception as e:
        print(f"❌ Error parsing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500