from flask import Blueprint, request, jsonify
from services.llm_service import evaluate_student_readiness

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route("/api/predict-readiness", methods=["POST"])
def predict_readiness():
    try:
        data = request.get_json()
        score = data.get("score", 0)
        total = data.get("total_questions", 1)
        category_scores = data.get("category_scores", {})
        detailed_answers = data.get("detailed_answers", [])
        year_level = data.get("year_level", "4th Year")

        result = evaluate_student_readiness(score, total, category_scores, detailed_answers, year_level)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500