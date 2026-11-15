from flask import Blueprint, request, jsonify
from services.llm_service import evaluate_student_readiness

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/api/predict-readiness", methods=["POST"])
def predict_readiness():
    try:
        data = request.get_json()
        score = data.get("score", 0)
        total = data.get("total_questions", 1)
        category_scores = data.get("category_scores", {})
        detailed_answers = data.get("detailed_answers", [])
        year_level = data.get("year_level", "4th Year")

        # --- TERMINAL INDICATORS START HERE ---
        print("\n" + "=" * 60, flush=True)
        print("[NURSECOMPASS AI LOG] New Assessment Evaluation Triggered!", flush=True)
        print(f"[NURSECOMPASS AI LOG] Student Level: {year_level}", flush=True)
        print(f"[NURSECOMPASS AI LOG] Raw Score Count: {score}/{total}", flush=True)

        # This line proves the system is receiving the specific answers
        print(
            f"[NURSECOMPASS AI LOG] Specific Answers Captured: {len(detailed_answers)} items",
            flush=True,
        )

        # This prints the actual text of the first question to your terminal so you can see it with your own eyes
        if len(detailed_answers) > 0:
            print(
                f"[NURSECOMPASS AI LOG] Sample Data Sent to AI: {detailed_answers[0]}",
                flush=True,
            )

        print(
            "[NURSECOMPASS AI LOG] Routing this data to the LLM Service...", flush=True
        )
        print("=" * 60 + "\n", flush=True)
        # --- TERMINAL INDICATORS END HERE ---

        result = evaluate_student_readiness(
            score, total, category_scores, detailed_answers, year_level
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
