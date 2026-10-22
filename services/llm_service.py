import requests
import json
import re
import config

def call_gemini(prompt):
    if not config.GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={config.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
    return None

def call_ollama(prompt, system_prompt=""):
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": config.LLM_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=300)
        if res.status_code == 200:
            return res.json().get("response", "")
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")
    return None

def generate_chat_reply(user_message, context=""):
    # 1. STRICT GUARDRAIL: If no relevant context in study materials, refuse immediately
    if not context or not context.strip():
        return "I am sorry, but no uploaded study materials were found regarding this topic. Please check your uploaded course reviewers or consult your clinical instructor."

    # 2. Strict Context-Only Prompt
    prompt = (
        f"You are NurseCompass AI Study Assistant for Universidad de Manila (UDM) College of Nursing.\n\n"
        f"STRICT INSTRUCTIONS:\n"
        f"1. Answer the student's question STRICTLY AND EXCLUSIVELY using the Study Material Context provided below.\n"
        f"2. DO NOT use external medical knowledge, assumptions, or unprovided facts.\n"
        f"3. If the provided context does NOT contain the complete answer, respond EXACTLY with:\n"
        f"'I am sorry, but that information is not available in the uploaded study materials. Please review your uploaded course reviewers or consult your clinical instructor.'\n\n"
        f"STUDY MATERIAL CONTEXT:\n{context}\n\n"
        f"STUDENT QUESTION:\n{user_message}"
    )

    reply = call_gemini(prompt)
    if not reply:
        reply = call_ollama(prompt)

    if not reply or not reply.strip():
        reply = "I am sorry, but that information is not available in the uploaded study materials. Please review your uploaded course reviewers or consult your clinical instructor."

    # HTML formatting for Blade template compatibility
    reply = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", reply)
    reply = reply.replace("\n", "<br>")
    return reply

def evaluate_student_readiness(score, total, category_scores, detailed_answers, year_level="4th Year"):
    percentage = (score / total) * 100 if total > 0 else 0
    prediction = "WORKFORCE READY" if percentage >= 75 else "NOT YET READY"

    incorrect_items = []
    correct_items = []

    for item in detailed_answers:
        q_text = item.get("question", "")
        section = item.get("section", "")
        is_correct = item.get("is_correct", False)
        student_ans = item.get("student_answer", "")
        correct_ans = item.get("correct_answer", "")

        if is_correct:
            correct_items.append(f"- [{section}] Question: {q_text}")
        else:
            incorrect_items.append(
                f"- [{section}] Question: {q_text} | Student Chose: '{student_ans}' | Correct Action: '{correct_ans}'"
            )

    incorrect_text = "\n".join(incorrect_items[:10]) if incorrect_items else "No major clinical errors made."
    correct_text = "\n".join(correct_items[:5]) if correct_items else "None recorded."

    prompt = (
        f"You are a Senior Clinical Nursing Evaluator at Universidad de Manila (UDM) College of Nursing.\n"
        f"Evaluate this {year_level} BSN Nursing student who completed their Comprehensive Clinical Assessment.\n\n"
        f"STUDENT METRICS:\n"
        f"- Overall Score: {score}/{total} ({percentage:.0f}%)\n"
        f"- Clinical Readiness Status: {prediction}\n\n"
        f"SPECIFIC CLINICAL MISTAKES & MISSED QUESTIONS:\n{incorrect_text}\n\n"
        f"CORRECTLY HANDLED CLINICAL SCENARIOS:\n{correct_text}\n\n"
        f"Generate EXACTLY TWO distinct output sections:\n\n"
        f"PREDICTION_NOTE: Write a 2 to 3 sentence professional evaluation highlighting their overall readiness for clinical nursing practice and their core domain strengths in ward care.\n\n"
        f"INTERPRETATION_NOTE: Write a 3 to 4 sentence clinical diagnostic specifically explaining WHERE THE STUDENT LACKS based on the exact missed nursing topics above (e.g., emergency triage, IV monitoring, documentation, or psychiatric safety). Explain WHY these unaddressed gaps impact patient safety during hospital duties, and recommend targeted clinical remediation.\n\n"
        f"Format your response EXACTLY as follows:\n"
        f"PREDICTION_NOTE: <your 2-3 sentences>\n"
        f"INTERPRETATION_NOTE: <your 3-4 sentences>"
    )

    raw_response = call_gemini(prompt)
    if not raw_response:
        raw_response = call_ollama(prompt)

    if raw_response:
        pred_match = re.search(r"PREDICTION_NOTE:\s*(.*?)(?=INTERPRETATION_NOTE:|$)", raw_response, re.DOTALL)
        interp_match = re.search(r"INTERPRETATION_NOTE:\s*(.*)", raw_response, re.DOTALL)

        pred_note = pred_match.group(1).strip() if pred_match else "Demonstrated baseline theoretical knowledge across core nursing domains."
        interp_note = interp_match.group(1).strip() if interp_match else "Requires focused review of missed clinical scenarios and safety protocols."

        pred_note = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", pred_note).replace("\n", " ")
        interp_note = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", interp_note).replace("\n", " ")

        return {
            "prediction": prediction,
            "personalized_note": pred_note,
            "assessment_interpretation": interp_note
        }

    return {
        "prediction": prediction,
        "personalized_note": f"Scored {score}/{total} ({percentage:.0f}%). Demonstrated baseline theoretical understanding in core BSN competencies.",
        "assessment_interpretation": f"Performance indicates a need for targeted skill building in missed clinical scenarios, emergency protocols, and patient safety procedures."
    }