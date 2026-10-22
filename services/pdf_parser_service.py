import os
import re
from langchain_community.document_loaders import PDFPlumberLoader
import config

class PDFParserService:
    @staticmethod
    def extract_questions_from_pdf(file_storage):
        temp_path = os.path.join(config.TRAINING_FOLDER, "temp_assessment.pdf")
        file_storage.save(temp_path)

        try:
            loader = PDFPlumberLoader(temp_path)
            docs = loader.load()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        full_text = "\n".join([doc.page_content for doc in docs])

        full_text = re.sub(
            r"\s*(?:Correct\s*answer|Answer|Ans)[\s\.\:\-]*([A-E])(?:\b|\.|\-)",
            r"\nCorrect answer: \1. ",
            full_text,
            flags=re.IGNORECASE,
        )

        full_text = re.sub(
            r"([a-z\.\:\;\"\'])\s+(?=(?:Q\s*)?([1-9]|[1-4][0-9])[\.\)\:\-]\s+[A-Z])",
            r"\1\n",
            full_text,
        )

        full_text = re.sub(
            r"([A-Za-z\.\:\;])\s+(?=SECTION\s+\d+)",
            r"\1\n",
            full_text,
            flags=re.IGNORECASE,
        )

        all_questions = []
        current_section = "General Knowledge"
        current_question_section = current_section

        lines = full_text.split("\n")
        current_question = None
        current_options = []
        current_state = None

        for line in lines:
            line_str = line.strip()
            upper_line = line_str.upper()

            if not line_str or "CHOOSE ONE BEST ANSWER" in upper_line:
                continue

            if "SECTION" in upper_line:
                if "ACADEMIC PERFORMANCE" in upper_line:
                    current_section = "Academic Performance"
                    continue
                elif "SELF-EFFICACY" in upper_line:
                    current_section = "Self-Efficacy"
                    continue
                elif "INITIATIVE" in upper_line:
                    current_section = "Initiative"
                    continue
                elif "MOTIVATION" in upper_line:
                    current_section = "Motivation"
                    continue

            q_match = re.search(
                r"^[\s\W]*?(?:Q|Question)?\s*?(\d+)[\.\)\:\-]\s*(.*)",
                line_str,
                re.IGNORECASE,
            )
            if q_match:
                if current_question and current_options:
                    if not any(opt.get("is_correct") for opt in current_options):
                        current_options[0]["is_correct"] = True

                    all_questions.append(
                        {
                            "section": current_question_section,
                            "question_text": current_question,
                            "options": current_options,
                        }
                    )

                current_question = q_match.group(2).strip()
                current_question_section = current_section
                current_options = []
                current_state = "QUESTION"
                continue

            opt_match = re.search(
                r"^[\s\W]*?([A-E])[\.\)\:\-]\s*(.*)", line_str, re.IGNORECASE
            )
            if opt_match and current_question:
                if len(current_options) >= 4:
                    continue

                opt_text = opt_match.group(2).strip()
                current_options.append({"option_text": opt_text, "is_correct": False})
                current_state = "OPTION"
                continue

            ans_match = re.search(
                r"^[\s\W]*?(?:Correct\s*answer|Answer|Ans)[\s\.\:\-]*([A-E])",
                line_str,
                re.IGNORECASE,
            )
            if ans_match and current_options:
                correct_letter = ans_match.group(1).upper()
                for idx, opt in enumerate(current_options):
                    opt["is_correct"] = chr(65 + idx) == correct_letter

                current_state = "EXPLANATION"
                continue

            if current_state == "QUESTION" and current_question:
                current_question += " " + line_str
            elif current_state == "OPTION" and current_options:
                current_options[-1]["option_text"] += " " + line_str
            elif current_state == "EXPLANATION":
                pass

        if current_question and current_options:
            if not any(opt.get("is_correct") for opt in current_options):
                current_options[0]["is_correct"] = True

            all_questions.append(
                {
                    "section": current_question_section,
                    "question_text": current_question,
                    "options": current_options,
                }
            )

        return all_questions