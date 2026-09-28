import json
import re

from gemini_client import generate_with_gemini


def clean_json_block(text: str) -> str:
    """
    Removes Markdown ```json ... ``` code fences
    and trims surrounding whitespace.
    """
    cleaned = re.sub(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        r"\1",
        text
    )

    return cleaned.strip()


def generate_quiz(text: str) -> list:
    """
    Generates 3 multiple-choice questions from a passage or topic.

    Returns:
        A Python list containing question objects.
    """

    if not text or not text.strip():
        return [
            {
                "error": "Please enter some text or a topic to generate a quiz."
            }
        ]

    prompt = f"""
You are EduGenie, an AI quiz generator for college students.

Create exactly 3 multiple-choice questions from the following
passage or topic.

Each question MUST contain:

- "question": The question text
- "options": Exactly 4 answer choices
- "answer": The correct answer

The "answer" MUST exactly match one of the four options.

Return ONLY valid JSON.
Do not include Markdown.
Do not include ```json.
Do not include explanations outside the JSON.

Required format:

[
  {{
    "question": "What is ...?",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A"
  }}
]

Passage or Topic:
{text.strip()}
"""

    try:

        # Use the centralized Gemini client.
        # json_mode=True asks Gemini for JSON output.
        quiz_text = generate_with_gemini(
            prompt,
            json_mode=True
        )

        if not quiz_text:
            return [
                {
                    "error": "Gemini returned an empty quiz."
                }
            ]

        cleaned_text = clean_json_block(quiz_text)

        # Parse JSON
        try:
            quiz_data = json.loads(cleaned_text)

        except json.JSONDecodeError:

            # Sometimes an LLM may return extra text.
            # Try to find the JSON array.
            match = re.search(
                r"\[[\s\S]*\]",
                cleaned_text
            )

            if not match:
                return [
                    {
                        "error": "Could not parse Gemini response as JSON."
                    }
                ]

            quiz_data = json.loads(match.group(0))

        # Handle expected list
        if isinstance(quiz_data, list):

            valid_questions = []

            for item in quiz_data:

                if not isinstance(item, dict):
                    continue

                question = item.get("question")
                options = item.get("options")
                answer = item.get("answer")

                if (
                    question
                    and isinstance(options, list)
                    and len(options) == 4
                    and answer in options
                ):
                    valid_questions.append(
                        {
                            "question": question,
                            "options": options,
                            "answer": answer
                        }
                    )

            if valid_questions:
                return valid_questions

            return [
                {
                    "error": "Gemini did not return valid quiz questions."
                }
            ]

        # Handle {"quiz": [...]}
        if isinstance(quiz_data, dict) and "quiz" in quiz_data:

            quiz_list = quiz_data["quiz"]

            if isinstance(quiz_list, list):
                return quiz_list

        return [
            {
                "error": "Unexpected quiz format returned by Gemini."
            }
        ]

    except Exception as e:

        return [
            {
                "error": f"Error in Quiz generation: {str(e)}"
            }
        ]