from gemini_client import generate_with_gemini


def summarize_text(text: str) -> str:
    """
    Summarizes educational content into concise,
    clear, and easy-to-understand takeaways.
    """

    if not text or not text.strip():
        return "⚠️ Please enter some text to summarize."

    prompt = f"""
You are EduGenie, an AI educational tutor.

Summarize the following educational content in simple,
clear language suitable for a college student.

Instructions:
- Identify the main ideas.
- Keep important facts and concepts.
- Remove unnecessary repetition.
- Use short paragraphs or bullet points.
- Make the summary easy to study and remember.
- Do not add information that is not present in the original text.

Educational Content:
{text.strip()}
"""

    try:
        result = generate_with_gemini(prompt)

        if not result:
            return "⚠️ Gemini returned an empty summary."

        return result

    except Exception as e:
        return f"⚠️ Error in Summary: {str(e)}"