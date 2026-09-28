import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)


# Current Gemini models
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


def generate_with_gemini(prompt: str, json_mode: bool = False) -> str:
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please add your Gemini API key to the .env file."
        )

    api_key = api_key.strip()

    client = genai.Client(api_key=api_key)

    last_error = None

    for model_name in MODELS:

        # Try the same model up to 2 times for temporary 503 errors
        for attempt in range(2):

            try:
                print(
                    f"Trying Gemini model: {model_name} "
                    f"(attempt {attempt + 1})"
                )

                config = None

                if json_mode:
                    config = {
                        "response_mime_type": "application/json"
                    }

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )

                if response and response.text:
                    print(f"Gemini success: {model_name}")
                    return response.text.strip()

                raise RuntimeError(
                    f"Gemini returned an empty response using {model_name}"
                )

            except Exception as exc:
                last_error = exc

                error_text = str(exc)

                print(
                    f"Gemini error with {model_name}: "
                    f"{error_text}"
                )

                # Retry temporary 503 errors
                if "503" in error_text or "UNAVAILABLE" in error_text:
                    if attempt == 0:
                        time.sleep(2)
                        continue

                    # Move to next model
                    break

                # For other errors, don't silently switch models
                raise RuntimeError(
                    f"Gemini API error using {model_name}: {exc}"
                ) from exc

    raise RuntimeError(
        "All Gemini models are temporarily unavailable. "
        f"Last error: {last_error}"
    )