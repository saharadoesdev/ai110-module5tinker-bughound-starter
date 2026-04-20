import os
from typing import Optional


class MockClient:
    """
    Offline stand-in for an LLM client.
    This lets the app run without an API key.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Very small, predictable behavior for demos.
        if "Return ONLY valid JSON" in system_prompt:
            # Purposely not JSON to force fallback unless students change behavior.
            return "I found some issues, but I'm not returning JSON right now."
        return "# MockClient: no rewrite available in offline mode.\n"


class GeminiClient:
    """
    Minimal Gemini API wrapper with added error resilience.

    Requirements:
    - google-generativeai installed
    - GEMINI_API_KEY set in environment (or loaded via python-dotenv)
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Create a .env file and set GEMINI_API_KEY=..."
            )

        # Import here so heuristic mode doesn't require the dependency at import time.
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = float(temperature)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a single request to Gemini.

        API failures are raised so the caller can log the exact error and then
        decide whether to fall back to heuristics.
        """
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(
                # [
                #     {"role": "system", "parts": [system_prompt]},
                #     {"role": "user", "parts": [user_prompt]},
                # ],
                combined_prompt,
                generation_config={"temperature": self.temperature},
            )

            # Defensive: text may be empty when content is blocked or filtered.
            text = response.text or ""
            if text.strip():
                return text

            finish_reason = "unknown"
            try:
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    finish_reason = str(getattr(candidates[0], "finish_reason", "unknown"))
            except Exception:
                pass

            raise RuntimeError(
                f"Gemini returned empty text (finish_reason={finish_reason})."
            )

        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e
