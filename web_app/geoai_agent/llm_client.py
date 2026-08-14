"""
LLM client for the GeoAI Agent core.

Targets the Gemini API free tier via the google-genai SDK. API key from
the GEMINI_API_KEY env var -- no OpenAI-compatible proxy, no billing
required, stays within the free-tier rate limits.

DEFAULT_MODEL is pinned to a specific stable snapshot rather than the
"-latest" alias: "gemini-flash-latest" currently resolves to a brand-new
preview-tier model (gemini-3.7-flash) with a much tighter free-tier quota
(20 requests/day, observed directly) than an established stable model --
switching to a pinned stable name fixed 429 RESOURCE_EXHAUSTED errors this
project was hitting under normal development use. Re-check
`client.models.list()` occasionally, since Google both retires old model
names (404 "no longer available to new users") and ships new ones.
"""

import json
import os

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Plain-text completion."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Completion constrained to a single JSON object.

        Uses Gemini's response_mime_type="application/json" mode. We also
        strip markdown code fences as a defensive fallback in case the model
        wraps the JSON anyway.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[len("json"):]
            raw = raw.strip()
        return json.loads(raw)
