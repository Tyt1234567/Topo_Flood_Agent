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
# The SDK has no default request timeout -- a slow/stalled path to Google's
# API (observed directly: intermittent on this project's sandboxed dev
# network) otherwise hangs generate_content() forever, with nothing to
# catch since it's not an exception. That's fatal for agent_analyze_stream
# in web_app/app.py: the whole NDJSON response holds open mid-pipeline and
# the frontend just sits on "Generating recommendation (LLM)" with no error
# ever surfacing. Started at 30s; raised to 60s after the chat endpoint's
# qa_context grew to include every raw local minimum (up to ~500) instead
# of a top-10 sample -- a bigger prompt plus a request to enumerate/filter
# over it takes noticeably longer to generate, and 30s was observed to
# genuinely trip (504 DEADLINE_EXCEEDED, not a hang) on exactly that kind
# of "list every basin matching X" chat question.
_REQUEST_TIMEOUT_MS = 60_000


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = genai.Client(
            api_key=api_key or os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Plain-text completion."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text

    def chat_json(self, system_prompt: str, user_prompt: str, max_attempts: int = 3) -> dict:
        """Completion constrained to a single JSON object.

        Uses Gemini's response_mime_type="application/json" mode. We also
        strip markdown code fences as a defensive fallback in case the model
        wraps the JSON anyway.

        response_mime_type="application/json" does not fully guarantee
        well-formed output: observed directly with gemini-3.5-flash, roughly
        one call in three returns malformed JSON (an extra closing brace, or
        a string cut off mid-value) while finish_reason is still STOP, i.e.
        not a token-budget truncation, just a model-generation glitch. Retry
        the whole call on a parse failure rather than trying to repair the
        malformed text -- the model reliably produces valid JSON on a
        fresh attempt.
        """
        last_error: json.JSONDecodeError | None = None
        for _ in range(max_attempts):
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
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
        raise last_error
