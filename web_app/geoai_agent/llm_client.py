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
from google.genai import errors, types

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
# of "list every basin matching X" chat question. Raised again to 90s once
# the flood-recommendation prompt grew to include land-cover/soil-derived
# per-basin runoff coefficients, river-gauge readings, TWI, building
# exposure, and the six-return-period expected-annual-runoff aggregation --
# a genuinely larger structured payload for the model to read and
# synthesize into prose, observed directly to occasionally exceed 60s now.
_REQUEST_TIMEOUT_MS = 90_000
# One retry for a plain chat() call (the recommendation/QA paths, not the
# already-retried chat_json): a request that times out due to a transient
# network stall (this sandbox's own docstring-documented intermittent path
# to Google's API, not a real service outage) often succeeds cleanly on a
# fresh attempt: the risk-ranking JSON re-serializes identically each call,
# so a retry costs one more request, not a different or stale prompt.
_CHAT_MAX_ATTEMPTS = 2


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = genai.Client(
            api_key=api_key or os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Plain-text completion. Retries once on a timeout (see
        _CHAT_MAX_ATTEMPTS) -- a transient network stall reaching Google's
        API, not a malformed-response issue (that's chat_json's concern
        below), so the fix here is just "try again", not "repair the text".

        Does NOT retry a `genai.errors.ClientError` (any 4xx -- most
        importantly 429 RESOURCE_EXHAUSTED, this project's documented
        free-tier daily quota limit, but also a bad request or an auth
        failure): a real bug caught by review -- the original `except
        Exception` retried EVERY failure including permanent ones, so once
        the daily quota was hit, every subsequent chat() call burned 2
        requests against that same exhausted quota instead of failing fast
        on the first. A raw network stall/timeout raises a plain httpx/
        socket exception here, not a ClientError (there's no HTTP response
        to build one from), so those still get their one retry."""
        last_error: Exception | None = None
        for attempt in range(_CHAT_MAX_ATTEMPTS):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                )
                return response.text
            except errors.ClientError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == _CHAT_MAX_ATTEMPTS - 1:
                    raise
        raise last_error  # unreachable, satisfies static analysis

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
