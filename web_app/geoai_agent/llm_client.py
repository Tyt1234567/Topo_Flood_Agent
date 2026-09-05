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
# Raised again to 150s: with a ~70-hotspot significant-basin run, each
# hotspot now carries ~20 fields (contributing area, five per-factor
# scores, runoff coefficient + source, peak runoff, flood-zone/river-gauge
# tags, TWI, building exposure, EAD), and 90s was observed to still time
# out on both attempts (a raw "The read operation timed out", not a 504
# from Google -- i.e. genuinely slow generation, not a hang or quota issue).
_REQUEST_TIMEOUT_MS = 150_000
# Two retries (three attempts total) for a plain chat() call (the
# recommendation/QA paths, not the already-retried chat_json): raised from
# one retry once a 90s timeout was observed to fail twice in a row on the
# largest recommendation prompts -- a transient network stall (this
# sandbox's own docstring-documented intermittent path to Google's API, not
# a real service outage) often succeeds cleanly on a fresh attempt, and the
# prompt re-serializes identically each call, so an extra retry costs one
# more request, not a different or stale prompt.
_CHAT_MAX_ATTEMPTS = 3


def _load_api_keys(api_key: str | None) -> list[str]:
    """Resolves the ordered list of API keys this client will fail over
    across. Multiple keys exist to survive one Google account's free-tier
    daily quota (20 requests/day, observed directly) running out mid-run --
    a real, repeated failure during this project's own development/demo
    use, not a hypothetical.

    An explicit `api_key` argument (if passed) is used alone, same as
    before this existed -- no rotation, matching a caller that deliberately
    wants one specific key. Otherwise GEMINI_API_KEY is tried first, then
    GEMINI_API_KEYS (comma-separated, from separate Google accounts/
    projects, each with its own independent daily quota) in the order
    listed."""
    if api_key:
        return [api_key]
    keys = []
    primary = os.environ.get("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    for extra in os.environ.get("GEMINI_API_KEYS", "").split(","):
        extra = extra.strip()
        if extra and extra not in keys:
            keys.append(extra)
    if not keys:
        raise RuntimeError("No Gemini API key configured -- set GEMINI_API_KEY (and optionally GEMINI_API_KEYS)")
    return keys


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api_keys = _load_api_keys(api_key)
        self._key_index = 0
        self.client = genai.Client(
            api_key=self._api_keys[0],
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )

    def _rotate_key(self) -> bool:
        """Switches to the next configured API key and returns True, or
        returns False if every configured key has already been tried.
        Rebuilds self.client entirely (a fresh genai.Client, not just a
        header swap) since the SDK binds the key at client-construction
        time. Persists on this LLMClient instance -- a caller that reuses
        the same instance for a follow-up call (e.g. reflect_on_
        recommendation right after the recommendation-writing chat() call)
        starts that follow-up call already on the key that actually worked,
        instead of re-discovering the same exhausted first key again."""
        if self._key_index + 1 >= len(self._api_keys):
            return False
        self._key_index += 1
        self.client = genai.Client(
            api_key=self._api_keys[self._key_index],
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )
        return True

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Plain-text completion. Retries once on a timeout (see
        _CHAT_MAX_ATTEMPTS) -- a transient network stall reaching Google's
        API, not a malformed-response issue (that's chat_json's concern
        below), so the fix here is just "try again", not "repair the text".

        Does NOT retry a `genai.errors.ClientError` (any 4xx -- most
        importantly 429 RESOURCE_EXHAUSTED, this project's documented
        free-tier daily quota limit, but also a bad request or an auth
        failure) against the SAME key: a real bug caught by review -- the
        original `except Exception` retried EVERY failure including
        permanent ones, so once the daily quota was hit, every subsequent
        chat() call burned 2 requests against that same exhausted quota
        instead of failing fast on the first. Instead, a ClientError (or a
        transient stall that has burned through every attempt on this key)
        rotates to the next configured key, if one exists, and retries
        there with a fresh attempt budget; only once every key has been
        tried does this actually raise. A raw network stall/timeout raises
        a plain httpx/socket exception here, not a ClientError (there's no
        HTTP response to build one from), so those still get their retries
        on the SAME key before rotation is attempted."""
        last_error: Exception | None = None
        while True:
            for attempt in range(_CHAT_MAX_ATTEMPTS):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(system_instruction=system_prompt),
                    )
                    return response.text
                except errors.ClientError as exc:
                    last_error = exc
                    if self._rotate_key():
                        break
                    raise
                except Exception as exc:
                    last_error = exc
                    if attempt == _CHAT_MAX_ATTEMPTS - 1:
                        if self._rotate_key():
                            break
                        raise
            else:
                raise last_error  # unreachable, satisfies static analysis
            continue  # a key rotation happened -- retry fresh on the new key

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

        A `genai.errors.ClientError` (429 quota, auth, etc.) is NOT treated
        as a parse failure to retry past on the same key -- it rotates to
        the next configured key (see LLMClient._rotate_key) and restarts
        the attempt budget there instead, only raising once every key is
        exhausted. Malformed JSON, by contrast, is never a key/quota
        problem, so it keeps retrying the SAME key rather than rotating."""
        last_error: Exception | None = None
        while True:
            for _ in range(max_attempts):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                        ),
                    )
                except errors.ClientError as exc:
                    last_error = exc
                    if self._rotate_key():
                        break
                    raise
                else:
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
            else:
                raise last_error
            continue  # a key rotation happened -- retry fresh on the new key
