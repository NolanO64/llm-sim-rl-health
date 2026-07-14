"""Client for the Nebula OpenAI-compatible gateway used to query the language model.

Besides the usual rate-limit / 5xx / timeout errors, the Nebula gateway reports
model cooldown and its five-concurrent-request limit as HTTP 400s; those must be
retried with backoff rather than raised. The API key is read from the environment
(NEBULA_API_KEY), optionally via a .env file at the repository root.
"""
import os
import random
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

NEBULA_BASE_URL = "https://nebula.cs.vu.nl/api/"

_TRANSIENT = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
_TRANSIENT_400_MARKERS = (
    "no deployments", "try again", "cooldown", "too many requests",
    "rate limit", "ratelimiterror", "exceeds the maximum", "429",
)


def _is_transient_400(error):
    if not isinstance(error, BadRequestError):
        return False
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_400_MARKERS)


def build_client():
    return OpenAI(base_url=NEBULA_BASE_URL, api_key=os.environ["NEBULA_API_KEY"])


def reasoning_extra_body(thinking=False):
    """Gateway-specific control that disables the model's chain-of-thought."""
    return {"chat_template_kwargs": {"enable_thinking": thinking}}


def chat(client, max_retries=40, **kwargs):
    """A chat-completions call that retries the gateway's transient failures."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            if not (isinstance(error, _TRANSIENT) or _is_transient_400(error)):
                raise
            # exponential backoff capped at 30s, with jitter so concurrent
            # workers desynchronise instead of retrying in lockstep
            wait = min(30, 2 ** min(attempt, 4)) + random.uniform(0, 4)
            if attempt % 6 == 0:
                print("  [transient %s, retry %d]" % (type(error).__name__, attempt + 1), flush=True)
            time.sleep(wait)
    raise RuntimeError("gateway still failing after %d retries" % max_retries)
