"""Provider-agnostic LLM client (Google Gemini free tier or Anthropic Claude).

Features: provider backends, free-tier throttling, rate-limit retries with backoff,
model fallback, JSON extraction + validation with re-asking, and an on-disk cache.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path


class LLMError(RuntimeError):
    """Raised for configuration problems or unusable model output."""


class LLMAPIError(LLMError):
    """The provider rejected the request (bad key, no quota, unknown model...). Not worth re-asking."""


# environment variable(s) holding the API key for each provider
KEY_ENV = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "anthropic": ("ANTHROPIC_API_KEY",)}


def key_env_name(provider: str) -> str:
    if provider not in KEY_ENV:
        raise LLMError(f"Unknown llm.provider '{provider}'. Use one of: {', '.join(KEY_ENV)}")
    return KEY_ENV[provider][0]


def find_api_key(provider: str):
    key_env_name(provider)  # validates provider
    return next((os.environ[n] for n in KEY_ENV[provider] if os.environ.get(n)), None)


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating markdown fences or stray prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # take whichever bracket type opens first, so a top-level array is not mistaken for an object
    spans = sorted((text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) != -1)
    for start, _, close in spans:
        end = text.rfind(close)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return valid JSON: {text[:200]!r}")


def _status(err: Exception):
    return getattr(err, "code", None) or getattr(err, "status_code", None)


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err)
    return _status(err) == 429 or "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _is_not_found(err: Exception) -> bool:
    return _status(err) == 404 or "NOT_FOUND" in str(err)


# ----------------------------------------------------------------------------- backends
class GeminiBackend:
    """Google Gemini via the google-genai SDK (free tier available in Google AI Studio)."""

    def __init__(self, cfg: dict, api_key: str, sdk_client=None):
        if sdk_client is None:
            from google import genai
            sdk_client = genai.Client(api_key=api_key)
        self.sdk = sdk_client
        self.cfg = cfg
        self.models = [cfg["model"], *cfg.get("fallback_models", [])]
        self.active = 0

    def generate(self, system: str, user: str) -> str:
        while True:
            try:
                resp = self.sdk.models.generate_content(
                    model=self.models[self.active],
                    contents=user,
                    config={
                        "system_instruction": system,
                        "temperature": self.cfg["temperature"],
                        "max_output_tokens": self.cfg["max_tokens"],
                        "response_mime_type": "application/json",
                    },
                )
                return resp.text or ""
            except Exception as err:
                # a model may be retired or unavailable to this key: move on to the next one
                if _is_not_found(err) and self.active + 1 < len(self.models):
                    self.active += 1
                    continue
                raise


class AnthropicBackend:
    """Anthropic Claude via the anthropic SDK (paid API)."""

    def __init__(self, cfg: dict, api_key: str, sdk_client=None):
        if sdk_client is None:
            import anthropic
            sdk_client = anthropic.Anthropic(api_key=api_key, max_retries=cfg.get("max_retries", 3))
        self.sdk = sdk_client
        self.cfg = cfg

    def generate(self, system: str, user: str) -> str:
        resp = self.sdk.messages.create(
            model=self.cfg["model"], max_tokens=self.cfg["max_tokens"],
            temperature=self.cfg["temperature"], system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")


BACKENDS = {"gemini": GeminiBackend, "anthropic": AnthropicBackend}


# ----------------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, llm_cfg: dict, backend=None):
        self.cfg = llm_cfg
        self.provider = llm_cfg.get("provider", "gemini")
        self.cache_dir = Path(llm_cfg.get("cache_dir", ".cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_calls = 0
        self.cache_hits = 0
        self._last_call = 0.0
        if backend is None:  # tests inject a fake backend
            key = find_api_key(self.provider)
            if not key:
                raise LLMError(f"{key_env_name(self.provider)} is not set. "
                               "Copy .env.example to .env and add your key.")
            backend = BACKENDS[self.provider](llm_cfg, key)
        self.backend = backend

    def _cache_path(self, system: str, user: str) -> Path:
        raw = json.dumps([self.provider, self.cfg["model"], self.cfg["temperature"], system, user])
        return self.cache_dir / (hashlib.sha256(raw.encode()).hexdigest() + ".json")

    def _throttle(self):
        """Stay under free-tier requests-per-minute limits."""
        wait = self._last_call + self.cfg.get("min_seconds_between_calls", 0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _call(self, system: str, user: str) -> str:
        retries = self.cfg.get("rate_limit_retries", 4)
        for attempt in range(retries + 1):
            self._throttle()
            self.api_calls += 1
            try:
                return self.backend.generate(system, user)
            except Exception as err:
                if _is_rate_limit(err) and attempt < retries:
                    time.sleep(self.cfg.get("retry_wait_seconds", 20) * (attempt + 1))
                    continue
                raise LLMAPIError(f"{self.provider} API error: {err}") from err
            finally:
                self._last_call = time.monotonic()

    def complete_json(self, system: str, user: str, validator=None):
        """Return parsed (and validated) JSON. Re-asks if the reply is unusable."""
        cache_file = self._cache_path(system, user)
        if cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text())

        last_err = None
        for _ in range(self.cfg.get("parse_attempts", 2)):
            try:
                data = extract_json(self._call(system, user))
                if validator:
                    data = validator(data)
                cache_file.write_text(json.dumps(data))
                return data
            except LLMAPIError:
                raise
            except (LLMError, ValueError, KeyError, TypeError) as err:
                last_err = err
        raise LLMError(f"Unusable model output after retries: {last_err}")
