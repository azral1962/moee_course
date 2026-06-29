"""Small JSON-agent adapters for local and hosted model providers."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """Raised when a model provider cannot return a usable decision."""


def load_env_file(path: str | Path) -> None:
    """Load a minimal KEY=VALUE file without overwriting the process environment."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_default_env() -> None:
    for candidate in (Path(".env"), Path("simulation/.env")):
        load_env_file(candidate)


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider JSON response must be an object")
    return parsed


class JSONAgentProvider(ABC):
    name = "abstract"

    @abstractmethod
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class HTTPJSONProvider(JSONAgentProvider):
    def __init__(self, *, base_url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"{self.name} HTTP {exc.code}: {detail[:500]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"{self.name} connection failed: {exc}") from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} returned non-JSON HTTP content") from exc
        if not isinstance(result, dict):
            raise ProviderError(f"{self.name} returned an unexpected response")
        return result


class OllamaProvider(HTTPJSONProvider):
    name = "ollama"

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        try:
            return _extract_json(str(response["message"]["content"]))
        except (KeyError, TypeError) as exc:
            raise ProviderError("Ollama response did not contain message.content") from exc


class OpenAIProvider(HTTPJSONProvider):
    name = "openai"

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._post(
            "/responses",
            {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt + "\nReturn one JSON object matching this schema:\n" + json.dumps(schema),
                "store": False,
            },
        )
        if isinstance(response.get("output_text"), str):
            return _extract_json(response["output_text"])
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return _extract_json(content["text"])
        raise ProviderError("OpenAI response did not contain output text")


class DeepSeekProvider(HTTPJSONProvider):
    name = "deepseek"

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt + "\nReturn JSON matching: " + json.dumps(schema),
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "stream": False,
            },
        )
        try:
            return _extract_json(str(response["choices"][0]["message"]["content"]))
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("DeepSeek response did not contain choices[0].message.content") from exc


class HeuristicProvider(JSONAgentProvider):
    """Auditable local fallback used for development and tests."""

    name = "heuristic"

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            state = _extract_json(user_prompt)
        except ProviderError:
            state = {}
        demand = float(state.get("demand", 0.5))
        disruption = float(state.get("disruption", 0.0))
        resources = float(state.get("available_resources", 1.0))
        capacity = max(float(state.get("capacity", 1.0)), 1e-9)
        intensity = max(0.0, min(1.0, 0.15 + 0.75 * demand - 0.2 * disruption))
        intensity = min(intensity, resources / capacity)
        return {
            "intensity": intensity,
            "confidence": 0.65,
            "rationale": "local heuristic balances demand, disruption, and resource feasibility",
        }


class FakeProvider(JSONAgentProvider):
    """Test double returning a fixed JSON response."""

    name = "fake"

    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls = 0

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return dict(self.response)


class ProviderFactory:
    @staticmethod
    def create(provider: str | None = None, model: str | None = None) -> JSONAgentProvider:
        load_default_env()
        provider_name = (provider or os.getenv("MOEE_LLM_PROVIDER", "heuristic")).lower()
        timeout = float(os.getenv("MOEE_HTTP_TIMEOUT", "60"))
        if provider_name in {"none", "heuristic", "offline"}:
            return HeuristicProvider()
        if provider_name == "ollama":
            return OllamaProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=model or os.getenv("OLLAMA_MODEL", "gemma3"),
                api_key=os.getenv("OLLAMA_API_KEY", ""),
                timeout=timeout,
            )
        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ProviderError("OPENAI_API_KEY is required for the OpenAI provider")
            return OpenAIProvider(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
                api_key=api_key,
                timeout=timeout,
            )
        if provider_name == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ProviderError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
            return DeepSeekProvider(
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                api_key=api_key,
                timeout=timeout,
            )
        raise ProviderError(f"unsupported provider: {provider_name}")
