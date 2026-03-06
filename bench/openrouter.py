from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PreflightStatus:
    has_key: bool
    reachable: bool
    models_loaded: bool
    error: Optional[str]
    models: List[Dict[str, Any]]


@dataclass
class GenResult:
    ok: bool
    content: str
    raw: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            self.headers["HTTP-Referer"] = referer
        if title:
            self.headers["X-Title"] = title

        self._models_cache: Optional[List[Dict[str, Any]]] = None

    def preflight(self, timeout: int = 15) -> PreflightStatus:
        if not self.api_key:
            return PreflightStatus(False, False, False, "OPENROUTER_API_KEY missing", [])

        # endpoint reachability
        try:
            r = requests.get(f"{self.base}/models", headers=self.headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            models = j.get("data", []) or []
            if isinstance(models, list):
                self._models_cache = models
            return PreflightStatus(True, True, True, None, self._models_cache or [])
        except Exception as e:
            # distinguish reachable vs models_loaded is not super important; keep simple
            return PreflightStatus(True, False, False, str(e), [])

    def list_models(self) -> List[Dict[str, Any]]:
        if self._models_cache is not None:
            return self._models_cache
        st = self.preflight()
        return st.models

    def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        timeout: int = 40,
    ) -> GenResult:
        if not self.api_key:
            return GenResult(False, "", {}, 0, 0, "OPENROUTER_API_KEY missing")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            r = requests.post(f"{self.base}/chat/completions", headers=self.headers, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            content = j["choices"][0]["message"]["content"]
            usage = j.get("usage", {}) or {}
            return GenResult(
                ok=True,
                content=content,
                raw=j,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            )
        except Exception as e:
            return GenResult(False, "", {}, 0, 0, str(e))


def calculate_cost(model_row: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> float:
    pricing = model_row.get("pricing", {}) or {}
    p = float(pricing.get("prompt", 0) or 0)
    c = float(pricing.get("completion", 0) or 0)
    return prompt_tokens * p + completion_tokens * c
