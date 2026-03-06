from __future__ import annotations

import json
from pathlib import Path

import requests

OUT = Path("data/models.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://openrouter.ai/api/v1/models"

def main() -> None:
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Optional: keep only fields you actually use (smaller + faster UI).
    slim = {"data": []}
    for m in data.get("data", []):
        slim["data"].append(
            {
                "id": m.get("id", ""),
                "name": m.get("name", ""),
                "canonical_slug": m.get("canonical_slug", ""),
                "context_length": m.get("context_length", None),
                "architecture": (m.get("architecture", {}) or {}),
                "pricing": (m.get("pricing", {}) or {}),
                "top_provider": (m.get("top_provider", {}) or {}),
                "supported_parameters": m.get("supported_parameters", []) or [],
                "default_parameters": m.get("default_parameters", {}) or {},
                "expiration_date": m.get("expiration_date", None),
            }
        )

    OUT.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(slim['data'])} models -> {OUT}")

if __name__ == "__main__":
    main()
