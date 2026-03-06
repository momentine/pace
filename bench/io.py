from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path("data")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_models() -> Dict[str, Any]:
    return read_json(DATA_DIR / "models.json")


def load_components() -> Dict[str, Any]:
    return read_json(DATA_DIR / "components.json")


def load_score() -> Dict[str, Any]:
    return read_json(DATA_DIR / "score.json")


def load_prompt_conditions() -> Dict[str, Any]:
    return read_json(DATA_DIR / "prompt_conditions.json")


def load_variants() -> Dict[str, Any]:
    return read_json(DATA_DIR / "variants.json")


def save_models(data: Dict[str, Any]) -> None:
    write_json(DATA_DIR / "models.json", data)


def save_components(data: Dict[str, Any]) -> None:
    write_json(DATA_DIR / "components.json", data)


def save_score(data: Dict[str, Any]) -> None:
    write_json(DATA_DIR / "score.json", data)


def save_prompt_conditions(data: Dict[str, Any]) -> None:
    write_json(DATA_DIR / "prompt_conditions.json", data)


def save_variants(data: Dict[str, Any]) -> None:
    write_json(DATA_DIR / "variants.json", data)