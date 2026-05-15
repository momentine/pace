# # from __future__ import annotations

# # import os
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Any, Dict, List, Optional

# # import requests
# # from dotenv import load_dotenv

# # load_dotenv()


# # @dataclass
# # class PreflightStatus:
# #     has_key: bool
# #     reachable: bool
# #     models_loaded: bool
# #     error: Optional[str]
# #     models: List[Dict[str, Any]]


# # @dataclass
# # class GenResult:
# #     ok: bool
# #     content: str
# #     raw: Dict[str, Any]
# #     prompt_tokens: int
# #     completion_tokens: int
# #     error: Optional[str] = None


# # class AzureFoundryClient:
# #     def __init__(
# #         self,
# #         api_key: Optional[str] = None,
# #         base_url: Optional[str] = None,
# #         models_path: Optional[str] = None,
# #     ):
# #         self.api_key = api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
# #         self.base_url = (base_url or os.getenv("AZURE_FOUNDRY_BASE_URL", "")).rstrip("/")
# #         self.models_path = Path(models_path or os.getenv("BENCH_MODELS_PATH", "data/selected_models.json")).resolve()

# #         self.headers = {
# #             "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
# #             "Content-Type": "application/json",
# #         }

# #         self._models_cache: Optional[List[Dict[str, Any]]] = None
# #         self._deployment_by_id: Dict[str, str] = {}

# #     def _load_local_models(self) -> List[Dict[str, Any]]:
# #         import json

# #         if not self.models_path.exists():
# #             return []

# #         try:
# #             data = json.loads(self.models_path.read_text(encoding="utf-8"))
# #             rows = data.get("data", []) or []
# #             if not isinstance(rows, list):
# #                 return []

# #             self._models_cache = rows
# #             self._deployment_by_id = {
# #                 str(row.get("id")).strip(): str(row.get("deployment")).strip()
# #                 for row in rows
# #                 if isinstance(row, dict)
# #                 and str(row.get("id") or "").strip()
# #                 and str(row.get("deployment") or "").strip()
# #             }
# #             return rows
# #         except Exception:
# #             return []

# #     def preflight(self, timeout: int = 15) -> PreflightStatus:
# #         if not self.api_key:
# #             return PreflightStatus(False, False, False, "AZURE_FOUNDRY_API_KEY missing", [])

# #         if not self.base_url:
# #             return PreflightStatus(True, False, False, "AZURE_FOUNDRY_BASE_URL missing", [])

# #         local_models = self._load_local_models()
# #         if not local_models:
# #             return PreflightStatus(
# #                 True,
# #                 False,
# #                 False,
# #                 f"No local Azure model definitions found at {self.models_path}",
# #                 [],
# #             )

# #         first = local_models[0]
# #         deployment = str(first.get("deployment") or "").strip()
# #         if not deployment:
# #             return PreflightStatus(True, False, False, "First model missing deployment", local_models)

# #         payload = {
# #             "model": deployment,
# #             "messages": [{"role": "user", "content": "ping"}],
# #             "max_tokens": 1,
# #         }

# #         try:
# #             r = requests.post(
# #                 f"{self.base_url}/chat/completions",
# #                 headers=self.headers,
# #                 json=payload,
# #                 timeout=timeout,
# #             )
# #             r.raise_for_status()
# #             return PreflightStatus(True, True, True, None, local_models)
# #         except Exception as e:
# #             return PreflightStatus(True, False, False, str(e), local_models)

# #     def list_models(self) -> List[Dict[str, Any]]:
# #         if self._models_cache is not None:
# #             return self._models_cache
# #         return self._load_local_models()

# #     def generate(
# #         self,
# #         model_id: str,
# #         system_prompt: str,
# #         user_prompt: str,
# #         max_tokens: int = 1200,
# #         temperature: float = 0.7,
# #         top_p: float = 0.95,
# #         timeout: int = 40,
# #     ) -> GenResult:
# #         if not self.api_key:
# #             return GenResult(False, "", {}, 0, 0, "AZURE_FOUNDRY_API_KEY missing")

# #         if not self.base_url:
# #             return GenResult(False, "", {}, 0, 0, "AZURE_FOUNDRY_BASE_URL missing")

# #         if self._models_cache is None:
# #             self._load_local_models()

# #         deployment = self._deployment_by_id.get(model_id, "").strip()
# #         if not deployment:
# #             return GenResult(False, "", {}, 0, 0, f"No Azure deployment mapped for model_id={model_id}")

# #         payload = {
# #             "model": deployment,
# #             "messages": [
# #                 {"role": "system", "content": system_prompt.strip()},
# #                 {"role": "user", "content": user_prompt.strip()},
# #             ],
# #             "max_tokens": max_tokens,
# #             "temperature": temperature,
# #             "top_p": top_p,
# #         }

# #         try:
# #             r = requests.post(
# #                 f"{self.base_url}/chat/completions",
# #                 headers=self.headers,
# #                 json=payload,
# #                 timeout=timeout,
# #             )
# #             r.raise_for_status()
# #             j = r.json()

# #             content = j["choices"][0]["message"]["content"]
# #             usage = j.get("usage", {}) or {}

# #             return GenResult(
# #                 ok=True,
# #                 content=content,
# #                 raw=j,
# #                 prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
# #                 completion_tokens=int(usage.get("completion_tokens", 0) or 0),
# #             )
# #         except Exception as e:
# #             return GenResult(False, "", {}, 0, 0, str(e))


# # def calculate_cost(model_row: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> float:
# #     pricing = model_row.get("pricing", {}) or {}
# #     p = float(pricing.get("prompt", 0) or 0)
# #     c = float(pricing.get("completion", 0) or 0)
# #     return prompt_tokens * p + completion_tokens * c

# from __future__ import annotations

# import os
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import requests
# from dotenv import load_dotenv

# load_dotenv()


# @dataclass
# class PreflightStatus:
#     has_key: bool
#     reachable: bool
#     models_loaded: bool
#     error: Optional[str]
#     models: List[Dict[str, Any]]


# @dataclass
# class GenResult:
#     ok: bool
#     content: str
#     raw: Dict[str, Any]
#     prompt_tokens: int
#     completion_tokens: int
#     error: Optional[str] = None


# class AzureFoundryClient:
#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         base_url: Optional[str] = None,
#         models_path: Optional[str] = None,
#     ):
#         self.api_key = api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
#         self.base_url = (base_url or os.getenv("AZURE_FOUNDRY_BASE_URL", "")).rstrip("/")
#         self.models_path = Path(
#             models_path or os.getenv("BENCH_MODELS_PATH", "data/selected_models.json")
#         ).resolve()

#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
#             "Content-Type": "application/json",
#         }

#         self._models_cache: Optional[List[Dict[str, Any]]] = None
#         self._deployment_by_id: Dict[str, str] = {}

#     def _load_local_models(self) -> List[Dict[str, Any]]:
#         import json

#         if not self.models_path.exists():
#             return []

#         try:
#             data = json.loads(self.models_path.read_text(encoding="utf-8"))
#             rows = data.get("data", []) or []
#             if not isinstance(rows, list):
#                 return []

#             self._models_cache = rows
#             self._deployment_by_id = {
#                 str(row.get("id")).strip(): str(row.get("deployment")).strip()
#                 for row in rows
#                 if isinstance(row, dict)
#                 and str(row.get("id") or "").strip()
#                 and str(row.get("deployment") or "").strip()
#             }
#             return rows
#         except Exception:
#             return []

#     def preflight(self, timeout: int = 15) -> PreflightStatus:
#         if not self.api_key:
#             return PreflightStatus(False, False, False, "AZURE_FOUNDRY_API_KEY missing", [])

#         if not self.base_url:
#             return PreflightStatus(True, False, False, "AZURE_FOUNDRY_BASE_URL missing", [])

#         local_models = self._load_local_models()
#         if not local_models:
#             return PreflightStatus(
#                 True,
#                 False,
#                 False,
#                 f"No local Azure model definitions found at {self.models_path}",
#                 [],
#             )

#         first = local_models[0]
#         deployment = str(first.get("deployment") or "").strip()
#         if not deployment:
#             return PreflightStatus(True, False, False, "First model missing deployment", local_models)

#         payload = {
#             "model": deployment,
#             "input": "ping",
#         }

#         try:
#             r = requests.post(
#                 f"{self.base_url}/responses",
#                 headers=self.headers,
#                 json=payload,
#                 timeout=timeout,
#             )

#             if not r.ok:
#                 return PreflightStatus(
#                     True,
#                     False,
#                     False,
#                     f"{r.status_code} {r.text}",
#                     local_models,
#                 )

#             return PreflightStatus(True, True, True, None, local_models)
#         except Exception as e:
#             return PreflightStatus(True, False, False, str(e), local_models)

#     def list_models(self) -> List[Dict[str, Any]]:
#         if self._models_cache is not None:
#             return self._models_cache
#         return self._load_local_models()

#     def generate(
#         self,
#         model_id: str,
#         system_prompt: str,
#         user_prompt: str,
#         max_tokens: int = 1200,
#         temperature: float = 0.7,
#         top_p: float = 0.95,
#         timeout: int = 40,
#     ) -> GenResult:
#         if not self.api_key:
#             return GenResult(False, "", {}, 0, 0, "AZURE_FOUNDRY_API_KEY missing")

#         if not self.base_url:
#             return GenResult(False, "", {}, 0, 0, "AZURE_FOUNDRY_BASE_URL missing")

#         if self._models_cache is None:
#             self._load_local_models()

#         deployment = self._deployment_by_id.get(model_id, "").strip()
#         if not deployment:
#             return GenResult(False, "", {}, 0, 0, f"No Azure deployment mapped for model_id={model_id}")

#         payload = {
#             "model": deployment,
#             "instructions": system_prompt.strip(),
#             "input": user_prompt.strip(),
#             "max_output_tokens": max_tokens,
#             "temperature": temperature,
#             "top_p": top_p,
#         }

#         try:
#             r = requests.post(
#                 f"{self.base_url}/responses",
#                 headers=self.headers,
#                 json=payload,
#                 timeout=timeout,
#             )

#             if not r.ok:
#                 return GenResult(False, "", {}, 0, 0, f"{r.status_code} {r.text}")

#             j = r.json()

#             content = j.get("output_text", "")
#             if not content:
#                 parts: List[str] = []
#                 for item in j.get("output", []) or []:
#                     for c in item.get("content", []) or []:
#                         text = c.get("text")
#                         if text:
#                             parts.append(text)
#                 content = "".join(parts)

#             usage = j.get("usage", {}) or {}

#             return GenResult(
#                 ok=True,
#                 content=content,
#                 raw=j,
#                 prompt_tokens=int(usage.get("input_tokens", 0) or 0),
#                 completion_tokens=int(usage.get("output_tokens", 0) or 0),
#             )
#         except Exception as e:
#             return GenResult(False, "", {}, 0, 0, str(e))


# def calculate_cost(model_row: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> float:
#     pricing = model_row.get("pricing", {}) or {}
#     p = float(pricing.get("prompt", 0) or 0)
#     c = float(pricing.get("completion", 0) or 0)
#     return prompt_tokens * p + completion_tokens * c