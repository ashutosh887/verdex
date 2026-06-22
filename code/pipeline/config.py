from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


class Config:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.provider: str = data["provider"]
        self.temperature: float = data.get("temperature", 0)
        self.max_output_tokens: int = data.get("max_output_tokens", 1200)
        self.reasoning_max_tokens: int = data.get("reasoning_max_tokens", 8000)
        self.repair_retries: int = data.get("repair_retries", 1)
        self.per_image_max_dim: int = data.get("per_image_max_dim", 1024)
        self.image_jpeg_quality: int = data.get("image_jpeg_quality", 85)
        self.structured_output_mode: str = data.get("structured_output_mode", "json_schema")
        self.adjudication_prompt_version: str = data.get("adjudication_prompt_version", "v2")
        self.ocr_injection_scan: bool = data.get("ocr_injection_scan", True)
        self.response_cache: Optional[str] = data.get("response_cache")
        if self.provider not in data.get("providers", {}):
            raise KeyError(f"provider {self.provider!r} not present under 'providers'")
        self.provider_cfg: Dict[str, Any] = data["providers"][self.provider]

    def model(self, role: str) -> str:
        key = f"model_{role}"
        if key not in self.provider_cfg:
            raise KeyError(f"no model for role {role!r} under provider {self.provider!r}")
        return self.provider_cfg[key]

    def pricing(self, model: str) -> Dict[str, float]:
        return self.provider_cfg.get("pricing", {}).get(model, {})


def load_config(path: Optional[str] = None) -> Config:
    with open(path or _DEFAULT_PATH, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
