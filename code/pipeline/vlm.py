from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from typing import Any, Callable, Dict, List, Optional

from .config import Config, load_config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env_file(path: Optional[str] = None) -> None:
    path = path or os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_SCHEMA_DROP_KEYS = (
    "title", "default", "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems", "examples",
)


def _strictify(node):
    if isinstance(node, dict):
        if "$ref" in node and len(node) > 1:
            ref = node["$ref"]
            node.clear()
            node["$ref"] = ref
            return node
        for k in _SCHEMA_DROP_KEYS:
            node.pop(k, None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
    return node


def strict_json_schema(model) -> dict:
    return _strictify(model.model_json_schema())


class ResponseCache:
    """Disk cache keyed on the full request (model + messages + schema). A re-run of an
    already-seen claim/image is a free cache hit, so development iteration never re-bills."""

    def __init__(self, path: str):
        self.path = path
        self.store: Dict[str, str] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}

    @staticmethod
    def key(model: str, messages, json_schema: Optional[dict]) -> str:
        blob = json.dumps([model, messages, json_schema], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    def put(self, key: str, value: str) -> None:
        self.store[key] = value
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f)


class VLMError(Exception):
    pass


class VLMValidationError(VLMError):
    pass


def encode_image(path: str, max_dim: int, quality: int):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        size = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    return url, len(raw), size


class VLMClient:
    def __init__(self, config: Config):
        self.cfg = config
        self._client = None
        self.usage: List[Dict[str, Any]] = []
        self.offline = False  # when True, a cache miss raises instead of billing the API
        self.cache: Optional[ResponseCache] = None
        if self.cfg.response_cache:
            path = self.cfg.response_cache
            if not os.path.isabs(path):
                path = os.path.join(_REPO_ROOT, "code", path)
            self.cache = ResponseCache(path)

    def _ensure_client(self):
        if self._client is None:
            if self.cfg.provider != "openai":
                raise VLMError(
                    f"provider {self.cfg.provider!r}: interface defined but not wired; "
                    "only 'openai' is implemented"
                )
            load_env_file()
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _build_messages(self, system: str, user_content: str, image_urls: List[str]):
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_content}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _is_reasoning(model: str) -> bool:
        # Reasoning models (gpt-5*, o-series) spend hidden reasoning tokens, reject temperature!=1,
        # and require max_completion_tokens with a budget large enough to leave room for output.
        return model.startswith(("gpt-5", "o1", "o3", "o4"))

    def _call(self, model: str, messages, json_schema: Optional[dict]) -> str:
        cache_key = None
        if self.cache is not None:
            cache_key = ResponseCache.key(model, messages, json_schema)
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.usage.append(
                    {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "cached": True}
                )
                return hit

        if self.offline:
            raise VLMError("offline mode: response not in cache (refusing to bill the API)")

        client = self._ensure_client()
        reasoning = self._is_reasoning(model)
        kwargs: Dict[str, Any] = dict(model=model, messages=messages)
        if not reasoning:
            kwargs["temperature"] = self.cfg.temperature  # reasoning models only allow the default
        if json_schema is not None and self.cfg.structured_output_mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "prediction", "strict": True, "schema": json_schema},
            }
        elif self.cfg.structured_output_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        # Reasoning models need a budget large enough for hidden reasoning + the visible answer.
        budget = self.cfg.reasoning_max_tokens if reasoning else self.cfg.max_output_tokens

        try:
            resp = client.chat.completions.create(max_tokens=budget, **kwargs)
        except TypeError:
            resp = client.chat.completions.create(max_completion_tokens=budget, **kwargs)
        except Exception as e:
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                resp = client.chat.completions.create(max_completion_tokens=budget, **kwargs)
            else:
                raise

        u = getattr(resp, "usage", None)
        self.usage.append(
            {
                "model": model,
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "cached": False,
            }
        )
        content = resp.choices[0].message.content
        if cache_key is not None:
            self.cache.put(cache_key, content)
        return content

    def infer(
        self,
        system: str,
        user_content: str,
        images: Optional[List[str]] = None,
        json_schema: Optional[dict] = None,
        role: str = "adjudication",
        validate: Optional[Callable[[dict], Any]] = None,
    ):
        model = self.cfg.model(role)
        image_urls = [
            encode_image(p, self.cfg.per_image_max_dim, self.cfg.image_jpeg_quality)[0]
            for p in (images or [])
        ]
        messages = self._build_messages(system, user_content, image_urls)

        last_err: Optional[Exception] = None
        for _ in range(self.cfg.repair_retries + 1):
            raw = self._call(model, messages, json_schema)
            data = None
            try:
                data = json.loads(raw)
            except Exception as e:
                last_err = e
            if data is not None:
                if validate is None:
                    return data
                try:
                    return validate(data)
                except Exception as e:
                    last_err = e
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Your previous response was invalid: {last_err}. "
                    "Reply with corrected JSON only, matching the required schema.",
                },
            ]
        raise VLMValidationError(
            f"VLM output failed validation after {self.cfg.repair_retries + 1} attempts: {last_err}"
        )

    def cost_estimate(self) -> Dict[str, Any]:
        calls = len(self.usage)
        cached_calls = sum(1 for u in self.usage if u.get("cached"))
        tin = sum(u["prompt_tokens"] for u in self.usage)
        tout = sum(u["completion_tokens"] for u in self.usage)
        cost = 0.0
        for u in self.usage:
            p = self.cfg.pricing(u["model"])
            if p:
                cost += u["prompt_tokens"] / 1e6 * p.get("input", 0)
                cost += u["completion_tokens"] / 1e6 * p.get("output", 0)
        return {
            "calls": calls,
            "cached_calls": cached_calls,
            "billed_calls": calls - cached_calls,
            "input_tokens": tin,
            "output_tokens": tout,
            "est_cost_usd": round(cost, 4),
        }


def make_client(config_path: Optional[str] = None) -> VLMClient:
    return VLMClient(load_config(config_path))
