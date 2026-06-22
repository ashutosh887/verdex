import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> code/

from pipeline.config import load_config  # noqa: E402
from pipeline.vlm import VLMClient, encode_image  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_IMG = sorted(glob.glob(os.path.join(REPO_ROOT, "dataset", "images", "sample", "*", "*.jpg")))[0]


def test_config_resolves_provider_and_roles():
    cfg = load_config()
    assert cfg.provider == "openai"
    assert cfg.model("adjudication")           # role resolves to some model id
    assert cfg.model("inspection") == "gpt-4o-mini"
    assert cfg.model("ocr") == "gpt-4o-mini"
    assert cfg.pricing("gpt-4o")["input"] == 2.50


def test_image_downscaled_within_budget():
    cfg = load_config()
    url, nbytes, size = encode_image(SAMPLE_IMG, cfg.per_image_max_dim, cfg.image_jpeg_quality)
    assert url.startswith("data:image/jpeg;base64,")
    assert max(size) <= cfg.per_image_max_dim
    assert nbytes > 0


def test_client_constructs_without_key_and_tracks_usage():
    cfg = load_config()
    client = VLMClient(cfg)
    assert client.usage == []
    est = client.cost_estimate()
    assert est["calls"] == 0
    assert est["billed_calls"] == 0
    assert est["est_cost_usd"] == 0.0


def test_cost_estimate_math():
    cfg = load_config()
    client = VLMClient(cfg)
    client.usage.append({"model": "gpt-4o", "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    est = client.cost_estimate()
    assert est["est_cost_usd"] == 12.5  # 2.50 in + 10.00 out per 1M


if __name__ == "__main__":
    import traceback

    tests = sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v))
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
