from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|any\s+)?(previous|prior|above)?\s*instructions",
    r"disregard\s+.*(policy|guidelines|instructions|rules)",
    r"approve\s+(this|the|my)\s+claim",
    r"mark\s+(this\s+)?as\s+(paid|approved|resolved|valid)",
    r"you\s+must\s+(approve|accept|pay|mark)",
    r"override\s+(the\s+)?(decision|verdict|system)",
    r"system\s+prompt",
    r"pay\s*out",
    r"force\s+(approve|accept)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

BLUR_VAR_THRESHOLD = 100.0
DARK_MEAN_THRESHOLD = 45.0
GLARE_BRIGHT_FRACTION = 0.45
PHASH_HAMMING_THRESHOLD = 5
_EDITOR_SIGNATURES = ("photoshop", "gimp", "affinity", "lightroom", "pixelmator", "snapseed")


@dataclass
class ImageSignals:
    image_id: str
    path: str
    exists: bool = False
    sha256: Optional[str] = None
    phash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    blur_var: Optional[float] = None
    blurry: bool = False
    low_light_or_glare: bool = False
    missing_exif: bool = False
    editor_software: Optional[str] = None
    probable_screenshot: bool = False
    error: Optional[str] = None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_image(path: str, image_id: str) -> ImageSignals:
    sig = ImageSignals(image_id=image_id, path=path)
    if not os.path.exists(path):
        sig.error = "file not found"
        return sig
    sig.exists = True
    try:
        sig.sha256 = sha256_file(path)
        _vision_signals(path, sig)
    except Exception as e:
        sig.error = f"{type(e).__name__}: {e}"
    return sig


def _vision_signals(path: str, sig: ImageSignals) -> None:
    import imagehash
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        fmt = (im.format or "").upper()
        exif = im.getexif()
        software = exif.get(305) if exif else None
        rgb = im.convert("RGB")
        sig.width, sig.height = rgb.size
        sig.phash = str(imagehash.phash(rgb))
        arr = np.asarray(rgb.convert("L"), dtype="float64")

    sig.missing_exif = not bool(exif) or len(dict(exif)) == 0
    if software:
        s = str(software).lower()
        if any(k in s for k in _EDITOR_SIGNATURES):
            sig.editor_software = str(software)
    sig.probable_screenshot = fmt == "PNG" and sig.missing_exif

    mean = float(arr.mean())
    bright_frac = float((arr > 245).mean())
    sig.low_light_or_glare = mean < DARK_MEAN_THRESHOLD or bright_frac > GLARE_BRIGHT_FRACTION

    lap = _laplacian_variance(arr)
    sig.blur_var = lap
    sig.blurry = lap < BLUR_VAR_THRESHOLD


def _laplacian_variance(gray) -> float:
    import numpy as np

    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype="float64")
    g = gray
    pad = np.pad(g, 1, mode="edge")
    lap = (
        k[0, 1] * pad[:-2, 1:-1]
        + k[1, 0] * pad[1:-1, :-2]
        + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:]
        + k[2, 1] * pad[2:, 1:-1]
    )
    return float(lap.var())


def hamming(a: str, b: str) -> int:
    import imagehash

    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def find_duplicate_pairs(signals: List[ImageSignals], threshold: int = PHASH_HAMMING_THRESHOLD):
    pairs = []
    have = [s for s in signals if s.phash]
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            if have[i].sha256 and have[i].sha256 == have[j].sha256:
                pairs.append((have[i].image_id, have[j].image_id, 0))
                continue
            d = hamming(have[i].phash, have[j].phash)
            if d <= threshold:
                pairs.append((have[i].image_id, have[j].image_id, d))
    return pairs


class PHashStore:
    """Maps a perceptual hash -> the SOURCE KEY of the first image that carried it. A source
    key uniquely identifies a file (its path), NOT the bare filename (`img_1` repeats across
    claims). Reuse = the same content appearing under a *different* source — so re-processing
    the same dataset is idempotent (an image only ever matches its own stored source, which is
    skipped), while genuine cross-claim reuse (two distinct paths, one content) still fires."""

    def __init__(self, path: str):
        self.path = path
        self.store: Dict[str, str] = {}
        if os.path.exists(path):
            try:
                import json

                with open(path, encoding="utf-8") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}

    def match(
        self, phash: str, source_key: str, threshold: int = PHASH_HAMMING_THRESHOLD
    ) -> Optional[str]:
        for known_hash, known_src in self.store.items():
            if known_src == source_key:
                continue  # same source -> not reuse; keeps re-runs idempotent
            if hamming(phash, known_hash) <= threshold:
                return known_src
        return None

    def add(self, phash: str, source_key: str) -> None:
        self.store.setdefault(phash, source_key)  # keep the first source for this content

    def save(self) -> None:
        import json

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.store, f)


def scan_transcript_for_injection(text: str) -> List[str]:
    hits = []
    for pat, rx in zip(INJECTION_PATTERNS, _INJECTION_RE):
        if rx.search(text or ""):
            hits.append(pat)
    return hits


def load_requirements(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_requirement(claim_object: str, issue_type: str, requirements: List[dict]) -> dict:
    by_id = {r["requirement_id"]: r for r in requirements}
    rid = None
    if claim_object == "car":
        if issue_type in ("dent", "scratch"):
            rid = "REQ_CAR_BODY_PANEL"
        elif issue_type in ("crack", "broken_part", "missing_part", "glass_shatter"):
            rid = "REQ_CAR_GLASS_LIGHT_MIRROR"
    elif claim_object == "laptop":
        if issue_type in ("hinge",) or issue_type in ("broken_part", "missing_part"):
            rid = "REQ_LAPTOP_BODY_HINGE_PORT"
        else:
            rid = "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"
    elif claim_object == "package":
        if issue_type in ("crushed_packaging", "torn_packaging"):
            rid = "REQ_PACKAGE_EXTERIOR"
        elif issue_type in ("water_damage", "stain"):
            rid = "REQ_PACKAGE_LABEL_OR_STAIN"
        elif issue_type in ("missing_part",):
            rid = "REQ_PACKAGE_CONTENTS"
    if rid not in by_id:
        rid = "REQ_GENERAL_OBJECT_PART"
    return by_id.get(rid, requirements[0])


@dataclass
class ClaimPregateResult:
    image_signals: List[ImageSignals] = field(default_factory=list)
    duplicate_pairs: list = field(default_factory=list)
    reused_image_ids: List[str] = field(default_factory=list)
    injection_hits: List[str] = field(default_factory=list)
    has_usable_image: bool = False


def run_claim_pregates(
    image_paths: List[str],
    image_ids: List[str],
    transcript: str,
    phash_store: Optional[PHashStore] = None,
) -> ClaimPregateResult:
    res = ClaimPregateResult()
    for path, iid in zip(image_paths, image_ids):
        res.image_signals.append(analyze_image(path, iid))
    res.duplicate_pairs = find_duplicate_pairs(res.image_signals)
    res.injection_hits = scan_transcript_for_injection(transcript)
    res.has_usable_image = any(s.exists and not s.blurry for s in res.image_signals) or any(
        s.exists for s in res.image_signals
    )
    if phash_store is not None:
        for s in res.image_signals:
            if not s.phash:
                continue
            # Source key = the file path (unique per file), so a re-run self-matches and is
            # skipped; only the SAME content under a DIFFERENT path counts as reuse fraud.
            if phash_store.match(s.phash, s.path) is not None:
                res.reused_image_ids.append(s.image_id)
            phash_store.add(s.phash, s.path)
    return res
