from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from .fusion import FusedResult, fuse
from .pregates import (
    ClaimPregateResult,
    PHashStore,
    load_requirements,
    run_claim_pregates,
    scan_transcript_for_injection,
    select_requirement,
)
from .schema import InputRow, IssueType, ObjectPart, OutputRow, image_ids_from_paths
from .stages import (
    ADJUDICATION_PROMPT_VERSION,
    EXTRACTION_PROMPT_VERSION,
    INSPECTION_PROMPT_VERSION,
    ClaimExtraction,
    StageBundle,
    adjudicate,
    extract_claim,
    inspect_image,
    transcribe_image_text,
)
from .vlm import VLMClient

_DEFAULT_EXTRACTION = ClaimExtraction(
    claim_summary="",
    asserted_issue_type=IssueType.unknown,
    asserted_object_part=ObjectPart("unknown"),
    multiple_parts_claimed=False,
    instruction_text_in_transcript=False,
)
_VERSIONS = {
    "extraction": EXTRACTION_PROMPT_VERSION,
    "inspection": INSPECTION_PROMPT_VERSION,
    "adjudication": ADJUDICATION_PROMPT_VERSION,
}


@dataclass
class ProcessedClaim:
    output: OutputRow
    pregates: ClaimPregateResult
    bundle: StageBundle
    audit: dict


def _safe(fn: Callable, default=None):
    try:
        return fn()
    except Exception:
        return default


def resolve_paths(row: InputRow, image_root: str):
    ids = image_ids_from_paths(row.image_paths)
    paths = []
    for raw in row.image_paths.split(";"):
        raw = raw.strip()
        if raw:
            paths.append(os.path.join(image_root, raw))
    return paths, ids


def process_claim(
    client: VLMClient,
    row: InputRow,
    requirements: List[dict],
    image_root: str,
    history_flags: Optional[List[str]] = None,
    phash_store: Optional[PHashStore] = None,
    bundle_fn: Optional[Callable[..., StageBundle]] = None,
) -> ProcessedClaim:
    """Full per-claim pipeline: pregates -> scoped VLM stages -> deterministic fusion.
    Resilient by design — any stage failure degrades to a safe-default (abstain + escalate)
    rather than crashing the batch. `bundle_fn` lets the eval swap in Strategy A."""
    paths, ids = resolve_paths(row, image_root)
    pregates = run_claim_pregates(paths, ids, row.user_claim, phash_store)
    existing = [(p, i) for p, i, s in zip(paths, ids, pregates.image_signals) if s.exists]

    if bundle_fn is not None:
        bundle = bundle_fn(client, row, existing, requirements)
    else:
        bundle = _build_bundle(client, row, existing, requirements)

    # In-image prompt-injection defense: OCR each image, treat the text as untrusted DATA, and
    # run the SAME injection regex as the transcript. Folds into pregates.injection_hits so the
    # existing fusion path raises text_instruction_present + manual_review. Resilient + opt-out.
    if getattr(client.cfg, "ocr_injection_scan", False):
        ocr_hit_ids = []
        for path, iid in existing:
            text = _safe(lambda p=path: transcribe_image_text(client, p), "")
            if text and scan_transcript_for_injection(text):
                ocr_hit_ids.append(iid)
        if ocr_hit_ids:
            pregates.injection_hits = list(pregates.injection_hits) + [
                f"in_image:{i}" for i in ocr_hit_ids
            ]

    fused: FusedResult = fuse(row, bundle, pregates, history_flags or [])
    return ProcessedClaim(output=fused.output, pregates=pregates, bundle=bundle, audit=fused.audit)


def _build_bundle(client, row, existing, requirements) -> StageBundle:
    versions = dict(_VERSIONS)
    versions["adjudication"] = getattr(
        client.cfg, "adjudication_prompt_version", ADJUDICATION_PROMPT_VERSION
    )
    extraction = _safe(lambda: extract_claim(client, row), _DEFAULT_EXTRACTION)
    requirement = select_requirement(
        row.claim_object.value, extraction.asserted_issue_type.value, requirements
    )
    inspections = []
    for path, iid in existing:
        ins = _safe(lambda p=path, i=iid: inspect_image(client, p, i, row, extraction))
        if ins is not None:
            inspections.append(ins)
    adjudication = None
    if inspections:
        adjudication = _safe(
            lambda: adjudicate(
                client, row, inspections, requirement["minimum_image_evidence"], extraction
            )
        )
    return StageBundle(
        extraction=extraction,
        inspections=inspections,
        adjudication=adjudication,
        requirement_id=requirement["requirement_id"],
        prompt_versions=versions,
    )


def load_requirements_for(dataset_dir: str) -> List[dict]:
    return load_requirements(os.path.join(dataset_dir, "evidence_requirements.csv"))
