from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pregates import ClaimPregateResult, ImageSignals
from .schema import (
    OBJECT_PARTS,
    ClaimObject,
    ClaimStatus,
    InputRow,
    IssueType,
    OutputRow,
    RiskFlag,
    Severity,
)
from .stages import ClaimAdjudication, InspectedImage, StageBundle, VisualQualityIssue

MANUAL_REVIEW_CONFIDENCE = 0.55

_SEVERITY_RANK = {Severity.none: 0, Severity.low: 1, Severity.medium: 2, Severity.high: 3}
_RISK_ORDER = list(RiskFlag)
_DAMAGE_ISSUES = {
    IssueType.dent, IssueType.scratch, IssueType.crack, IssueType.glass_shatter,
    IssueType.broken_part, IssueType.missing_part, IssueType.torn_packaging,
    IssueType.crushed_packaging, IssueType.water_damage, IssueType.stain,
}


def load_user_history(path: str) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["user_id"]: r for r in csv.DictReader(f)}


def parse_history_flags(record: Optional[dict]) -> List[str]:
    if not record:
        return []
    raw = (record.get("history_flags") or "").strip()
    if not raw or raw == "none":
        return []
    return [p.strip() for p in raw.split(";") if p.strip() and p.strip() != "none"]


@dataclass
class FusedResult:
    output: OutputRow
    audit: dict = field(default_factory=dict)


def _sig_by_id(pregates: ClaimPregateResult) -> Dict[str, ImageSignals]:
    return {s.image_id: s for s in pregates.image_signals}


def _is_authentic(sig: Optional[ImageSignals], reused: set) -> bool:
    # Calibrated from the batch scan: missing-EXIF is the NORM (22/29 imgs), so it must NOT
    # imply manipulation. Only an editor-software signature or a pHash reuse are reliable.
    if sig is None or not sig.exists:
        return False
    if sig.image_id in reused:
        return False
    if sig.editor_software:
        return False
    return True


def _quality_usable(sig: Optional[ImageSignals], ins: Optional[InspectedImage]) -> bool:
    if sig is None or not sig.exists:
        return False
    if sig.blurry:
        return False
    if ins is not None and VisualQualityIssue.cropped_or_obstructed in ins.inspection.quality_issues:
        return False
    return True


def _max_visible_severity(inspections: List[InspectedImage]) -> Severity:
    best = None
    for ins in inspections:
        sev = ins.inspection.visible_severity
        if sev in _SEVERITY_RANK and (best is None or _SEVERITY_RANK[sev] > _SEVERITY_RANK[best]):
            best = sev
    return best or Severity.unknown


def fuse(
    row: InputRow,
    bundle: StageBundle,
    pregates: ClaimPregateResult,
    history_flags: List[str],
) -> FusedResult:
    reused = set(pregates.reused_image_ids)
    sigs = _sig_by_id(pregates)
    inspections = bundle.inspections
    allowed_parts = OBJECT_PARTS[row.claim_object]
    flags: List[RiskFlag] = []

    def add(flag: RiskFlag):
        if flag not in flags:
            flags.append(flag)

    # --- image-quality + authenticity flags: surfaced from ALL images regardless of verdict ---
    any_editor = any(s.editor_software for s in pregates.image_signals)
    for s in pregates.image_signals:
        if s.blurry:
            add(RiskFlag.blurry_image)
        if s.low_light_or_glare:
            add(RiskFlag.low_light_or_glare)
    for ins in inspections:
        for q in ins.inspection.quality_issues:
            add(RiskFlag(q.value))
        if ins.inspection.instruction_text_in_image:
            add(RiskFlag.text_instruction_present)
    if pregates.injection_hits:
        add(RiskFlag.text_instruction_present)
    if reused:
        add(RiskFlag.non_original_image)
    if any_editor or reused:
        add(RiskFlag.possible_manipulation)

    # valid_image is OWNED here, not by the VLM: it is about image USABILITY/AUTHENTICITY as
    # evidence, independent of whether the images answer the claim.
    # A clear, genuine photo of the wrong part is still valid_image=true.
    valid_image = any(
        _is_authentic(sigs.get(ins.image_id), reused) and _quality_usable(sigs.get(ins.image_id), ins)
        for ins in inspections
    )

    adj: Optional[ClaimAdjudication] = bundle.adjudication
    flip_applied = False

    if adj is None:
        # No usable images / VLM failed -> safe default. Never auto-decide a blind claim.
        asserted_part = bundle.extraction.asserted_object_part.value
        object_part = asserted_part if asserted_part in allowed_parts else "unknown"
        status = ClaimStatus.not_enough_information
        issue_type = IssueType.unknown
        evidence_met = False
        evidence_reason = "No usable image evidence was available for automated review."
        justification = "Insufficient usable image evidence to evaluate the claim; routed to manual review."
        supporting: List[str] = []
        severity = Severity.unknown
        add(RiskFlag.damage_not_visible)
        confidence = 0.0
    else:
        status = adj.claim_status
        issue_type = adj.issue_type
        object_part = adj.object_part.value
        evidence_met = adj.evidence_standard_met
        evidence_reason = adj.evidence_standard_met_reason
        justification = adj.claim_status_justification
        supporting = list(adj.supporting_image_ids)
        severity = adj.severity
        confidence = adj.confidence

        usable = [ins for ins in inspections if _quality_usable(sigs.get(ins.image_id), ins)]

        # Wrong-object -> contradicted flip: if every usable image clearly shows a DIFFERENT
        # object than claimed, that is a visible contradiction, not an abstention.
        if (
            status == ClaimStatus.not_enough_information
            and usable
            and all(not ins.inspection.shows_claimed_object for ins in usable)
        ):
            flip_applied = True
            status = ClaimStatus.contradicted
            evidence_met = True
            issue_type = IssueType.unknown
            object_part = "unknown"
            severity = _max_visible_severity(usable)
            supporting = [usable[0].image_id]
            add(RiskFlag.wrong_object)
            add(RiskFlag.claim_mismatch)

        # Normalize evidence/severity/issue by status to match labeler conventions.
        if status == ClaimStatus.not_enough_information:
            evidence_met = False
            issue_type = IssueType.unknown
            severity = Severity.unknown
            supporting = []
        else:
            evidence_met = True

        # Derive the contradiction/insufficiency reason flags from what the VLM observed.
        asserted_is_damage = bundle.extraction.asserted_issue_type in _DAMAGE_ISSUES
        if status == ClaimStatus.contradicted:
            if any(not ins.inspection.shows_claimed_object for ins in usable):
                add(RiskFlag.wrong_object)
                add(RiskFlag.claim_mismatch)
            elif issue_type in (IssueType.none, IssueType.unknown):
                add(RiskFlag.damage_not_visible)
            elif not adj.visible_issue_matches_claim:
                add(RiskFlag.claim_mismatch)
        elif status == ClaimStatus.not_enough_information:
            part_unseen = all(not ins.inspection.claimed_part_in_frame for ins in inspections) if inspections else True
            obj_seen = any(ins.inspection.shows_claimed_object for ins in inspections)
            if part_unseen and obj_seen and RiskFlag.cropped_or_obstructed not in flags:
                add(RiskFlag.wrong_angle)
            if asserted_is_damage:
                add(RiskFlag.damage_not_visible)

        if issue_type == IssueType.none:
            severity = Severity.none

    if object_part not in allowed_parts:
        object_part = "unknown"

    # Authenticity overrides usability: a non-original / manipulated set is not valid evidence.
    if RiskFlag.non_original_image in flags or RiskFlag.possible_manipulation in flags:
        valid_image = False

    # --- history fusion: DETERMINISTIC. Output flags ⊇ history flags; history only ADDS,
    # it never flips a clear visual verdict (a supported claim stays supported). ---
    for hf in history_flags:
        try:
            add(RiskFlag(hf))
        except ValueError:
            pass
    if RiskFlag.user_history_risk in flags:
        add(RiskFlag.manual_review_required)

    # --- confidence-gated escalation: route the hardest cases, never auto-decide them. ---
    escalation_reasons = []
    if adj is None:
        escalation_reasons.append("no_usable_evidence")
    if confidence < MANUAL_REVIEW_CONFIDENCE and adj is not None:
        escalation_reasons.append("low_confidence")
    if RiskFlag.text_instruction_present in flags:
        escalation_reasons.append("injection")
    if RiskFlag.possible_manipulation in flags or RiskFlag.non_original_image in flags:
        escalation_reasons.append("authenticity")
    if status == ClaimStatus.contradicted and RiskFlag.user_history_risk in flags:
        escalation_reasons.append("contradicted_high_risk_user")
    # Self-contradiction guardrail: the model called it supported yet reported the visible issue
    # does NOT match the claim. That is exactly the false-confirm failure mode — don't auto-decide
    # it, route to a human. (The verdict itself stays the model's; code only adds the leash.)
    if adj is not None and status == ClaimStatus.supported and not adj.visible_issue_matches_claim:
        add(RiskFlag.claim_mismatch)
        escalation_reasons.append("supported_claim_mismatch")
    if escalation_reasons:
        add(RiskFlag.manual_review_required)

    ordered = [f for f in _RISK_ORDER if f in flags]

    output = OutputRow(
        user_id=row.user_id,
        image_paths=row.image_paths,
        user_claim=row.user_claim,
        claim_object=row.claim_object,
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=ordered,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=status,
        claim_status_justification=justification,
        supporting_image_ids=supporting,
        valid_image=valid_image,
        severity=severity,
    )

    audit = {
        "requirement_id": bundle.requirement_id,
        "prompt_versions": bundle.prompt_versions,
        "adjudication_confidence": confidence,
        "valid_image_basis": "deterministic(authentic+quality)",
        "wrong_object_flip": flip_applied,
        "escalation_reasons": escalation_reasons,
        "history_flags_in": list(history_flags),
    }
    return FusedResult(output=output, audit=audit)
