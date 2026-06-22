import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> code/

import pytest  # noqa: E402

from pipeline.fusion import fuse, parse_history_flags  # noqa: E402
from pipeline.pregates import ClaimPregateResult, ImageSignals  # noqa: E402
from pipeline.schema import (  # noqa: E402
    ClaimObject,
    ClaimStatus,
    InputRow,
    IssueType,
    ObjectPart,
    RiskFlag,
    Severity,
)
from pipeline.stages import (  # noqa: E402
    ClaimAdjudication,
    ClaimExtraction,
    ImageInspection,
    InspectedImage,
    StageBundle,
    VisualQualityIssue,
)


def _row(obj="car", paths="images/test/c/img_1.jpg"):
    return InputRow(user_id="u1", image_paths=paths, user_claim="Customer: damage", claim_object=obj)


def _extraction(issue="dent", part="door", instr=False):
    return ClaimExtraction(
        claim_summary="x", asserted_issue_type=issue, asserted_object_part=ObjectPart(part),
        multiple_parts_claimed=False, instruction_text_in_transcript=instr,
    )


def _inspection(iid="img_1", shows=True, in_frame=True, issue="dent", part="door",
                sev="medium", quality=None, instr=False, usable=True, conf=0.9):
    ins = ImageInspection(
        visible_description="d", shows_claimed_object=shows, claimed_part_in_frame=in_frame,
        visible_issue_type=issue, visible_object_part=ObjectPart(part), visible_severity=sev,
        quality_issues=quality or [], instruction_text_in_image=instr, usable_for_review=usable,
        confidence=conf,
    )
    return InspectedImage(image_id=iid, path=f"/x/{iid}.jpg", inspection=ins)


def _adj(status="supported", issue="dent", part="door", evid=True, support=("img_1",),
         matches=True, valid=True, sev="medium", conf=0.9):
    return ClaimAdjudication(
        evidence_standard_met=evid, evidence_standard_met_reason="r", issue_type=issue,
        object_part=ObjectPart(part), claim_status=status, claim_status_justification="j",
        supporting_image_ids=list(support), visible_issue_matches_claim=matches, valid_image=valid,
        severity=sev, confidence=conf,
    )


def _sig(iid="img_1", exists=True, blurry=False, glare=False, editor=None, reused=False):
    return ImageSignals(
        image_id=iid, path=f"/x/{iid}.jpg", exists=exists, sha256="s", phash="p",
        blurry=blurry, low_light_or_glare=glare, missing_exif=True, editor_software=editor,
    )


def _pregates(sigs, reused=(), injection=()):
    return ClaimPregateResult(
        image_signals=list(sigs), duplicate_pairs=[], reused_image_ids=list(reused),
        injection_hits=list(injection), has_usable_image=True,
    )


def _bundle(extraction, inspections, adjudication):
    return StageBundle(
        extraction=extraction, inspections=list(inspections), adjudication=adjudication,
        requirement_id="REQ_CAR_BODY_PANEL", prompt_versions={"adjudication": "v1"},
    )


def test_supported_clean_no_flags():
    out = fuse(_row(), _bundle(_extraction(), [_inspection()], _adj()),
               _pregates([_sig()]), []).output
    assert out.claim_status == ClaimStatus.supported
    assert out.risk_flags == []
    assert out.valid_image is True


def test_history_only_adds_and_never_flips_verdict():
    # user_031: supported claim stays supported, history adds risk + manual review.
    out = fuse(_row(), _bundle(_extraction(), [_inspection()], _adj()),
               _pregates([_sig()]), ["user_history_risk"]).output
    assert out.claim_status == ClaimStatus.supported
    assert RiskFlag.user_history_risk in out.risk_flags
    assert RiskFlag.manual_review_required in out.risk_flags


def test_blurry_image_flag_but_still_supported():
    # user_003: one blurry image, decision rests on the clear one.
    pregates = _pregates([_sig("img_1", blurry=True), _sig("img_2")])
    bundle = _bundle(_extraction(), [_inspection("img_1", usable=False), _inspection("img_2")],
                     _adj(support=("img_2",)))
    out = fuse(_row(paths="a/img_1.jpg;a/img_2.jpg"), bundle, pregates, []).output
    assert RiskFlag.blurry_image in out.risk_flags
    assert out.claim_status == ClaimStatus.supported
    assert out.valid_image is True


def test_contradicted_by_absence_damage_not_visible():
    # user_020/034: claimed part visible & intact -> contradicted via damage_not_visible.
    adj = _adj(status="contradicted", issue="none", matches=False, sev="none")
    out = fuse(_row(), _bundle(_extraction(), [_inspection(issue="none", sev="none")], adj),
               _pregates([_sig()]), []).output
    assert out.claim_status == ClaimStatus.contradicted
    assert RiskFlag.damage_not_visible in out.risk_flags
    assert out.severity == Severity.none


def test_contradicted_by_difference_claim_mismatch():
    # user_005/008: different/lesser damage than claimed.
    adj = _adj(status="contradicted", issue="scratch", matches=False, sev="low")
    out = fuse(_row(), _bundle(_extraction(issue="dent"), [_inspection(issue="scratch", sev="low")], adj),
               _pregates([_sig()]), []).output
    assert RiskFlag.claim_mismatch in out.risk_flags


def test_nei_wrong_angle_part_out_of_frame():
    # user_006: object shown, claimed part not in frame.
    adj = _adj(status="not_enough_information", issue="unknown", evid=False, support=(), sev="unknown")
    bundle = _bundle(_extraction(issue="crack", part="headlight"),
                     [_inspection(in_frame=False, issue="unknown", part="body")], adj)
    out = fuse(_row(), bundle, _pregates([_sig()]), []).output
    assert out.claim_status == ClaimStatus.not_enough_information
    assert RiskFlag.wrong_angle in out.risk_flags
    assert RiskFlag.damage_not_visible in out.risk_flags
    assert out.valid_image is True  # clear photo of the wrong part is still valid evidence
    assert out.supporting_image_ids == []


def test_wrong_object_flip_to_contradicted():
    # user_033: a clearly different object than claimed -> flip nei to contradicted.
    adj = _adj(status="not_enough_information", issue="unknown", evid=False, support=(), sev="unknown")
    bundle = _bundle(_extraction(issue="crushed_packaging", part="box"),
                     [_inspection(shows=False, issue="crushed_packaging", part="box", sev="low")], adj)
    out = fuse(_row(obj="package", paths="a/img_1.jpg"), bundle, _pregates([_sig()]), []).output
    assert out.claim_status == ClaimStatus.contradicted
    assert RiskFlag.wrong_object in out.risk_flags
    assert RiskFlag.claim_mismatch in out.risk_flags
    assert out.supporting_image_ids == ["img_1"]


def test_non_original_forces_invalid_and_manual_review():
    # user_008: reused image -> non_original + possible_manipulation, valid_image false, escalate.
    out = fuse(_row(), _bundle(_extraction(), [_inspection()], _adj()),
               _pregates([_sig()], reused=["img_1"]), []).output
    assert RiskFlag.non_original_image in out.risk_flags
    assert RiskFlag.possible_manipulation in out.risk_flags
    assert out.valid_image is False
    assert RiskFlag.manual_review_required in out.risk_flags


def test_injection_sets_flag_and_escalates():
    out = fuse(_row(), _bundle(_extraction(), [_inspection(instr=True)], _adj()),
               _pregates([_sig()]), []).output
    assert RiskFlag.text_instruction_present in out.risk_flags
    assert RiskFlag.manual_review_required in out.risk_flags


def test_low_confidence_escalates():
    out = fuse(_row(), _bundle(_extraction(), [_inspection()], _adj(conf=0.3)),
               _pregates([_sig()]), []).output
    assert RiskFlag.manual_review_required in out.risk_flags


def test_safe_default_when_no_adjudication():
    bundle = _bundle(_extraction(part="door"), [], None)
    out = fuse(_row(), bundle, _pregates([]), []).output
    assert out.claim_status == ClaimStatus.not_enough_information
    assert out.evidence_standard_met is False
    assert out.valid_image is False
    assert RiskFlag.manual_review_required in out.risk_flags
    assert RiskFlag.damage_not_visible in out.risk_flags
    assert out.supporting_image_ids == []


def test_parse_history_flags():
    assert parse_history_flags(None) == []
    assert parse_history_flags({"history_flags": "none"}) == []
    assert parse_history_flags({"history_flags": "user_history_risk;manual_review_required"}) == [
        "user_history_risk", "manual_review_required"
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
