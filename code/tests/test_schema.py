import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> code/

from pipeline.schema import OUTPUT_COLUMNS, OutputRow  # noqa: E402
from pipeline.validate import read_labeled_rows  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_CSV = os.path.join(REPO_ROOT, "dataset", "sample_claims.csv")


def _base(**overrides):
    row = dict(
        user_id="user_001",
        image_paths="images/test/case_001/img_1.jpg",
        user_claim="Customer: dent on rear bumper.",
        claim_object="car",
        evidence_standard_met=True,
        evidence_standard_met_reason="Rear bumper visible; dent verifiable.",
        risk_flags="none",
        issue_type="dent",
        object_part="rear_bumper",
        claim_status="supported",
        claim_status_justification="img_1 shows a dent on the rear bumper.",
        supporting_image_ids="img_1",
        valid_image=True,
        severity="medium",
    )
    row.update(overrides)
    return row


def _raises(**overrides):
    try:
        OutputRow(**_base(**overrides))
    except Exception:
        return True
    return False


def test_valid_supported_row_serializes_lowercase_bools():
    r = OutputRow(**_base())
    d = r.to_csv_dict()
    assert list(d.keys()) == OUTPUT_COLUMNS
    assert d["evidence_standard_met"] == "true"
    assert d["valid_image"] == "true"
    assert d["risk_flags"] == "none"
    assert d["supporting_image_ids"] == "img_1"


def test_valid_image_true_with_evidence_false_is_legal():
    # Clear photo of the WRONG part: usable for review, but not sufficient to decide.
    r = OutputRow(**_base(
        valid_image=True, evidence_standard_met=False,
        claim_status="not_enough_information", supporting_image_ids="none",
        risk_flags="wrong_object_part",
    ))
    assert r.valid_image is True and r.evidence_standard_met is False


def test_evidence_false_must_be_nei():
    assert _raises(evidence_standard_met=False, claim_status="supported")


def test_supported_requires_supporting_ids():
    assert _raises(claim_status="supported", supporting_image_ids="none")


def test_nei_requires_no_supporting_ids():
    assert _raises(
        evidence_standard_met=False, claim_status="not_enough_information",
        supporting_image_ids="img_1",
    )


def test_issue_none_cannot_be_supported():
    assert _raises(issue_type="none")


def test_object_part_must_match_object():
    assert _raises(claim_object="car", object_part="screen")  # screen is laptop-only


def test_laptop_part_ok():
    r = OutputRow(**_base(claim_object="laptop", object_part="screen", issue_type="crack"))
    assert r.object_part == "screen"


def test_invalid_image_with_sufficient_evidence_is_legal():
    # Golden-set case (sample user_008): a non-original/manipulated image whose content
    # still contradicts the claim. valid_image (authenticity) and evidence_standard_met
    # (content sufficiency) are independent axes; this combo MUST be allowed.
    r = OutputRow(**_base(
        valid_image=False, evidence_standard_met=True, claim_status="contradicted",
        issue_type="broken_part", object_part="front_bumper",
        risk_flags="claim_mismatch;non_original_image;manual_review_required",
        supporting_image_ids="img_1",
    ))
    assert r.valid_image is False and r.evidence_standard_met is True


def test_risk_flags_parse_and_dedup():
    r = OutputRow(**_base(risk_flags="blurry_image;manual_review_required;blurry_image"))
    assert r.to_csv_dict()["risk_flags"] == "blurry_image;manual_review_required"


def test_risk_flags_none_cannot_mix():
    assert _raises(risk_flags="none;blurry_image")


def test_illegal_enum_rejected():
    assert _raises(issue_type="smashed")


def test_multi_supporting_ids_join():
    r = OutputRow(**_base(supporting_image_ids="img_1;img_2"))
    assert r.to_csv_dict()["supporting_image_ids"] == "img_1;img_2"


def test_all_sample_rows_are_valid():
    # The 20 labeled golden rows MUST pass our schema unchanged. If one fails, our
    # schema disagrees with the labelers' conventions — a real finding, not a nit.
    rows = read_labeled_rows(SAMPLE_CSV)
    assert len(rows) == 20


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
