import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> code/

import pytest  # noqa: E402

from pipeline.schema import ClaimObject  # noqa: E402
from pipeline.stages import (  # noqa: E402
    ADJUDICATION_PROMPT_VERSION,
    ClaimAdjudication,
    ClaimExtraction,
    ImageInspection,
    _adjudication_validator,
)
from pipeline.vlm import strict_json_schema  # noqa: E402

STAGE_MODELS = [ClaimExtraction, ImageInspection, ClaimAdjudication]


def _walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for v in node.values():
            yield from _walk_objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_objects(v)


def test_strict_schema_every_object_closed_and_fully_required():
    for model in STAGE_MODELS:
        schema = strict_json_schema(model)
        objs = list(_walk_objects(schema))
        assert objs, f"{model.__name__} produced no object nodes"
        for obj in objs:
            assert obj["additionalProperties"] is False
            assert set(obj["required"]) == set(obj["properties"].keys())


def test_strict_schema_drops_unsupported_keywords():
    for model in STAGE_MODELS:
        schema = strict_json_schema(model)
        text = repr(schema)
        for banned in ("'title'", "'default'", "'minimum'", "'maxLength'"):
            assert banned not in text, f"{model.__name__} schema still has {banned}"


def test_strict_schema_constrains_enums():
    schema = strict_json_schema(ClaimAdjudication)
    defs = schema.get("$defs", {})
    statuses = next(d["enum"] for d in defs.values() if set(d.get("enum", [])) >= {"supported", "contradicted"})
    assert set(statuses) == {"supported", "contradicted", "not_enough_information"}


def _valid_adjudication(**over):
    base = dict(
        evidence_standard_met=True,
        evidence_standard_met_reason="part visible",
        issue_type="dent",
        object_part="door",
        claim_status="supported",
        claim_status_justification="img_1 shows a door dent",
        supporting_image_ids=["img_1"],
        visible_issue_matches_claim=True,
        valid_image=True,
        severity="medium",
        confidence=0.8,
    )
    base.update(over)
    return base


def test_adjudication_cross_field_invariants():
    bad_cases = [
        dict(evidence_standard_met=False, claim_status="supported"),
        dict(claim_status="supported", supporting_image_ids=[]),
        dict(claim_status="contradicted", supporting_image_ids=[]),
        dict(claim_status="not_enough_information", supporting_image_ids=["img_1"]),
        dict(issue_type="none", claim_status="supported"),
        dict(confidence=1.4),
    ]
    for over in bad_cases:
        with pytest.raises(Exception):
            ClaimAdjudication.model_validate(_valid_adjudication(**over))


def test_adjudication_validator_rejects_illegal_part_for_object():
    validate = _adjudication_validator(ClaimObject.car, ["img_1"])
    with pytest.raises(Exception):
        validate(_valid_adjudication(object_part="screen"))  # laptop part on a car
    assert validate(_valid_adjudication(object_part="door")).object_part.value == "door"


def test_adjudication_validator_rejects_unknown_support_id():
    validate = _adjudication_validator(ClaimObject.car, ["img_1"])
    with pytest.raises(Exception):
        validate(_valid_adjudication(supporting_image_ids=["img_9"]))


def test_nei_path_validates():
    validate = _adjudication_validator(ClaimObject.car, ["img_1"])
    adj = validate(
        _valid_adjudication(
            evidence_standard_met=False,
            claim_status="not_enough_information",
            supporting_image_ids=[],
            issue_type="unknown",
            object_part="headlight",
            severity="unknown",
            confidence=0.3,
        )
    )
    assert adj.claim_status.value == "not_enough_information"


def test_prompt_versions_present():
    assert ADJUDICATION_PROMPT_VERSION
    assert ImageInspection.model_fields["visible_description"].description


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
