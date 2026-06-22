from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ClaimObject(str, Enum):
    car = "car"
    laptop = "laptop"
    package = "package"


class ClaimStatus(str, Enum):
    supported = "supported"
    contradicted = "contradicted"
    not_enough_information = "not_enough_information"


class IssueType(str, Enum):
    dent = "dent"
    scratch = "scratch"
    crack = "crack"
    glass_shatter = "glass_shatter"
    broken_part = "broken_part"
    missing_part = "missing_part"
    torn_packaging = "torn_packaging"
    crushed_packaging = "crushed_packaging"
    water_damage = "water_damage"
    stain = "stain"
    none = "none"
    unknown = "unknown"


class Severity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class RiskFlag(str, Enum):
    none = "none"
    blurry_image = "blurry_image"
    cropped_or_obstructed = "cropped_or_obstructed"
    low_light_or_glare = "low_light_or_glare"
    wrong_angle = "wrong_angle"
    wrong_object = "wrong_object"
    wrong_object_part = "wrong_object_part"
    damage_not_visible = "damage_not_visible"
    claim_mismatch = "claim_mismatch"
    possible_manipulation = "possible_manipulation"
    non_original_image = "non_original_image"
    text_instruction_present = "text_instruction_present"
    user_history_risk = "user_history_risk"
    manual_review_required = "manual_review_required"


OBJECT_PARTS = {
    ClaimObject.car: {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    ClaimObject.laptop: {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    },
    ClaimObject.package: {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

ALL_OBJECT_PARTS = sorted({p for parts in OBJECT_PARTS.values() for p in parts})

ObjectPart = Enum("ObjectPart", {p: p for p in ALL_OBJECT_PARTS}, type=str)


OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]


def _bool_to_token(b: bool) -> str:
    return "true" if b else "false"


def image_ids_from_paths(image_paths: str) -> List[str]:
    ids = []
    for raw in image_paths.split(";"):
        p = raw.strip().replace("\\", "/")
        if not p:
            continue
        name = p.rsplit("/", 1)[-1]
        ids.append(name.rsplit(".", 1)[0] if "." in name else name)
    return ids


class InputRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObject

    @property
    def image_ids(self) -> List[str]:
        return image_ids_from_paths(self.image_paths)


class OutputRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: ClaimObject
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: List[RiskFlag]
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: List[str]
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags", mode="before")
    @classmethod
    def _parse_risk_flags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(";") if p.strip()]
            if not parts or parts == ["none"]:
                return []
            if "none" in parts:
                raise ValueError("'none' cannot be combined with other risk_flags")
            return parts
        return v

    @field_validator("risk_flags", mode="after")
    @classmethod
    def _normalize_risk_flags(cls, v):
        if any(f == RiskFlag.none for f in v):
            raise ValueError("RiskFlag.none must not be stored; 'no risk' is []")
        out = []
        for f in v:
            if f not in out:
                out.append(f)
        return out

    @field_validator("supporting_image_ids", mode="before")
    @classmethod
    def _parse_support_ids(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(";") if p.strip()]
            if not parts or parts == ["none"]:
                return []
            return parts
        return v

    @model_validator(mode="after")
    def _check_object_part(self):
        allowed = OBJECT_PARTS[self.claim_object]
        if self.object_part not in allowed:
            raise ValueError(
                f"object_part '{self.object_part}' invalid for claim_object "
                f"'{self.claim_object.value}'; allowed: {sorted(allowed)}"
            )
        return self

    @model_validator(mode="after")
    def _cross_field_rules(self):
        if self.evidence_standard_met is False and self.claim_status != ClaimStatus.not_enough_information:
            raise ValueError("evidence_standard_met=false requires claim_status=not_enough_information")
        if self.claim_status == ClaimStatus.supported and not self.supporting_image_ids:
            raise ValueError("claim_status=supported requires non-empty supporting_image_ids")
        if self.claim_status == ClaimStatus.not_enough_information and self.supporting_image_ids:
            raise ValueError("claim_status=not_enough_information requires supporting_image_ids=none")
        if self.issue_type == IssueType.none and self.claim_status == ClaimStatus.supported:
            raise ValueError("issue_type=none cannot yield claim_status=supported")
        return self

    def to_csv_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object.value,
            "evidence_standard_met": _bool_to_token(self.evidence_standard_met),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": ";".join(f.value for f in self.risk_flags) if self.risk_flags else "none",
            "issue_type": self.issue_type.value,
            "object_part": self.object_part,
            "claim_status": self.claim_status.value,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": ";".join(self.supporting_image_ids) if self.supporting_image_ids else "none",
            "valid_image": _bool_to_token(self.valid_image),
            "severity": self.severity.value,
        }
