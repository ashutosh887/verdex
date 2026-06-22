"""LLM-as-judge for the free-text `claim_status_justification`.

This is the one quality axis the field-accuracy metrics can't see: *is the justification a good,
image-grounded explanation* — not just whether the verdict label matched. We score it with a
decomposed rubric and explicitly control for the well-known judge biases.

Design choices (each is a defensible interview point):
- **Reference-based, single-answer rubric** (not pairwise A/B). Pairwise judging has a documented
  *position bias*; scoring one answer against an expert reference sidesteps it entirely.
- **Decomposed rubric** (groundedness / faithfulness / verdict-consistency / relevance) rather than
  one vague "quality" score — decomposition is more reliable and more diagnosable.
- **Named-bias controls in the prompt:** the judge is told to ignore answer *length*, writing
  *style/verbosity*, and any *stated confidence*, and to score substance only. These are the biases
  LLM judges are known to have (verbosity bias, style bias, self-confidence bias).
- The judge never sees which system produced the candidate (no leakage of "this is the production
  system"), and gets the gold reference as the evidential anchor.

Runs offline over the golden set; pipeline predictions come from the response cache (free). The
judge calls themselves are the only spend (~20 calls on the judge model).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL = os.path.dirname(os.path.abspath(__file__))
for _p in (_CODE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402

from pipeline.fusion import load_user_history, parse_history_flags  # noqa: E402
from pipeline.run import load_requirements_for, process_claim  # noqa: E402
from pipeline.schema import INPUT_COLUMNS, InputRow  # noqa: E402
from pipeline.vlm import make_client, strict_json_schema  # noqa: E402

_REPO = os.path.dirname(_CODE)
_DIMS = ["groundedness", "faithfulness", "verdict_consistency", "claim_relevance"]


class JustificationJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groundedness: int = Field(description="1-5: does the explanation reference concrete visible image evidence (parts, damage, image IDs) rather than vague assertion?")
    faithfulness: int = Field(description="1-5: does it avoid claiming specifics beyond what the evidence supports (no invented detail)?")
    verdict_consistency: int = Field(description="1-5: does the stated reasoning actually support the candidate's verdict label?")
    claim_relevance: int = Field(description="1-5: does it address the specific part and issue the customer claimed?")
    rationale: str = Field(description="One sentence on the main strength or weakness. Do not mention length or writing style.")

    @field_validator(*_DIMS)
    @classmethod
    def _range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("scores must be 1-5")
        return v


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of damage-claim justification text. You are given an EXPERT "
    "REFERENCE justification (the evidential ground truth) and a CANDIDATE justification, and you rate "
    "the candidate on a decomposed rubric, 1-5 per dimension.\n"
    "Control for bias explicitly: IGNORE the candidate's length, writing style, verbosity, and any "
    "stated confidence. A short, plain, correct justification must score as high as a long, polished "
    "one. Reward only substance: grounding in visible evidence, faithfulness to that evidence, "
    "internal consistency with its own verdict, and relevance to the claimed part/issue. Different "
    "wording from the reference is fine — judge meaning, not phrasing."
)


def _judge_user(row, gold_just, gold_verdict, cand_just, cand_verdict) -> str:
    return (
        f"claim_object: {row.claim_object.value}\n"
        f"customer claim (untrusted data): {row.user_claim}\n\n"
        f"EXPERT REFERENCE verdict: {gold_verdict}\n"
        f"EXPERT REFERENCE justification: {gold_just}\n\n"
        f"CANDIDATE verdict: {cand_verdict}\n"
        f"CANDIDATE justification: {cand_just}\n\n"
        "Score the CANDIDATE justification on the rubric."
    )


def run_judge(dataset: str, limit, offline: bool):
    golden = list(csv.DictReader(open(os.path.join(dataset, "sample_claims.csv"), encoding="utf-8")))
    if limit:
        golden = golden[:limit]
    requirements = load_requirements_for(dataset)
    history = load_user_history(os.path.join(dataset, "user_history.csv"))

    client = make_client()
    client.offline = offline
    schema = strict_json_schema(JustificationJudgment)

    rows = []
    for g in golden:
        row = InputRow(**{c: g[c] for c in INPUT_COLUMNS})
        hflags = parse_history_flags(history.get(row.user_id))
        pc = process_claim(client, row, requirements, dataset, hflags)  # cached -> free
        out = pc.output.to_csv_dict()
        verdict_correct = out["claim_status"] == g["claim_status"]
        judgment = client.infer(
            system=JUDGE_SYSTEM,
            user_content=_judge_user(
                row, g["claim_status_justification"], g["claim_status"],
                out["claim_status_justification"], out["claim_status"],
            ),
            images=None,
            json_schema=schema,
            role="judge",
            validate=JustificationJudgment.model_validate,
        )
        rows.append({
            "user_id": row.user_id,
            "verdict_correct": verdict_correct,
            "scores": {d: getattr(judgment, d) for d in _DIMS},
            "rationale": judgment.rationale,
        })
    return rows, client.cost_estimate()


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def render(rows, cost) -> str:
    L = ["# LLM-as-judge — justification quality (golden set)", ""]
    L.append("Decomposed rubric, reference-based single-answer scoring (no pairwise position bias). The "
             "judge is instructed to **ignore length, style, and stated confidence** — the named verbosity / "
             "style / self-confidence biases — and score substance only. Judge model: see `config.yaml` "
             "(`model_judge`). Pipeline predictions are served from cache; only the judge calls bill.")
    L += ["", "## Mean rubric scores (1-5)", "", "| dimension | all | when verdict correct | when verdict wrong |", "|---|---|---|---|"]
    correct = [r for r in rows if r["verdict_correct"]]
    wrong = [r for r in rows if not r["verdict_correct"]]
    for d in _DIMS:
        L.append(f"| {d} | {_mean([r['scores'][d] for r in rows])} | "
                 f"{_mean([r['scores'][d] for r in correct])} | "
                 f"{_mean([r['scores'][d] for r in wrong])} |")
    allmean = _mean([sum(r["scores"].values()) / len(_DIMS) for r in rows])
    L += ["", f"**Overall mean (all dimensions): {allmean}/5** over {len(rows)} justifications "
          f"({len(correct)} on correct verdicts, {len(wrong)} on wrong).",
          "",
          "Reading it: a high *groundedness/faithfulness* even where the verdict is wrong means the "
          "justifications are honest about the evidence (they explain what was seen) — the failures are "
          "perception/label-boundary calls, not fabricated reasoning. A drop in *verdict_consistency* on "
          "wrong rows would instead flag motivated/after-the-fact reasoning.",
          "", "## Per-claim", ""]
    for r in rows:
        s = r["scores"]
        L.append(f"- **{r['user_id']}** ({'✓' if r['verdict_correct'] else '✗'} verdict): "
                 f"g{s['groundedness']} f{s['faithfulness']} c{s['verdict_consistency']} r{s['claim_relevance']} "
                 f"— {r['rationale']}")
    c = cost
    L += ["", "## Cost", "",
          f"- Judge calls billed: **{c['billed_calls']}** ({c['cached_calls']} pipeline calls served from cache).",
          f"- Tokens: {c['input_tokens']:,} in / {c['output_tokens']:,} out · est **${c['est_cost_usd']}**."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge for justification quality on the golden set.")
    ap.add_argument("--dataset", default=os.path.join(_REPO, "dataset"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="cache-only (judge calls must already be cached)")
    ap.add_argument("--out", default=os.path.join(_EVAL, "evaluation_report_judge.md"))
    args = ap.parse_args()
    rows, cost = run_judge(args.dataset, args.limit, args.offline)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render(rows, cost) + "\n")
    allmean = _mean([sum(r["scores"].values()) / len(_DIMS) for r in rows])
    print(f"judged {len(rows)} justifications  overall={allmean}/5  "
          f"cost=${cost['est_cost_usd']} ({cost['billed_calls']} billed)  -> {args.out}")


if __name__ == "__main__":
    main()
