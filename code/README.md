# Multi-Modal Evidence Review

A pipeline that verifies damage claims (`car`, `laptop`, `package`) from submitted images, a chat transcript, the user's claim history, and a minimum-evidence checklist — producing one structured `output.csv` row per claim.

## Quality, defined for this task

Quality here is three things, in order:

1. **Correct `claim_status`** — `supported` / `contradicted` / `not_enough_information`.
2. **Calibrated abstention** — we return `not_enough_information` only when the evidence is *genuinely* insufficient, never as a dodge. Measured with a risk–coverage curve / AURC, not vibes.
3. **Image-grounded justifications** — every verdict quotes what is visible, and we trade this against cost & latency deliberately (see the strategy comparison).

**The non-negotiable invariant: images are the only source of truth.** User history can *add* `risk_flags` and route a case to human review, but it can **never** flip a clear visual verdict. That separation is structural (it lives in deterministic code), not a polite request in a prompt.

## Architecture

```
claims.csv row
   │
   ▼
[1] Deterministic PRE-GATES   (no LLM)  pregates.py
      SHA-256 · pHash dedup (re-used-image fraud) · blur/glare · EXIF/editor signature
      · transcript injection scan · evidence-requirement lookup
   │
   ▼
[2] VLM BRAIN — scoped stages (not one mega-prompt)   stages.py
      Stage 1  extract claim     transcript ONLY, treated as untrusted DATA   (gpt-4o-mini)
      Stage 2  inspect images    one call/image, "describe then judge"        (gpt-4o-mini)
      Stage 2b OCR text scan     transcribe in-image text → injection regex   (gpt-4o-mini)
      Stage 3  adjudicate        images + observations, NO history            (gpt-4.1)
   │
   ▼
[3] Deterministic FUSION   (no LLM)  fusion.py
      history → risk_flags · valid_image · wrong-object→contradicted
      · confidence-gated manual_review · cross-field invariants · safe defaults
   │
   ▼
output.csv row   (strict 14-column schema, schema.py)
```

**Design principle:** *the LLM observes and proposes; deterministic code decides the rule-governed fields.* Anything that can be a rule (history propagation, schema legality, `valid_image`, escalation) **is** a rule — so it's testable, explainable, and immune to prompt injection.

## Named patterns

- **Evidence-sufficiency gate** — qualitative minimum-evidence rule per (object, issue family), read from `evidence_requirements.csv` (no hardcoding).
- **Grounded adjudication** — quote the pixels before judging; abstain rather than guess.
- **Context-engineering per stage** — an explicit attention budget per call instead of one overloaded prompt.
- **Instruction/data separation** — the transcript *and* in-image text (via a dedicated OCR pass) are *data*; an embedded "approve this claim" is recorded (`text_instruction_present`) and routed to review, never obeyed.
- **Perceptual-hash dedup** — the same image content under a *different* claim is a fraud signal (`non_original_image`). Reuse is keyed on the file path and skips self-matches, so re-runs are idempotent.
- **Confidence-gated human escalation** — route the hardest cases to `manual_review_required`; never auto-decide them.
- **Structured outputs as contracts** — OpenAI strict `json_schema` + Pydantic invariants + a one-shot repair retry.
- **Response cache** — keyed on (model + prompt + image); re-runs and duplicate images are free.

## Evaluation — two angles

We evaluate on both axes that matter for an agent like this:
1. **Accuracy against the labeled golden set** (`sample_claims.csv`) — per-field accuracy, a `claim_status` confusion matrix, imbalance-aware κ / balanced accuracy, multi-label `risk_flags` F1, abstention (AURC) and calibration (ECE/Brier), plus a full error analysis. Two prompt strategies and a model-routing variant are compared (`evaluation/main.py`, `strategy_comparison.md`).
2. **LLM-as-judge** for the free-text justifications — a decomposed, bias-controlled rubric (`evaluation/judge.py`).

`main.py` runs the agent against **`claims.csv`** to produce `output.csv`; **`sample_claims.csv` is used only for testing/building** (it's the labeled set), so there's no train/eval leakage.

## Results (golden set, `sample_claims.csv`, zero-shot)

The production pipeline (**Strategy B**, gpt-4.1 adjudication + OCR injection scan) on the 20 labeled rows:

| field | accuracy |
|---|---|
| `claim_status` | **0.85** (Cohen's κ 0.66, balanced acc 0.80) |
| `issue_type` | 0.70 |
| `object_part` | 0.80 |
| `severity` | 0.40 |

`risk_flags` micro-F1 0.77 · AURC 0.126 · ECE 0.162. Full numbers + the **A-vs-B**, the **adjudication-model selection** (gpt-4o → gpt-4.1 → gpt-5 family), and the error analysis are in [`evaluation/strategy_comparison.md`](evaluation/strategy_comparison.md) and the auto-generated [`evaluation/evaluation_report.md`](evaluation/evaluation_report.md).

**Justification quality (LLM-as-judge).** Field accuracy can't see whether the *free-text explanation* is good. A decomposed-rubric judge (groundedness / faithfulness / verdict-consistency / relevance), reference-based and bias-controlled (it's told to ignore length, style, and stated confidence — the verbosity / style / self-confidence biases; single-answer scoring avoids pairwise position bias), scores the justifications **4.54/5** overall. The diagnostic that matters: faithfulness **4.82→1.67** and verdict-consistency **5.0→1.67** between correct and wrong verdicts — *when the model is wrong, its own explanation is unfaithful and self-inconsistent*, which is itself an escalation signal. See [`evaluation/evaluation_report_judge.md`](evaluation/evaluation_report_judge.md).

**Position:** B beats A on the decisions that matter (claim_status 0.85 vs 0.75, κ +0.15, issue_type +15pp, better calibration) at ~2× the calls — worth it for evidence review. The biggest error pile was `contradicted`→`supported` false-confirms (the model confirms on damage *presence* without checking the claim's part/magnitude).

**How we moved it — measured model selection, not prompt-nudging.** Two prompt/perception fixes were tried first and *both regressed* on gpt-4o (a part/magnitude *match-test* prompt and a mini→gpt-4o inspection upgrade each dropped claim_status to 0.75) — kept as documented negatives. The fixes that actually worked: **upgrading the adjudicator gpt-4o → gpt-4.1** (0.80 → **0.85**, κ 0.56 → 0.66, `contradicted` recall 1/5 → 2/5, *cheaper*), and a **dedicated OCR pass** that catches in-image (pixel-embedded) injections the transcript scan can't see (now flags `user_034`, `risk_flags` F1 0.76 → 0.77). gpt-5 / gpt-5-mini were also evaluated (via a reasoning-aware client) and did **not** beat gpt-4.1 on this perceptual task. The residual floor is label-boundary perception (2 cases). Full tables + reproduction in [`evaluation/strategy_comparison.md`](evaluation/strategy_comparison.md).

**Reproducibility fix shipped this pass:** the persistent perceptual-hash fraud store used to self-match on a second `output.csv` run (it keyed reuse on the bare filename), falsely flagging every image `non_original_image`. It now keys on the unique file path and skips self-matches — re-runs are idempotent, genuine cross-claim reuse still fires, and `output.csv` is corrected (4 legitimate `valid_image=false`, no phantom fraud flags). Guarded by regression tests.

## Severity rubric (visible-cue mapping)

`severity` is set from what is *visibly* damaged, anchored per object. This is the rubric (also encoded in the v2 adjudication prompt). It is our weakest field (golden acc 0.40 — a systematic over-rating one tier, reported honestly), so we lean on the abstention/escalation layer rather than treating raw severity as decisive.

| object | low | medium | high |
|---|---|---|---|
| **car** | minor surface scratch, small shallow dent | clear dent, deep/multiple scratches, cracked lamp or trim | panel deformation, shattered glass, structural/frame damage, airbag deployed |
| **laptop** | cosmetic scuff, single mark | cracked corner/casing dent, dead-pixel cluster, bent port | shattered/spider-cracked screen, snapped hinge, liquid-damage spread |
| **package** | surface scuff, small dent in box | crushed corner, partial tear/seam split | fully crushed, torn open, contents exposed/spilled |

## Setup & run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # or put it in a gitignored .env at the repo root

# Produce output.csv for the test set:
python code/main.py              # reads dataset/claims.csv -> writes output.csv (repo root)

# Evaluate on the labeled golden set:
python code/evaluation/main.py --strategy b      # production 3-stage
python code/evaluation/main.py --strategy a      # single-prompt baseline

# Justification quality (LLM-as-judge over the golden justifications):
python code/evaluation/judge.py                  # writes evaluation_report_judge.md

# Re-runs are free (response cache). Add --offline to forbid any API call (cache-only).
python -m pytest code/tests/                      # 54 offline tests
```

## Configuration

All knobs live in [`config.yaml`](config.yaml) — provider, per-role model ids (extraction/inspection/ocr `gpt-4o-mini`, adjudication `gpt-4.1`, judge `gpt-4o`), pricing (verified June 2026), image max-dim/quality, temperature, repair retries, structured-output mode, `adjudication_prompt_version`, `ocr_injection_scan`, and `reasoning_max_tokens`. So every experiment — model, prompt version, OCR on/off — is a one-line config flip, not a code edit. The `VLMClient` is **reasoning-model-aware** (gpt-5 / o-series get no `temperature` and a larger completion budget) and provider-agnostic by design (OpenAI is wired and run; Anthropic/Gemini are documented extension points). **Secrets are read from environment variables only**, never hardcoded.
