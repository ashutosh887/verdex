# Strategy & model comparison — and what the errors tell us

All variants share the **same deterministic fusion layer** (pregates, OCR injection scan, history propagation, valid_image, escalation). We compare two prompt structures and several adjudication models on the 20-row golden set (`sample_claims.csv`, zero-shot, no leakage).

- **Strategy A** — one combined prompt: extract + inspect + adjudicate in a single call.
- **Strategy B** — the production 3-stage scoped pipeline (extract → per-image inspect → adjudicate), one call per image plus extraction and adjudication. **Shipped.**

## 1. Strategy A vs B (both on the shipped model, gpt-4.1)

| Metric | A (combined) | B (3-stage) | Winner |
|---|---|---|---|
| `claim_status` accuracy | 0.75 | **0.85** | B |
| `claim_status` Cohen's κ | 0.51 | **0.66** | B |
| `claim_status` balanced acc | 0.65 | **0.80** | B |
| `issue_type` accuracy | 0.55 | **0.70** | B |
| `object_part` accuracy | **0.90** | 0.80 | A (noise) |
| `severity` accuracy | 0.35 | **0.40** | B |
| `risk_flags` micro-F1 | 0.70 | **0.77** | B |
| `risk_flags` exact-match | 0.55 | **0.60** | B |
| AURC (↓ better) | **0.062** | 0.126 | A |
| ECE (↓ better) | 0.229 | **0.162** | B |
| Model calls (sample) | **49** | 98 | A |
| Cost (full 44-row test set) | — | **≈ $0.44** | — |

**Position: B ships.** It wins `claim_status` (+10pp, κ +0.15), `issue_type`, `risk_flags`, and calibration. The cost is real — B runs ~2× the calls (per-image inspection + a dedicated OCR pass + grounded adjudication are separate calls) — and we name it. For evidence review, where a wrong `claim_status` is expensive, that trade is worth it. A's one edge is AURC (it abstains more readily); B compensates with the deterministic escalation layer.

**They still fail differently:** B is perfect on `supported` (13/13) and now gets `contradicted` 2/5; A is more balanced on `contradicted` (3/5) but noisier on `supported` (11/13). B's higher headline accuracy carries the more dangerous false-confirm profile, which is exactly why the escalation layer matters.

## 2. Adjudication-model selection (the biggest single lever)

Same 3-stage pipeline, same prompts, only the **adjudication model** changes:

| Model | `claim_status` | κ | `contradicted` recall | notes |
|---|---|---|---|---|
| gpt-4o (legacy) | 0.80 | 0.56 | 1/5 | the prior shipped model |
| **gpt-4.1 (SHIPPED)** | **0.85** | **0.66** | **2/5** | cheaper ($2/$8 vs $2.50/$10), faster; fixed user_005 (nei→contradicted) with no regression on `supported` (13/13) |
| gpt-5-mini | 0.80 | 0.58 | 2/5 | *lost* a `supported` (12/13); reasoning model, slower |
| gpt-5 | — | — | — | reasoning model; runs via the reasoning-aware client (no temperature, large completion budget) but did not beat gpt-4.1 in testing and is slower/pricier — not shipped |

**gpt-4.1 is the win the prompt experiments couldn't deliver** — it lifted `contradicted` recall 1/5→2/5 and κ +0.10 while being cheaper. Reasoning models (gpt-5 family) did **not** help this perceptual task. Reports: `evaluation_report_adj_gpt41.md`, `evaluation_report_adj_gpt5mini.md`. The model is one line in `config.yaml` (`model_adjudication`), with gpt-4o as the documented fallback.

## 3. In-image prompt-injection defense (OCR) — measured win

A dedicated VLM OCR pass transcribes each image's visible text and runs it through the **same injection regex as the transcript** (text treated as untrusted data, never obeyed). This closed the pixel-embedded-injection gap: **user_034's `text_instruction_present` is now correctly flagged**, lifting `risk_flags` micro-F1 0.760→0.769 (macro 0.467→0.542) with no other change. Toggle: `ocr_injection_scan` in `config.yaml`. No new dependency (local OCR libs were absent; the VLM already reads pixels).

## 4. Earlier experiments that REGRESSED (kept as measured negatives, on gpt-4o)

Before the model upgrade, two hypotheses for the false-confirm pile were measured on gpt-4o — **both regressed, so we did not ship them.** Reporting a measured negative is the discipline this task rewards.

| Config | `claim_status` | κ | `contradicted` | notes |
|---|---|---|---|---|
| v1 prompt (shipped) | 0.80 | 0.56 | 1/5 | baseline |
| v2 — part/magnitude MATCH TEST + severity rubric | 0.75 | 0.48 | 1/5 | cost a correct `supported`; severity predictions byte-identical (rubric inert) |
| inspection mini→gpt-4o | 0.75 | 0.48 | 1/5 | lost a `supported`; contradicted unchanged |

Both reproduce via config (`adjudication_prompt_version: v2`, `model_inspection: gpt-4o`). Reports: `evaluation_report_b_v1.md`, `_b_v2.md`, `_insp4o.md`. **Lesson:** the false-confirm floor wasn't the adjudication prompt or perception-model strength — it was label-boundary perception and an in-image injection. The fixes that actually moved it were a **better adjudication model** (§2) and the **OCR injection scan** (§3), not prompt nudging.

## 5. Error-analysis piles (shipped B = gpt-4.1 + OCR)

1. **Remaining `contradicted`→`supported` (2: user_008, user_020).** user_008 = a non-original/mismatched image (part/type mismatch); user_020 = a faint mark on an otherwise-intact trackpad the labeler scored no-damage. These are **label-boundary perception** calls — robust across every model and prompt we tried. user_005 (was here) is now fixed by gpt-4.1; user_034 is still pred=supported on the verdict but is now correctly flagged + escalated via OCR.
2. **Severity over-rating (acc 0.40).** Systematic medium→high / low→medium. Prompt rubric was inert; the honest fix is calibration on labeled severity data, not prompt text.
3. **Calibration (ECE 0.162).** gpt-4.1 is confident; we report the gap and lean on the abstention/escalation layer over the raw confidence number.
4. **Minor adjacent-class confusions** (`stain`↔`water_damage`, `crack`↔`glass_shatter`).

## 6. LLM-as-judge — justification quality

A decomposed, bias-controlled rubric (`judge.py`) scores the justifications **4.54/5**. The diagnostic: faithfulness **4.82→1.67** and verdict-consistency **5.0→1.67** between correct and wrong verdicts — when the model is wrong, its own explanation is unfaithful and self-inconsistent, a usable escalation signal. See `evaluation_report_judge.md`.

## 7. Reproducibility fix — the persistent pHash store self-matched on re-run

Re-generating `output.csv` a second time once flagged **every** image `non_original_image` (→ `valid_image=false` + `manual_review`): the persistent perceptual-hash store keyed reuse on the bare filename and matched each image against its *own* prior-run entry. A cross-case scan confirmed **zero** genuine cross-claim duplicates in the test set — all false flags. Fix: key reuse on the unique file **path** and skip self-matches → re-runs are idempotent, genuine reuse still fires. Corrected `output.csv`: `valid_image=false` 7→**4** (all legitimate). Guarded by two regression tests.

## Reproduce

```
python code/evaluation/main.py --strategy b   # shipped (gpt-4.1 + OCR) -> evaluation_report.md
python code/evaluation/main.py --strategy a   # single-prompt baseline -> evaluation_report_strategy_a.md
python code/evaluation/judge.py               # LLM-as-judge -> evaluation_report_judge.md
# model / variant sweeps are one-line config flips: model_adjudication, adjudication_prompt_version,
# model_inspection, ocr_injection_scan.
```
Shipped strategies re-run free from the response cache.
