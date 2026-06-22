# LLM-as-judge — justification quality (golden set)

Decomposed rubric, reference-based single-answer scoring (no pairwise position bias). The judge is instructed to **ignore length, style, and stated confidence** — the named verbosity / style / self-confidence biases — and score substance only. Judge model: see `config.yaml` (`model_judge`). Pipeline predictions are served from cache; only the judge calls bill.

## Mean rubric scores (1-5)

| dimension | all | when verdict correct | when verdict wrong |
|---|---|---|---|
| groundedness | 4.55 | 4.88 | 2.67 |
| faithfulness | 4.35 | 4.82 | 1.67 |
| verdict_consistency | 4.5 | 5.0 | 1.67 |
| claim_relevance | 4.75 | 4.94 | 3.67 |

**Overall mean (all dimensions): 4.54/5** over 20 justifications (17 on correct verdicts, 3 on wrong).

Reading it: a high *groundedness/faithfulness* even where the verdict is wrong means the justifications are honest about the evidence (they explain what was seen) — the failures are perception/label-boundary calls, not fabricated reasoning. A drop in *verdict_consistency* on wrong rows would instead flag motivated/after-the-fact reasoning.

## Per-claim

- **user_001** (✓ verdict): g5 f5 c5 r5 — The candidate accurately references visible evidence of a dent on the rear bumper, directly supporting the claim.
- **user_002** (✓ verdict): g4 f3 c5 r5 — The candidate correctly identifies a scratch on the front bumper but mentions 'additional damage' not supported by the reference.
- **user_004** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies visible evidence of the crack in img_1, supporting the claim.
- **user_007** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies visible damage to the side mirror, directly supporting the claim.
- **user_005** (✓ verdict): g5 f5 c5 r4 — The candidate correctly identifies the location of the damage and supports the verdict with specific image references, but it could more directly address the customer's claim of bumper damage.
- **user_006** (✓ verdict): g4 f5 c5 r5 — The candidate correctly identifies the lack of clear evidence for the headlight issue, but could specify the image ID more precisely.
- **user_003** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies and describes the visible dent in img_2, directly supporting the claim about the car door.
- **user_008** (✗ verdict): g2 f1 c1 r3 — The candidate incorrectly describes the image as showing scratches, contradicting the expert's observation of severe front-end damage.
- **user_009** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies and describes the visible crack on the laptop screen, directly addressing the customer's claim.
- **user_010** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies visible hinge damage in img_1, directly supporting the claim.
- **user_011** (✓ verdict): g5 f4 c5 r5 — The candidate is well-grounded in visible evidence but slightly overstates the presence of pooled water.
- **user_012** (✓ verdict): g5 f5 c5 r5 — The candidate accurately references visible evidence of the dent and aligns with the claim about the laptop's corner.
- **user_018** (✓ verdict): g5 f5 c5 r5 — The candidate accurately describes visible evidence of screen damage, supporting the verdict and addressing the claim.
- **user_020** (✗ verdict): g3 f2 c2 r4 — The candidate references a scratch but does not convincingly link it to the claimed damage severity.
- **user_015** (✓ verdict): g5 f5 c5 r5 — The candidate accurately describes the visible evidence of the crushed corner, directly supporting the claim.
- **user_030** (✓ verdict): g5 f5 c5 r5 — The candidate accurately references visible evidence in img_1 that supports the claim of torn packaging.
- **user_031** (✓ verdict): g5 f5 c5 r5 — The candidate accurately references visible evidence of water stains on the package, supporting the claim.
- **user_032** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies the lack of visible evidence to verify the missing product claim.
- **user_033** (✓ verdict): g5 f5 c5 r5 — The candidate correctly identifies that the image does not show the claimed shipping box, aligning with the verdict.
- **user_034** (✗ verdict): g3 f2 c2 r4 — The candidate references specific images and damage but contradicts the expert's assessment of the evidence.

## Cost

- Judge calls billed: **20** (98 pipeline calls served from cache).
- Tokens: 10,138 in / 973 out · est **$0.0351**.
