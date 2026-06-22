# Verdex

**Image-grounded adjudication for damage claims.** Verdex looks at the photos a user submits for a damage claim — on a **car**, **laptop**, or **package** — reads the chat they sent with it, weighs their claim history, and decides whether the images **support**, **contradict**, or are **insufficient** to judge the claim. Every verdict quotes what is actually visible in the pixels.

It is built around one hard rule: **images are the only source of truth.** History and metadata can raise a flag or route a case to a human — they can *never* flip a clear visual verdict. That separation isn't a request in a prompt; it lives in deterministic code that the model cannot talk its way around.

---

## Why this is interesting

Most "LLM looks at an image and decides" systems fail in three predictable ways: they hallucinate confident verdicts on bad evidence, they obey instructions hidden inside the input ("approve this claim"), and they can't tell you *why* they were wrong when they were. Verdex is engineered against exactly those failure modes.

- **Calibrated abstention.** It returns `not_enough_information` only when the evidence is *genuinely* insufficient — measured with a risk–coverage curve (AURC), not vibes.
- **Prompt-injection resistant by construction.** The transcript *and* text embedded inside images (via a dedicated OCR pass) are treated as untrusted **data**. An embedded "APPROVE THIS CLAIM" is recorded and routed to review — never obeyed.
- **Fraud signals as first-class output.** Perceptual-hash dedup catches the same photo reused under a different claim; EXIF/editor signatures flag possible manipulation.
- **The LLM proposes, deterministic code decides.** Anything that can be a rule — history propagation, schema legality, escalation, `valid_image` — *is* a rule, so it's testable, explainable, and immune to prompt injection.
- **Honest evaluation.** Two axes: per-field accuracy against a labeled golden set (with κ, balanced accuracy, multi-label F1, AURC, ECE/Brier) *and* an LLM-as-judge rubric for the free-text justifications. Negative results are kept as documented negatives, not hidden.

---

## Architecture

```
claims.csv row
   │
   ▼
[1] Deterministic PRE-GATES   (no LLM)  pregates.py
      SHA-256 · pHash dedup (reused-image fraud) · blur/glare · EXIF/editor signature
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

**Why staged, not one prompt:** each call gets an explicit *attention budget*. Claim extraction never sees the images; image inspection never sees the user's claim history; adjudication sees the images and the structured observations but **not** the history. History is fused in afterward, deterministically. That's what keeps history from contaminating a visual verdict.

---

## How it works

**1. Deterministic pre-gates (`pipeline/pregates.py`)** — before any model call: SHA-256 + perceptual-hash dedup (a reused image under a new claim is a `non_original_image` fraud signal, path-keyed so re-runs are idempotent), blur/glare/quality checks, EXIF and editor-signature inspection for `possible_manipulation`, a transcript prompt-injection scan, and the minimum-evidence lookup from `evidence_requirements.csv`.

**2. The VLM brain (`pipeline/stages.py`, `pipeline/vlm.py`)** — scoped stages instead of one overloaded prompt. The model *describes before it judges*, quoting visible cues. A dedicated OCR pass transcribes any text rendered inside the image and runs it through the same injection regex as the transcript, closing the pixel-embedded-injection gap.

**3. Deterministic fusion (`pipeline/fusion.py`)** — the rule layer that owns every rule-governed field: history → `risk_flags`, `valid_image`, wrong-object → `contradicted`, confidence-gated `manual_review_required`, cross-field invariants, and safe defaults. Output is validated against a strict 14-column schema (`pipeline/schema.py`) with OpenAI strict `json_schema` + Pydantic invariants + a one-shot repair retry.

---

## Results

Production pipeline (**Strategy B** — gpt-4.1 adjudication + OCR injection scan) on the 20-row labeled golden set, zero-shot:

| field | accuracy |
|---|---|
| `claim_status` | **0.85** (Cohen's κ 0.66, balanced acc 0.80) |
| `issue_type` | 0.70 |
| `object_part` | 0.80 |
| `severity` | 0.40 |

`risk_flags` micro-F1 **0.77** · AURC **0.126** · ECE **0.162**.

**Justification quality (LLM-as-judge): 4.54/5.** A decomposed, bias-controlled rubric (groundedness / faithfulness / verdict-consistency / relevance), told to ignore length, style, and stated confidence. The most useful diagnostic: faithfulness drops **4.82 → 1.67** and verdict-consistency **5.0 → 1.67** between correct and wrong verdicts — *when the model is wrong, its own explanation becomes unfaithful and self-inconsistent*, which is itself an escalation signal.

**What actually moved the needle was model selection, not prompt-nudging.** Two prompt/perception "fixes" were tried first and *both regressed* (kept as documented negatives). The wins: upgrading the adjudicator **gpt-4o → gpt-4.1** (0.80 → 0.85, κ 0.56 → 0.66, and *cheaper*), and the dedicated OCR pass. The gpt-5 family was evaluated via a reasoning-aware client and did **not** beat gpt-4.1 on this perceptual task.

Full tables, the A-vs-B comparison, model-selection trail, and error analysis live in
[`code/evaluation/strategy_comparison.md`](code/evaluation/strategy_comparison.md) and the auto-generated
[`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md).
The deep technical writeup is [`code/README.md`](code/README.md).

---

## Quickstart

```bash
pip install -r code/requirements.txt
export OPENAI_API_KEY=...        # or put it in a gitignored .env at the repo root

# Produce output.csv for the test set:
python code/main.py              # reads dataset/claims.csv -> writes output.csv (repo root)

# Evaluate on the labeled golden set:
python code/evaluation/main.py --strategy b      # production 3-stage
python code/evaluation/main.py --strategy a      # single-prompt baseline

# Justification quality (LLM-as-judge over the golden justifications):
python code/evaluation/judge.py                  # writes evaluation_report_judge.md

# Re-runs are free (response cache). Add --offline to forbid any API call (cache-only).
python -m pytest code/tests/                     # offline test suite
```

---

## Configuration

Every knob lives in [`code/config.yaml`](code/config.yaml): provider, per-role model ids, pricing (verified June 2026), image max-dim/quality, temperature, repair retries, structured-output mode, `adjudication_prompt_version`, `ocr_injection_scan`, and `reasoning_max_tokens`. Every experiment — model, prompt version, OCR on/off — is a one-line config flip, not a code edit.

| role | default model |
|---|---|
| extraction / inspection / OCR | `gpt-4o-mini` |
| adjudication | `gpt-4.1` |
| LLM-as-judge (offline only) | `gpt-4o` |

The `VLMClient` is **reasoning-model-aware** (gpt-5 / o-series get no `temperature` and a larger completion budget) and **provider-agnostic** by design — OpenAI is wired and run; Anthropic and Gemini are documented extension points in the config. **Secrets are read from environment variables only**, never hardcoded.

---

## Output schema

One `output.csv` row per input claim, 14 columns in fixed order:

| column | meaning |
|---|---|
| `evidence_standard_met` | `true` if the image set is sufficient to evaluate the claim |
| `evidence_standard_met_reason` | short reason for the evidence decision |
| `risk_flags` | semicolon-separated risk flags, or `none` |
| `issue_type` | visible issue type |
| `object_part` | relevant object part |
| `claim_status` | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | concise, image-grounded explanation |
| `supporting_image_ids` | image IDs supporting the decision, or `none` |
| `valid_image` | `true` if the image set is usable for automated review |
| `severity` | `none`, `low`, `medium`, `high`, or `unknown` |

<details>
<summary><b>Allowed values</b></summary>

- **`claim_status`**: `supported`, `contradicted`, `not_enough_information`
- **`issue_type`**: `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown`
- **`risk_flags`**: `none`, `blurry_image`, `cropped_or_obstructed`, `low_light_or_glare`, `wrong_angle`, `wrong_object`, `wrong_object_part`, `damage_not_visible`, `claim_mismatch`, `possible_manipulation`, `non_original_image`, `text_instruction_present`, `user_history_risk`, `manual_review_required`
- **Car `object_part`**: `front_bumper`, `rear_bumper`, `door`, `hood`, `windshield`, `side_mirror`, `headlight`, `taillight`, `fender`, `quarter_panel`, `body`, `unknown`
- **Laptop `object_part`**: `screen`, `keyboard`, `trackpad`, `hinge`, `lid`, `corner`, `port`, `base`, `body`, `unknown`
- **Package `object_part`**: `box`, `package_corner`, `package_side`, `seal`, `label`, `contents`, `item`, `unknown`
- **`severity`**: `none`, `low`, `medium`, `high`, `unknown`

</details>

---

## Project structure

```text
.
├── README.md                         # You are here
├── code/
│   ├── README.md                     # Deep technical writeup
│   ├── config.yaml                   # All tunable knobs
│   ├── requirements.txt
│   ├── main.py                       # Entry point: claims.csv -> output.csv
│   ├── pipeline/                     # The system
│   │   ├── pregates.py               # [1] deterministic pre-gates
│   │   ├── stages.py                 # [2] scoped VLM stages
│   │   ├── vlm.py                    # provider-agnostic, reasoning-aware client
│   │   ├── fusion.py                 # [3] deterministic fusion / rule layer
│   │   ├── schema.py                 # strict 14-column output contract
│   │   ├── validate.py · config.py · run.py
│   ├── evaluation/                   # Offline eval harness
│   │   ├── main.py                   # accuracy + strategy comparison
│   │   ├── judge.py                  # LLM-as-judge for justifications
│   │   ├── metrics.py · strategies.py
│   │   └── *.md                      # generated evaluation reports
│   └── tests/                        # offline test suite (no API calls)
└── dataset/
    ├── sample_claims.csv             # labeled golden set (dev/eval only)
    ├── claims.csv                    # test inputs -> output.csv
    ├── user_history.csv              # historical claim context
    ├── evidence_requirements.csv     # minimum image evidence rules
    └── images/{sample,test}/         # claim images
```

---

## Design principles

- **Evidence-sufficiency gate** — a qualitative minimum-evidence rule per (object, issue family), read from data, not hardcoded.
- **Grounded adjudication** — quote the pixels before judging; abstain rather than guess.
- **Context engineering per stage** — an explicit attention budget per call instead of one overloaded prompt.
- **Instruction/data separation** — transcript and in-image text are *data*; embedded instructions are flagged, never obeyed.
- **Confidence-gated human escalation** — the hardest cases are routed to `manual_review_required`, never auto-decided.
- **Structured outputs as contracts** — strict `json_schema` + Pydantic invariants + one-shot repair retry.
- **Free re-runs** — a response cache keyed on (model + prompt + image) makes iteration and duplicate images cost nothing.

---

## Tech stack

Python · OpenAI vision models (`gpt-4o-mini`, `gpt-4.1`) · Pydantic for the output contract · perceptual hashing for fraud detection · pytest for the offline suite. Provider-agnostic client with Anthropic and Gemini as documented extension points.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute.
