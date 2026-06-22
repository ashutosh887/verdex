# Evaluation & Operational Report — Strategy A

Golden set: `sample_claims.csv` (20 rows). Prompts are zero-shot (no few-shot exemplars), so there is no train/eval leakage to leave out.

## 1. Field accuracy

| field | accuracy | notes |
|---|---|---|
| claim_status | 0.75 | kappa=0.5122, balanced_acc=0.6487 (3-class imbalanced — trust over raw acc) |
| issue_type | 0.55 |  |
| object_part | 0.9 |  |
| severity | 0.35 |  |

### claim_status confusion matrix

| gold \ pred | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 11 | 1 | 1 |
| contradicted | 1 | 3 | 1 |
| not_enough_information | 1 | 0 | 1 |

## 2. risk_flags (multi-label)

micro-F1 **0.702** · macro-F1 **0.524** · Hamming loss **0.071** · exact-match **0.550**

## 3. Abstention & calibration

- **AURC** (area under risk–coverage) **0.0615** — lower is better; does it abstain on the cases it would otherwise get wrong?
- **ECE** (10-bin) **0.229** · **Brier** **0.2237**.

## 4. Error analysis — every disagreement

17 rows disagree on at least one field — read them, don't just count:

- **user_001** (conf 1.0): status gold=`supported`/pred=`supported`, issue `dent`/`dent`, part `rear_bumper`/`rear_bumper`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_002** (conf 0.97): status gold=`supported`/pred=`contradicted`, issue `scratch`/`scratch`, part `front_bumper`/`front_bumper`, sev `low`/`high`; flags gold=`none` pred=`claim_mismatch`.
- **user_004** (conf 1.0): status gold=`supported`/pred=`supported`, issue `crack`/`crack`, part `windshield`/`windshield`, sev `medium`/`high`; flags gold=`none` pred=`blurry_image`.
- **user_007** (conf 1.0): status gold=`supported`/pred=`supported`, issue `broken_part`/`crack`, part `side_mirror`/`side_mirror`, sev `medium`/`medium`; flags gold=`none` pred=`none`.
- **user_005** (conf 0.98): status gold=`contradicted`/pred=`contradicted`, issue `scratch`/`dent`, part `rear_bumper`/`rear_bumper`, sev `low`/`none`; flags gold=`claim_mismatch;user_history_risk;manual_review_required` pred=`cropped_or_obstructed;claim_mismatch;user_history_risk;manual_review_required`.
- **user_008** (conf 0.85): status gold=`contradicted`/pred=`not_enough_information`, issue `broken_part`/`unknown`, part `front_bumper`/`hood`, sev `high`/`unknown`; flags gold=`claim_mismatch;non_original_image;user_history_risk;manual_review_required` pred=`low_light_or_glare;damage_not_visible;user_history_risk;manual_review_required`.
- **user_009** (conf 1.0): status gold=`supported`/pred=`supported`, issue `crack`/`crack`, part `screen`/`screen`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_010** (conf 1.0): status gold=`supported`/pred=`supported`, issue `broken_part`/`broken_part`, part `hinge`/`hinge`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_011** (conf 0.95): status gold=`supported`/pred=`supported`, issue `stain`/`water_damage`, part `keyboard`/`keyboard`, sev `medium`/`medium`; flags gold=`none` pred=`none`.
- **user_012** (conf 0.95): status gold=`supported`/pred=`not_enough_information`, issue `dent`/`unknown`, part `corner`/`corner`, sev `low`/`unknown`; flags gold=`none` pred=`low_light_or_glare;damage_not_visible;possible_manipulation;manual_review_required`.
- **user_018** (conf 1.0): status gold=`supported`/pred=`supported`, issue `crack`/`glass_shatter`, part `screen`/`screen`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_020** (conf 0.95): status gold=`contradicted`/pred=`contradicted`, issue `none`/`scratch`, part `trackpad`/`trackpad`, sev `none`/`low`; flags gold=`damage_not_visible;user_history_risk;manual_review_required` pred=`claim_mismatch;user_history_risk;manual_review_required`.
- **user_030** (conf 1.0): status gold=`supported`/pred=`supported`, issue `torn_packaging`/`torn_packaging`, part `seal`/`seal`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_031** (conf 1.0): status gold=`supported`/pred=`supported`, issue `water_damage`/`water_damage`, part `package_side`/`box`, sev `medium`/`medium`; flags gold=`user_history_risk;manual_review_required` pred=`blurry_image;user_history_risk;manual_review_required`.
- **user_032** (conf 0.95): status gold=`not_enough_information`/pred=`supported`, issue `unknown`/`missing_part`, part `contents`/`contents`, sev `unknown`/`unknown`; flags gold=`cropped_or_obstructed;damage_not_visible;manual_review_required` pred=`manual_review_required`.
- **user_033** (conf 1.0): status gold=`contradicted`/pred=`contradicted`, issue `unknown`/`unknown`, part `unknown`/`unknown`, sev `low`/`medium`; flags gold=`wrong_object;claim_mismatch;user_history_risk;manual_review_required` pred=`wrong_object;claim_mismatch;user_history_risk;manual_review_required`.
- **user_034** (conf 1.0): status gold=`contradicted`/pred=`supported`, issue `none`/`torn_packaging`, part `seal`/`seal`, sev `none`/`high`; flags gold=`damage_not_visible;text_instruction_present;user_history_risk;manual_review_required` pred=`text_instruction_present;user_history_risk;manual_review_required`.

## 5. Operational analysis

- Model calls (sample run): **49** (20 billed, 29 cached).
- Tokens: **38,825** in / **6,211** out.
- Images processed: **29**.
- Sample-set est. cost: **$0.1273** (≈ $0.0064/claim).
- Extrapolated full test set (44 rows): **≈ $0.28** (pricing assumptions in `config.yaml`; verify vs current OpenAI pricing).
- Wall-clock: **94.6s** for 20 claims (serial).

**TPM/RPM & efficiency levers:** temperature 0 for determinism; per-image downscale to 1024px; **response cache** keyed on (model+prompt+image) so re-runs and duplicate images are free; cheap model for extraction/inspection, flagship only for adjudication; one repair retry on schema-invalid output. Scale-up: token-bucket rate limiting + exponential backoff with jitter honoring `retry-after`; 429 (our quota) → slow down, 529 (provider overload) → back off / optional fallback; batch the offline eval.
