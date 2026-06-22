# Evaluation & Operational Report — Strategy B

Golden set: `sample_claims.csv` (20 rows). Prompts are zero-shot (no few-shot exemplars), so there is no train/eval leakage to leave out.

## 1. Field accuracy

| field | accuracy | notes |
|---|---|---|
| claim_status | 0.8 | kappa=0.5767, balanced_acc=0.7744 (3-class imbalanced — trust over raw acc) |
| issue_type | 0.65 |  |
| object_part | 0.8 |  |
| severity | 0.4 |  |

### claim_status confusion matrix

| gold \ pred | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 0 | 1 |
| contradicted | 3 | 2 | 0 |
| not_enough_information | 0 | 0 | 2 |

## 2. risk_flags (multi-label)

micro-F1 **0.755** · macro-F1 **0.536** · Hamming loss **0.054** · exact-match **0.600**

## 3. Abstention & calibration

- **AURC** (area under risk–coverage) **0.1335** — lower is better; does it abstain on the cases it would otherwise get wrong?
- **ECE** (10-bin) **0.108** · **Brier** **0.157**.

## 4. Error analysis — every disagreement

15 rows disagree on at least one field — read them, don't just count:

- **user_001** (conf 0.92): status gold=`supported`/pred=`supported`, issue `dent`/`dent`, part `rear_bumper`/`rear_bumper`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_002** (conf 0.88): status gold=`supported`/pred=`supported`, issue `scratch`/`scratch`, part `front_bumper`/`front_bumper`, sev `low`/`medium`; flags gold=`none` pred=`none`.
- **user_004** (conf 0.93): status gold=`supported`/pred=`supported`, issue `crack`/`crack`, part `windshield`/`windshield`, sev `medium`/`high`; flags gold=`none` pred=`blurry_image`.
- **user_007** (conf 0.9): status gold=`supported`/pred=`supported`, issue `broken_part`/`broken_part`, part `side_mirror`/`side_mirror`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_005** (conf 0.86): status gold=`contradicted`/pred=`contradicted`, issue `scratch`/`none`, part `rear_bumper`/`rear_bumper`, sev `low`/`none`; flags gold=`claim_mismatch;user_history_risk;manual_review_required` pred=`damage_not_visible;user_history_risk;manual_review_required`.
- **user_008** (conf 0.9): status gold=`contradicted`/pred=`supported`, issue `broken_part`/`scratch`, part `front_bumper`/`hood`, sev `high`/`medium`; flags gold=`claim_mismatch;non_original_image;user_history_risk;manual_review_required` pred=`low_light_or_glare;user_history_risk;manual_review_required`.
- **user_009** (conf 0.9): status gold=`supported`/pred=`supported`, issue `crack`/`crack`, part `screen`/`screen`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_011** (conf 0.9): status gold=`supported`/pred=`supported`, issue `stain`/`water_damage`, part `keyboard`/`keyboard`, sev `medium`/`medium`; flags gold=`none` pred=`none`.
- **user_012** (conf 0.62): status gold=`supported`/pred=`not_enough_information`, issue `dent`/`unknown`, part `corner`/`corner`, sev `low`/`unknown`; flags gold=`none` pred=`damage_not_visible;possible_manipulation;manual_review_required`.
- **user_018** (conf 0.92): status gold=`supported`/pred=`supported`, issue `crack`/`glass_shatter`, part `screen`/`screen`, sev `medium`/`high`; flags gold=`none` pred=`none`.
- **user_020** (conf 0.89): status gold=`contradicted`/pred=`supported`, issue `none`/`scratch`, part `trackpad`/`trackpad`, sev `none`/`low`; flags gold=`damage_not_visible;user_history_risk;manual_review_required` pred=`user_history_risk;manual_review_required`.
- **user_030** (conf 0.9): status gold=`supported`/pred=`supported`, issue `torn_packaging`/`torn_packaging`, part `seal`/`package_side`, sev `medium`/`medium`; flags gold=`none` pred=`none`.
- **user_032** (conf 0.78): status gold=`not_enough_information`/pred=`not_enough_information`, issue `unknown`/`unknown`, part `contents`/`package_side`, sev `unknown`/`unknown`; flags gold=`cropped_or_obstructed;damage_not_visible;manual_review_required` pred=`damage_not_visible;manual_review_required`.
- **user_033** (conf 0.63): status gold=`contradicted`/pred=`contradicted`, issue `unknown`/`unknown`, part `unknown`/`unknown`, sev `low`/`medium`; flags gold=`wrong_object;claim_mismatch;user_history_risk;manual_review_required` pred=`wrong_object;claim_mismatch;user_history_risk;manual_review_required`.
- **user_034** (conf 0.9): status gold=`contradicted`/pred=`supported`, issue `none`/`torn_packaging`, part `seal`/`package_side`, sev `none`/`medium`; flags gold=`damage_not_visible;text_instruction_present;user_history_risk;manual_review_required` pred=`text_instruction_present;user_history_risk;manual_review_required`.

## 5. Operational analysis

- Model calls (sample run): **100** (22 billed, 78 cached).
- Tokens: **38,111** in / **11,710** out.
- Images processed: **29**.
- Sample-set est. cost: **$0.0329** (≈ $0.0016/claim).
- Extrapolated full test set (44 rows): **≈ $0.07** (pricing assumptions in `config.yaml`; verify vs current OpenAI pricing).
- Wall-clock: **171.7s** for 20 claims (serial).

**TPM/RPM & efficiency levers:** temperature 0 for determinism; per-image downscale to 1024px; **response cache** keyed on (model+prompt+image) so re-runs and duplicate images are free; cheap model for extraction/inspection, flagship only for adjudication; one repair retry on schema-invalid output. Scale-up: token-bucket rate limiting + exponential backoff with jitter honoring `retry-after`; 429 (our quota) → slow down, 529 (provider overload) → back off / optional fallback; batch the offline eval.
