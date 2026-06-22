from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL = os.path.dirname(os.path.abspath(__file__))
for _p in (_CODE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics as M  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

from pipeline.fusion import load_user_history, parse_history_flags  # noqa: E402
from pipeline.run import load_requirements_for, process_claim  # noqa: E402
from pipeline.schema import INPUT_COLUMNS, InputRow  # noqa: E402
from pipeline.vlm import make_client  # noqa: E402

_REPO = os.path.dirname(_CODE)
_FIELDS = ["claim_status", "issue_type", "object_part", "severity"]
_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


def _flags(raw: str):
    return [p.strip() for p in (raw or "").split(";") if p.strip() and p.strip() != "none"]


def _read_golden(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(strategy: str, dataset: str, limit, offline: bool):
    golden = _read_golden(os.path.join(dataset, "sample_claims.csv"))
    if limit:
        golden = golden[:limit]
    requirements = load_requirements_for(dataset)
    history = load_user_history(os.path.join(dataset, "user_history.csv"))
    bundle_fn = STRATEGIES[strategy]

    client = make_client()
    client.offline = offline

    preds = {f: [] for f in _FIELDS}
    golds = {f: [] for f in _FIELDS}
    pred_flag_sets, gold_flag_sets = [], []
    confidences, status_correct = [], []
    failures = []

    t0 = time.time()
    n_images = 0
    for g in golden:
        row = InputRow(**{c: g[c] for c in INPUT_COLUMNS})
        hflags = parse_history_flags(history.get(row.user_id))
        pc = process_claim(client, row, requirements, dataset, hflags, bundle_fn=bundle_fn)
        out = pc.output.to_csv_dict()
        n_images += sum(1 for s in pc.pregates.image_signals if s.exists)

        for fld in _FIELDS:
            preds[fld].append(out[fld])
            golds[fld].append(g[fld])
        pred_flag_sets.append(_flags(out["risk_flags"]))
        gold_flag_sets.append(_flags(g["risk_flags"]))
        conf = pc.audit.get("adjudication_confidence", 0.0)
        ok = out["claim_status"] == g["claim_status"]
        confidences.append(conf)
        status_correct.append(ok)
        if not ok or any(out[f] != g[f] for f in _FIELDS):
            failures.append({
                "user_id": row.user_id,
                "gold": {f: g[f] for f in _FIELDS},
                "pred": {f: out[f] for f in _FIELDS},
                "gold_flags": g["risk_flags"],
                "pred_flags": out["risk_flags"],
                "confidence": round(conf, 2),
            })
    elapsed = time.time() - t0

    stats = {
        "strategy": strategy,
        "n": len(golden),
        "n_images": n_images,
        "elapsed_s": round(elapsed, 1),
        "cost": client.cost_estimate(),
        "fields": {},
        "risk_flags": M.multilabel_metrics(pred_flag_sets, gold_flag_sets),
        "abstention": {"aurc": round(M.aurc(confidences, status_correct), 4)},
        "calibration": {
            "ece": round(M.expected_calibration_error(confidences, status_correct), 4),
            "brier": round(M.brier_score(confidences, status_correct), 4),
        },
        "failures": failures,
    }
    for fld in _FIELDS:
        entry = {"accuracy": round(M.accuracy(preds[fld], golds[fld]), 4)}
        if fld == "claim_status":
            entry["cohen_kappa"] = round(M.cohen_kappa(preds[fld], golds[fld]), 4)
            entry["balanced_accuracy"] = round(M.balanced_accuracy(preds[fld], golds[fld]), 4)
            entry["confusion"] = M.confusion_matrix(preds[fld], golds[fld], _STATUS_LABELS)
        stats["fields"][fld] = entry
    return stats


def _confusion_md(labels, matrix):
    head = "| gold \\ pred | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    rows = [f"| {labels[i]} | " + " | ".join(str(x) for x in matrix[i]) + " |" for i in range(len(labels))]
    return "\n".join([head, sep] + rows)


def render_report(stats) -> str:
    c = stats["cost"]
    per_claim = c["est_cost_usd"] / stats["n"] if stats["n"] else 0
    test_rows = 44
    est_test_cost = round(per_claim * test_rows, 2)
    L = [f"# Evaluation & Operational Report — Strategy {stats['strategy'].upper()}", ""]
    L.append(f"Golden set: `sample_claims.csv` ({stats['n']} rows). Prompts are zero-shot (no few-shot "
             "exemplars), so there is no train/eval leakage to leave out.")
    L += ["", "## 1. Field accuracy", "", "| field | accuracy | notes |", "|---|---|---|"]
    for fld in _FIELDS:
        e = stats["fields"][fld]
        note = ""
        if fld == "claim_status":
            note = f"kappa={e['cohen_kappa']}, balanced_acc={e['balanced_accuracy']} (3-class imbalanced — trust over raw acc)"
        L.append(f"| {fld} | {e['accuracy']} | {note} |")
    L += ["", "### claim_status confusion matrix", ""]
    labels, matrix = stats["fields"]["claim_status"]["confusion"]
    L.append(_confusion_md(labels, matrix))
    rf = stats["risk_flags"]
    L += ["", "## 2. risk_flags (multi-label)", ""]
    L.append(f"micro-F1 **{rf['micro_f1']:.3f}** · macro-F1 **{rf['macro_f1']:.3f}** · "
             f"Hamming loss **{rf['hamming_loss']:.3f}** · exact-match **{rf['exact_match']:.3f}**")
    L += ["", "## 3. Abstention & calibration", ""]
    L.append(f"- **AURC** (area under risk–coverage) **{stats['abstention']['aurc']}** — lower is better; "
             "does it abstain on the cases it would otherwise get wrong?")
    L.append(f"- **ECE** (10-bin) **{stats['calibration']['ece']}** · **Brier** **{stats['calibration']['brier']}**.")
    L += ["", "## 4. Error analysis — every disagreement", ""]
    if not stats["failures"]:
        L.append("No disagreements on the golden set.")
    else:
        L.append(f"{len(stats['failures'])} rows disagree on at least one field — read them, don't just count:")
        L.append("")
        for f in stats["failures"]:
            L.append(f"- **{f['user_id']}** (conf {f['confidence']}): "
                     f"status gold=`{f['gold']['claim_status']}`/pred=`{f['pred']['claim_status']}`, "
                     f"issue `{f['gold']['issue_type']}`/`{f['pred']['issue_type']}`, "
                     f"part `{f['gold']['object_part']}`/`{f['pred']['object_part']}`, "
                     f"sev `{f['gold']['severity']}`/`{f['pred']['severity']}`; "
                     f"flags gold=`{f['gold_flags']}` pred=`{f['pred_flags']}`.")
    L += ["", "## 5. Operational analysis", ""]
    L.append(f"- Model calls (sample run): **{c['calls']}** ({c['billed_calls']} billed, {c['cached_calls']} cached).")
    L.append(f"- Tokens: **{c['input_tokens']:,}** in / **{c['output_tokens']:,}** out.")
    L.append(f"- Images processed: **{stats['n_images']}**.")
    L.append(f"- Sample-set est. cost: **${c['est_cost_usd']}** (≈ ${per_claim:.4f}/claim).")
    L.append(f"- Extrapolated full test set ({test_rows} rows): **≈ ${est_test_cost}** "
             "(pricing assumptions in `config.yaml`; verify vs current OpenAI pricing).")
    L.append(f"- Wall-clock: **{stats['elapsed_s']}s** for {stats['n']} claims (serial).")
    L += ["", "**TPM/RPM & efficiency levers:** temperature 0 for determinism; per-image downscale to "
          "1024px; **response cache** keyed on (model+prompt+image) so re-runs and duplicate images are "
          "free; cheap model for extraction/inspection, flagship only for adjudication; one repair retry "
          "on schema-invalid output. Scale-up: token-bucket rate limiting + exponential backoff with "
          "jitter honoring `retry-after`; 429 (our quota) → slow down, 529 (provider overload) → back off "
          "/ optional fallback; batch the offline eval."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the pipeline on the golden set.")
    ap.add_argument("--strategy", choices=["a", "b"], default="b",
                    help="b = 3-stage scoped pipeline (default); a = single combined prompt")
    ap.add_argument("--dataset", default=os.path.join(_REPO, "dataset"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true",
                    help="cache-only: a cache miss raises instead of billing the API")
    ap.add_argument("--out", default=os.path.join(_EVAL, "evaluation_report.md"))
    args = ap.parse_args()

    stats = evaluate(args.strategy, args.dataset, args.limit, args.offline)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_report(stats) + "\n")
    print(f"strategy={stats['strategy']}  n={stats['n']}  "
          f"claim_status_acc={stats['fields']['claim_status']['accuracy']}  "
          f"cost=${stats['cost']['est_cost_usd']} ({stats['cost']['billed_calls']} billed)  -> {args.out}")


if __name__ == "__main__":
    main()
