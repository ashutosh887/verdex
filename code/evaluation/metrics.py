from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def confusion_matrix(preds: Sequence[str], golds: Sequence[str], labels: List[str]):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for p, g in zip(preds, golds):
        if g in idx and p in idx:
            m[idx[g]][idx[p]] += 1
    return labels, m


def cohen_kappa(preds: Sequence[str], golds: Sequence[str]) -> float:
    from sklearn.metrics import cohen_kappa_score

    if len(set(golds)) < 2:
        return float("nan")
    return float(cohen_kappa_score(list(golds), list(preds)))


def balanced_accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(list(golds), list(preds)))


def multilabel_metrics(
    pred_sets: Sequence[Sequence[str]], gold_sets: Sequence[Sequence[str]]
) -> Dict[str, float]:
    from sklearn.metrics import f1_score, hamming_loss
    from sklearn.preprocessing import MultiLabelBinarizer

    universe = sorted({l for s in list(pred_sets) + list(gold_sets) for l in s})
    if not universe:
        return {"micro_f1": 1.0, "macro_f1": 1.0, "hamming_loss": 0.0, "exact_match": 1.0}
    mlb = MultiLabelBinarizer(classes=universe)
    y_pred = mlb.fit_transform([list(s) for s in pred_sets])
    y_gold = mlb.transform([list(s) for s in gold_sets])
    exact = sum(set(p) == set(g) for p, g in zip(pred_sets, gold_sets)) / len(pred_sets)
    return {
        "micro_f1": float(f1_score(y_gold, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_gold, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_gold, y_pred)),
        "exact_match": float(exact),
    }


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> Tuple[List[float], List[float]]:
    """Order predictions most-confident-first; report (coverage, risk) as we admit each.
    A well-calibrated abstainer keeps risk low at low coverage and lets it rise only as it is
    forced to answer the cases it is unsure about."""
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    n = len(order)
    cov, risk, errs = [], [], 0
    for k, i in enumerate(order, start=1):
        if not correct[i]:
            errs += 1
        cov.append(k / n)
        risk.append(errs / k)
    return cov, risk


def aurc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    cov, risk = risk_coverage_curve(confidences, correct)
    if len(cov) < 2:
        return float(risk[0]) if risk else 0.0
    area = 0.0
    for i in range(1, len(cov)):
        area += (cov[i] - cov[i - 1]) * (risk[i] + risk[i - 1]) / 2
    return area / (cov[-1] - cov[0])


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return ece


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct)) / len(confidences)
