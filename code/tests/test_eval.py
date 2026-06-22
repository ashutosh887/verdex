import math
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL = os.path.join(_CODE, "evaluation")
for _p in (_CODE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics as M  # noqa: E402
import pytest  # noqa: E402
from strategies import CombinedOutput  # noqa: E402

from pipeline.vlm import strict_json_schema  # noqa: E402


def test_accuracy_and_confusion():
    preds = ["a", "b", "a", "c"]
    golds = ["a", "b", "c", "c"]
    assert M.accuracy(preds, golds) == 0.75
    labels, m = M.confusion_matrix(preds, golds, ["a", "b", "c"])
    assert m[0][0] == 1  # gold a, pred a
    assert m[2][0] == 1  # gold c, pred a
    assert m[2][2] == 1  # gold c, pred c


def test_multilabel_metrics_perfect_and_partial():
    perfect = M.multilabel_metrics([["x", "y"]], [["x", "y"]])
    assert perfect["exact_match"] == 1.0 and perfect["micro_f1"] == 1.0
    partial = M.multilabel_metrics([["x"]], [["x", "y"]])
    assert partial["exact_match"] == 0.0
    assert 0.0 < partial["micro_f1"] < 1.0


def test_risk_coverage_and_aurc():
    # perfectly ranked: confident ones are correct, unsure ones wrong -> low AURC
    conf = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    cov, risk = M.risk_coverage_curve(conf, correct)
    assert cov[-1] == 1.0
    assert risk[0] == 0.0  # most confident is correct
    good = M.aurc(conf, correct)
    bad = M.aurc(conf, [False, False, True, True])  # anti-correlated
    assert good < bad


def test_calibration_perfect_is_low_ece():
    conf = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert M.expected_calibration_error(conf, correct) == 0.0
    assert M.brier_score(conf, correct) == 0.0
    bad = M.brier_score([1.0, 1.0], [False, False])
    assert bad == 1.0


def test_kappa_and_balanced_accuracy():
    assert M.balanced_accuracy(["a", "b"], ["a", "b"]) == 1.0
    k = M.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert abs(k - 1.0) < 1e-9 or math.isnan(k) is False


def _walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for v in node.values():
            yield from _walk_objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_objects(v)


def test_combined_strategy_schema_is_strict():
    schema = strict_json_schema(CombinedOutput)
    objs = list(_walk_objects(schema))
    assert objs
    for obj in objs:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"].keys())


def test_judge_schema_is_strict_and_range_validated():
    from judge import JustificationJudgment

    schema = strict_json_schema(JustificationJudgment)
    for obj in _walk_objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"].keys())
    ok = JustificationJudgment(groundedness=5, faithfulness=4, verdict_consistency=3,
                               claim_relevance=5, rationale="grounded")
    assert ok.groundedness == 5
    with pytest.raises(Exception):
        JustificationJudgment(groundedness=6, faithfulness=4, verdict_consistency=3,
                              claim_relevance=5, rationale="out of range")


def _load_eval_main():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_main", os.path.join(_EVAL, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_report_smoke():
    render_report = _load_eval_main().render_report

    stats = {
        "strategy": "b", "n": 2, "n_images": 3, "elapsed_s": 1.0,
        "cost": {"calls": 6, "billed_calls": 6, "cached_calls": 0,
                 "input_tokens": 100, "output_tokens": 20, "est_cost_usd": 0.01},
        "fields": {
            "claim_status": {"accuracy": 1.0, "cohen_kappa": 1.0, "balanced_accuracy": 1.0,
                             "confusion": (["supported", "contradicted", "not_enough_information"],
                                           [[1, 0, 0], [0, 1, 0], [0, 0, 0]])},
            "issue_type": {"accuracy": 1.0},
            "object_part": {"accuracy": 1.0},
            "severity": {"accuracy": 0.5},
        },
        "risk_flags": {"micro_f1": 1.0, "macro_f1": 1.0, "hamming_loss": 0.0, "exact_match": 1.0},
        "abstention": {"aurc": 0.0},
        "calibration": {"ece": 0.0, "brier": 0.0},
        "failures": [],
    }
    md = render_report(stats)
    assert "Evaluation & Operational Report" in md
    assert "confusion matrix" in md
    assert "Operational analysis" in md


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
