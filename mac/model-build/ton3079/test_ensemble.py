#!/usr/bin/env python3
"""TON-3090 — unit tests for the escalate-only ensemble join (no GPU needed).

Proves the union + escalate-only semantics that sec_eval.ensemble_join implements,
so that when the 8B per-item cache is regenerated in a GPU window the reported
ensemble row is trustworthy. Run: python3 test_ensemble.py
"""
import sec_eval as SE

# minimal synthetic corpus: 2 crit/high malicious, 2 benign, 1 adversarial malicious
RECORDS = [
    {"id": "m1", "malicious": True,  "severity": "critical", "adversarial": False},
    {"id": "m2", "malicious": True,  "severity": "high",     "adversarial": False},
    {"id": "a1", "malicious": True,  "severity": "high",     "adversarial": True},
    {"id": "b1", "malicious": False, "severity": "none",     "adversarial": False},
    {"id": "b2", "malicious": False, "severity": "none",     "adversarial": False},
]


def P(action, undecided=False, lat=1.0):
    return {"action": action, "undecided": undecided, "latency_ms": lat}


def test_escalate_only_suppresses_flag_on_benign():
    # right (8B) over-flags both benign with 'flag' but never 'escalate' them.
    left = {r["id"]: P("allow") for r in RECORDS}          # det: everything allow
    right = {"m1": P("escalate"), "m2": P("escalate"), "a1": P("escalate"),
             "b1": P("flag"), "b2": P("flag")}              # 8B flags benign
    # raw union would flag both benign (FPR 1.0); escalate-only must suppress them.
    joined = SE.ensemble_join(RECORDS, left, right, right_escalate_only=True)
    assert joined["b1"]["action"] == "allow", joined["b1"]
    assert joined["b2"]["action"] == "allow", joined["b2"]
    assert joined["m1"]["action"] == "escalate"
    m = SE.evaluate(RECORDS, joined)
    assert m["benign_fpr"] == 0.0, m["benign_fpr"]
    assert m["critical_high_recall"] == 1.0, m["critical_high_recall"]
    print("ok escalate_only suppresses benign flag -> FPR 0.0, recall 1.0")


def test_union_is_complementary():
    # det catches m1 only; 8B-escalate catches m2 only. Union must catch both.
    left = {"m1": P("escalate"), "m2": P("allow"), "a1": P("allow"),
            "b1": P("allow"), "b2": P("allow")}
    right = {"m1": P("allow"), "m2": P("escalate"), "a1": P("escalate"),
             "b1": P("allow"), "b2": P("allow")}
    joined = SE.ensemble_join(RECORDS, left, right, right_escalate_only=True)
    m = SE.evaluate(RECORDS, joined)
    assert m["critical_high_recall"] == 1.0, m           # m1∪m2 both caught
    assert m["adversarial_recall"] == 1.0, m             # a1 caught by right
    assert m["benign_fpr"] == 0.0, m
    print("ok union is complementary -> recall exceeds either channel alone")


def test_flag_kept_when_not_escalate_only():
    # without escalate_only, a 'flag' still counts as positive (risky).
    left = {r["id"]: P("allow") for r in RECORDS}
    right = {r["id"]: P("allow") for r in RECORDS}
    right["b1"] = P("flag")
    joined = SE.ensemble_join(RECORDS, left, right, right_escalate_only=False)
    assert joined["b1"]["action"] == "escalate", joined["b1"]  # flag -> positive
    print("ok flag counts as positive when escalate_only is off")


def test_latency_is_serial_sum_and_undecided_propagates():
    left = {r["id"]: P("allow", lat=2.0) for r in RECORDS}
    right = {r["id"]: P("allow", lat=3.0) for r in RECORDS}
    right["b1"] = P("escalate", undecided=True, lat=3.0)
    joined = SE.ensemble_join(RECORDS, left, right, right_escalate_only=True)
    assert joined["m1"]["latency_ms"] == 5.0, joined["m1"]
    assert joined["b1"]["undecided"] is True, joined["b1"]
    print("ok latency = serial sum; undecided propagates")


def test_missing_channel_fails_safe_positive():
    left = {r["id"]: P("allow") for r in RECORDS}
    right = {}  # channel produced nothing for m1 -> must fail safe to positive
    joined = SE.ensemble_join(RECORDS, left, right, right_escalate_only=True)
    assert joined["m1"]["action"] == "escalate", joined["m1"]
    print("ok missing channel result fails safe to positive")


if __name__ == "__main__":
    test_escalate_only_suppresses_flag_on_benign()
    test_union_is_complementary()
    test_flag_kept_when_not_escalate_only()
    test_latency_is_serial_sum_and_undecided_propagates()
    test_missing_channel_fails_safe_positive()
    print("\nALL PASS (5/5)")
