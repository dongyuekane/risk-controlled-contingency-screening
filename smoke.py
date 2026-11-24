"""Recompute the AC outcome and the feedback score of one archived sample."""

import json

import feedback_guard_experiment as feedback
import numpy as np
import pandas as pd


def main():
    saved = feedback.HERE / "fresh_release/confirmation"
    raw = json.loads((saved / "confirmation_inputs.json").read_text())[0]
    case = pd.read_csv(saved / "confirmation_cases.csv").iloc[0]
    assert int(case["pair"]) == raw["pair"]

    outage = feedback.CONT[raw["event"]]
    ac = feedback.coupled30(raw, outage)
    first_pass = feedback.coupled30(raw, outage, linear=True)
    score = max(first_pass["violation"], feedback.gate(raw, first_pass))
    assert ac["status"] == case["status"]
    assert int(not ac["safe"]) == case["unsafe"]
    np.testing.assert_allclose(ac["violation"], case["ac_margin"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(score, case["feedback_score"], atol=1e-7, rtol=0)
    print(f"Sample {raw['pair']} reproduced with AC margin {ac['violation']:.8f}")


if __name__ == "__main__":
    main()
