"""Unit tests for the report-pair impact-list construction (app.diff).

The diff module is a pure function of report payloads and input snapshots.
These tests cover name-set alignment, added/removed samples, exact point
deltas, non-unique interval handling, crossed-recomputation attribution
(baseline-only / samples-only / joint / unattributable) and the rule that
a non-unique solution is never treated as a definite proportion.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from app.compute import compute_report
from app.diff import (
    ATTRIBUTION_LABELS,
    DiffError,
    build_impact_list,
    check_name_sets,
)
from app.inputs import BaselineInput, SamplesInput

ELEMENTS = ("Fe", "SiO2", "Al2O3", "CaO")
ORES = ("A", "B", "C")


def baseline(rows, ores=ORES, elements=ELEMENTS) -> BaselineInput:
    return BaselineInput(
        tuple(elements),
        tuple(ores),
        tuple(tuple(Fraction(v) for v in row) for row in rows),
    )


def samples(sample_rows, elements=ELEMENTS) -> SamplesInput:
    ids = tuple(sample_id for sample_id, _ in sample_rows)
    rows = tuple(
        tuple(Fraction(v) for v in values) for _, values in sample_rows
    )
    return SamplesInput(tuple(elements), ids, rows)


B0 = baseline(
    [[62, 4, 2, 1], [68, 3, 1, 1], [55, 6, 3, 2]]
)
B1 = baseline(
    [[63, 4.2, 2, 1], [67, 2.8, 1, 1.1], [56, 5.8, 3.1, 2]]
)
S0 = samples([("X", [63, 4.3, 2.1, 1.2])])
S1 = samples([("X", [64, 4.0, 2.3, 1.0])])


def _pair(
    old_baseline=B0,
    old_samples=S0,
    new_baseline=B1,
    new_samples=S1,
    *,
    baseline_changed=True,
    samples_changed=True,
):
    old_payload = compute_report(old_baseline, old_samples)
    new_payload = compute_report(new_baseline, new_samples)
    return build_impact_list(
        1,
        2,
        old_payload,
        new_payload,
        baseline_changed=baseline_changed,
        samples_changed=samples_changed,
        old_baseline=old_baseline,
        old_samples=old_samples,
        new_baseline=new_baseline,
        new_samples=new_samples,
    )


# ---------------------------------------------------------------------------
# Name-set alignment
# ---------------------------------------------------------------------------


def test_ore_name_set_mismatch_is_rejected():
    old_payload = compute_report(B0, S0)
    renamed = baseline(
        [[63, 4.2, 2, 1], [67, 2.8, 1, 1.1], [56, 5.8, 3.1, 2]],
        ores=("A", "B", "D"),
    )
    new_payload = compute_report(renamed, S1)
    with pytest.raises(DiffError, match="原矿名称集合不一致"):
        build_impact_list(
            1, 2, old_payload, new_payload,
            baseline_changed=True, samples_changed=False,
        )


def test_element_name_set_mismatch_is_rejected():
    old_payload = compute_report(B0, S0)
    other_elements = ("Fe", "SiO2", "MgO", "CaO")
    new_baseline = baseline(
        [[63, 4.2, 2, 1], [67, 2.8, 1, 1.1], [56, 5.8, 3.1, 2]],
        elements=other_elements,
    )
    new_samples = samples([("X", [64, 4.0, 2.3, 1.0])], elements=other_elements)
    new_payload = compute_report(new_baseline, new_samples)
    with pytest.raises(DiffError, match="元素名称集合不一致"):
        build_impact_list(
            1, 2, old_payload, new_payload,
            baseline_changed=True, samples_changed=True,
            old_baseline=B0, old_samples=S0,
            new_baseline=new_baseline, new_samples=new_samples,
        )


def test_reordered_element_columns_align_by_name():
    reordered_elements = ("CaO", "Fe", "SiO2", "Al2O3")
    new_samples = samples(
        [("X", [1.0, 64, 4.0, 2.3])], elements=reordered_elements
    )
    old_payload = compute_report(B0, S0)
    new_payload = compute_report(B0, new_samples)
    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=False, samples_changed=True,
        old_baseline=B0, old_samples=S0,
        new_baseline=B0, new_samples=new_samples,
    )
    # Canonical order follows the old report regardless of CSV ordering.
    assert impact["elements"] == list(ELEMENTS)
    entry = impact["entries"][0]
    # Fe is the first canonical element and changed from 63 to 64.
    assert entry["measured"][0]["old"] == "63"
    assert entry["measured"][0]["new"] == "64"
    assert Fraction(entry["measured"][0]["delta"]) == Fraction(1)


def test_check_name_sets_returns_canonical_order():
    old_payload = compute_report(B0, S0)
    new_payload = compute_report(B1, S1)
    elements, ores = check_name_sets(old_payload, new_payload)
    assert elements == list(ELEMENTS)
    assert ores == list(ORES)


def test_same_report_pair_is_rejected():
    payload = compute_report(B0, S0)
    with pytest.raises(DiffError, match="同一份报告"):
        build_impact_list(
            1, 1, payload, payload,
            baseline_changed=False, samples_changed=False,
        )


# ---------------------------------------------------------------------------
# Added / removed samples
# ---------------------------------------------------------------------------


def test_added_and_removed_samples_are_listed_by_id():
    old_samples = samples(
        [("X", [63, 4.3, 2.1, 1.2]), ("GONE", [60, 5, 2, 1.4])]
    )
    new_samples = samples(
        [("X", [64, 4.0, 2.3, 1.0]), ("NEW", [59, 5.2, 2.6, 1.6])]
    )
    old_payload = compute_report(B0, old_samples)
    new_payload = compute_report(B0, new_samples)
    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=False, samples_changed=True,
        old_baseline=B0, old_samples=old_samples,
        new_baseline=B0, new_samples=new_samples,
    )
    summary = impact["summary"]
    assert summary == {
        "total": 3,
        "added": 1,
        "removed": 1,
        "shared": 1,
        "changed": 3,
        "non_unique_pairs": 0,
    }
    by_id = {entry["sample_id"]: entry for entry in impact["entries"]}
    assert by_id["GONE"]["status"] == "removed"
    assert by_id["NEW"]["status"] == "added"
    assert by_id["X"]["status"] == "common"
    # Removed entries keep old values only; added entries keep new values only.
    assert by_id["GONE"]["measured"][0]["old"] is not None
    assert by_id["GONE"]["measured"][0]["new"] is None
    assert by_id["NEW"]["measured"][0]["old"] is None
    assert by_id["NEW"]["measured"][0]["new"] is not None
    # Presence changes are driven by the samples input.
    assert by_id["GONE"]["attribution"] == "samples_only"
    assert by_id["NEW"]["attribution"] == "samples_only"


# ---------------------------------------------------------------------------
# Point deltas and single-side attribution
# ---------------------------------------------------------------------------


def test_baseline_only_change_attributes_every_solver_delta_to_baseline():
    impact = _pair(
        new_baseline=B1, new_samples=S0,
        baseline_changed=True, samples_changed=False,
    )
    entry = impact["entries"][0]
    assert entry["status"] == "common"
    # Measured values did not move.
    assert all(d["attribution"] == "none" for d in entry["measured"])
    # Solver outputs that moved are baseline-only.
    assert all(
        d["attribution"] in ("none", "baseline_only")
        for d in entry["weights"] + entry["predicted"] + entry["residuals"]
    )
    assert any(
        d["attribution"] == "baseline_only" and d["changed"]
        for d in entry["weights"]
    )
    assert entry["attribution"] == "baseline_only"


def test_samples_only_change_attributes_measured_and_solver_to_samples():
    impact = _pair(
        new_baseline=B0, new_samples=S1,
        baseline_changed=False, samples_changed=True,
    )
    entry = impact["entries"][0]
    assert any(d["attribution"] == "samples_only" for d in entry["measured"])
    assert all(
        d["attribution"] in ("none", "samples_only")
        for d in entry["weights"] + entry["measured"]
    )
    assert entry["attribution"] == "samples_only"
    # Point delta is the exact rational difference.
    measured_fe = entry["measured"][0]
    assert measured_fe["old"] == "63" and measured_fe["new"] == "64"
    assert Fraction(measured_fe["delta"]) == Fraction(1)
    assert measured_fe["changed"] is True


def test_unchanged_shared_sample_has_no_changes():
    same = samples([("X", [63, 4.3, 2.1, 1.2])])
    old_payload = compute_report(B0, S0)
    new_payload = compute_report(B0, same)
    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=False, samples_changed=True,
        old_baseline=B0, old_samples=S0,
        new_baseline=B0, new_samples=same,
    )
    entry = impact["entries"][0]
    assert entry["has_changes"] is False
    assert entry["attribution"] == "none"
    assert all(d["changed"] is False for d in entry["measured"])
    assert impact["summary"]["changed"] == 0


# ---------------------------------------------------------------------------
# Both-side changes and crossed recomputation
# ---------------------------------------------------------------------------


def test_both_side_change_stores_two_crosses_and_joint_label():
    impact = _pair()
    assert impact["both_changed"] is True
    assert set(impact["crosses"]) == {"b0s1", "b1s0"}
    entry = impact["entries"][0]
    labels = {
        d["attribution"]
        for d in (
            entry["weights"]
            + entry["predicted"]
            + entry["residuals"]
            + entry["measured"]
            + [entry["sse"]]
        )
    }
    # The constructed noisy sample responds to both inputs, so at least one
    # component must be a joint effect.
    assert "joint" in labels
    # Measured values remain attributable to the samples side alone.
    assert "baseline_only" not in {d["attribution"] for d in entry["measured"]}
    # Cross labels exist for every attribution used in the payload.
    for label in labels:
        assert label in ATTRIBUTION_LABELS


def test_crosses_isolate_single_side_effects():
    """b0s1 must use the new samples; b1s0 the new baseline."""

    old_payload = compute_report(B0, S0)
    new_payload = compute_report(B1, S1)
    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=True, samples_changed=True,
        old_baseline=B0, old_samples=S0,
        new_baseline=B1, new_samples=S1,
    )
    b0s1 = compute_report(B0, S1)["results"][0]
    b1s0 = compute_report(B1, S0)["results"][0]
    assert impact["crosses"]["b0s1"]["results"]["X"]["weights"] == (
        b0s1["weights"]
    )
    assert impact["crosses"]["b1s0"]["results"]["X"]["weights"] == (
        b1s0["weights"]
    )


def test_non_unique_pair_is_never_a_definite_proportion():
    ambiguous_b0 = baseline(
        [[60, 4, 2, 1], [60, 4, 2, 1], [50, 6, 3, 2]]
    )
    ambiguous_b1 = baseline(
        [[61, 4, 2, 1], [61, 4, 2, 1], [50, 6, 3, 2]]
    )
    old_samples = samples([("AMBIG", [55, 5, 2.5, 1.5])])
    new_samples = samples([("AMBIG", [56, 5, 2.5, 1.5])])
    old_payload = compute_report(ambiguous_b0, old_samples)
    new_payload = compute_report(ambiguous_b1, new_samples)
    assert old_payload["results"][0]["unique"] is False
    assert new_payload["results"][0]["unique"] is False

    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=True, samples_changed=True,
        old_baseline=ambiguous_b0, old_samples=old_samples,
        new_baseline=ambiguous_b1, new_samples=new_samples,
    )
    assert impact["summary"]["non_unique_pairs"] == 1
    entry = impact["entries"][0]
    # Range descriptors only – no chosen weight/prediction/residual.
    assert all(d["kind"] == "range" for d in entry["weights"])
    assert all(d["kind"] == "range" for d in entry["predicted"])
    assert all(d["kind"] == "range" for d in entry["residuals"])
    assert all(
        d["attribution"] == "unattributable"
        for d in entry["weights"] + entry["predicted"] + entry["residuals"]
    )
    assert entry["attribution"] == "unattributable"
    # Measured values and SSE stay definite.
    assert all(d["kind"] == "point" for d in entry["measured"])
    assert entry["sse"]["kind"] == "point"
    assert entry["measured"][0]["attribution"] == "samples_only"


def test_non_unique_under_single_side_change_is_attributable():
    ambiguous_b0 = baseline(
        [[60, 4, 2, 1], [60, 4, 2, 1], [50, 6, 3, 2]]
    )
    old_samples = samples([("AMBIG", [55, 5, 2.5, 1.5])])
    new_samples = samples([("AMBIG", [56, 5, 2.5, 1.5])])
    old_payload = compute_report(ambiguous_b0, old_samples)
    new_payload = compute_report(ambiguous_b0, new_samples)
    impact = build_impact_list(
        1, 2, old_payload, new_payload,
        baseline_changed=False, samples_changed=True,
        old_baseline=ambiguous_b0, old_samples=old_samples,
        new_baseline=ambiguous_b0, new_samples=new_samples,
    )
    entry = impact["entries"][0]
    # Range descriptor, but the single moving input is still known.
    assert entry["weights"][0]["kind"] == "range"
    changed = [d for d in entry["weights"] if d["changed"]]
    assert changed
    assert all(d["attribution"] == "samples_only" for d in changed)
    assert impact["crosses"] is None
