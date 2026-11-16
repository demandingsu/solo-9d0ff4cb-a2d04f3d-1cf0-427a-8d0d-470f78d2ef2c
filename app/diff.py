"""Sample-level impact lists between two delivered reports.

Given an *old* report and the *new* report (the latest, non-stale one),
this module produces a deterministic impact list:

* samples are aligned by ``sample_id`` – new ids are *added*, missing ids
  are *removed*;
* ores and elements are aligned **by name**.  When the name sets differ the
  comparison is refused instead of silently pairing the wrong columns;
* each shared sample records changes of measured values, proportions (or
  the exact attainable interval when the optimum is not unique), predicted
  values and residuals, plus the SSE delta;
* when *both* inputs changed between the two reports, the two crossed
  combinations (old baseline × new samples, new baseline × old samples) are
  recomputed so every change can be labelled as one-side only, a joint
  effect, or – whenever a non-unique solution would otherwise be treated as
  a definite proportion – not attributable.

The module is a pure function of report payloads and the four input
snapshots; nothing here touches the database.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .compute import compute_report
from .inputs import BaselineInput, SamplesInput

ATTRIBUTION_LABELS = {
    "none": "无变化",
    "baseline_only": "基准单侧改动",
    "samples_only": "成品单侧改动",
    "joint": "共同影响",
    "unattributable": "无法归因",
}


class DiffError(ValueError):
    """The two reports cannot be compared."""


# ---------------------------------------------------------------------------
# Name-set alignment
# ---------------------------------------------------------------------------


def check_name_sets(
    old_payload: dict[str, Any], new_payload: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Refuse the comparison when ore/element name sets differ.

    Ordering of the returned canonical lists follows the old report, so the
    impact list is stable regardless of column reordering inside a CSV.
    """

    old_ores = list(old_payload["ores"])
    new_ores = list(new_payload["ores"])
    if set(old_ores) != set(new_ores):
        missing = [name for name in old_ores if name not in new_ores]
        extra = [name for name in new_ores if name not in old_ores]
        details = []
        if missing:
            details.append("新报告缺少原矿：" + "、".join(missing))
        if extra:
            details.append("新报告多出原矿：" + "、".join(extra))
        raise DiffError("两份报告的原矿名称集合不一致，拒绝对照；" + "；".join(details))

    old_elements = list(old_payload["elements"])
    new_elements = list(new_payload["elements"])
    if set(old_elements) != set(new_elements):
        missing = [name for name in old_elements if name not in new_elements]
        extra = [name for name in new_elements if name not in old_elements]
        details = []
        if missing:
            details.append("新报告缺少元素：" + "、".join(missing))
        if extra:
            details.append("新报告多出元素：" + "、".join(extra))
        raise DiffError("两份报告的元素名称集合不一致，拒绝对照；" + "；".join(details))

    return old_elements, old_ores


def _canonical_results(
    payload: dict[str, Any],
    elements_order: list[str],
    ores_order: list[str],
) -> dict[str, dict[str, Any]]:
    """Re-index every sample's arrays into the old report's name order."""

    element_index = [payload["elements"].index(name) for name in elements_order]
    ore_index = [payload["ores"].index(name) for name in ores_order]

    def reorder(values: list[Any] | None, index: list[int]) -> list[Any] | None:
        if values is None:
            return None
        return [values[i] for i in index]

    def reorder_ranges(
        ranges: list[list[Any]], index: list[int]
    ) -> list[list[Any]]:
        return [ranges[i] for i in index]

    result: dict[str, dict[str, Any]] = {}
    for row in payload["results"]:
        result[row["sample_id"]] = {
            "unique": row["unique"],
            "measured": reorder(row["measured"], element_index),
            "weights": reorder(row["weights"], ore_index),
            "weight_ranges": reorder_ranges(row["weight_ranges"], ore_index),
            "predicted": reorder(row["predicted"], element_index),
            "predicted_ranges": reorder_ranges(
                row["predicted_ranges"], element_index
            ),
            "residuals": reorder(row["residuals"], element_index),
            "residual_ranges": reorder_ranges(
                row["residual_ranges"], element_index
            ),
            "sse": row["sse"],
        }
    return result


# ---------------------------------------------------------------------------
# Crossed recomputation when both inputs changed
# ---------------------------------------------------------------------------


def _compact_result(row: dict[str, Any]) -> dict[str, Any]:
    """Strip fields that are redundant in the cross-combination payloads."""

    return {
        "unique": row["unique"],
        "weights": row["weights"],
        "weight_ranges": row["weight_ranges"],
        "predicted": row["predicted"],
        "predicted_ranges": row["predicted_ranges"],
        "residuals": row["residuals"],
        "residual_ranges": row["residual_ranges"],
        "sse": row["sse"],
    }


def compute_crosses(
    old_baseline: BaselineInput,
    old_samples: SamplesInput,
    new_baseline: BaselineInput,
    new_samples: SamplesInput,
) -> dict[str, dict[str, Any]]:
    """Recompute the two off-diagonal input combinations.

    ``b0s1`` keeps the old baseline with the new samples (isolated samples
    change); ``b1s0`` keeps the new baseline with the old samples (isolated
    baseline change).  Only common samples can appear in every combination,
    so the compact payloads are keyed by sample id.
    """

    crosses: dict[str, dict[str, Any]] = {}
    combinations = (
        ("b0s1", "旧基准 × 新成品（隔离成品改动）", old_baseline, new_samples),
        ("b1s0", "新基准 × 旧成品（隔离基准改动）", new_baseline, old_samples),
    )
    for key, label, baseline, samples in combinations:
        payload = compute_report(baseline, samples)
        crosses[key] = {
            "label": label,
            "results": {
                row["sample_id"]: _compact_result(row)
                for row in payload["results"]
            },
        }
    return crosses


# ---------------------------------------------------------------------------
# Change descriptors
# ---------------------------------------------------------------------------


def _point_descriptor(
    old_value: Any,
    new_value: Any,
    attribution: str = "none",
) -> dict[str, Any]:
    old_f = Fraction(old_value)
    new_f = Fraction(new_value)
    return {
        "kind": "point",
        "old": old_value,
        "new": new_value,
        "delta": str(new_f - old_f),
        "changed": new_f != old_f,
        "attribution": attribution,
    }


def _range_descriptor(
    old_range: list[Any], new_range: list[Any], attribution: str = "none"
) -> dict[str, Any]:
    old_pair = (Fraction(old_range[0]), Fraction(old_range[1]))
    new_pair = (Fraction(new_range[0]), Fraction(new_range[1]))
    return {
        "kind": "range",
        "old_range": list(old_range),
        "new_range": list(new_range),
        "changed": old_pair != new_pair,
        "attribution": attribution,
    }


def _scalar_attribution(
    v00: Fraction, v01: Fraction, v10: Fraction, v11: Fraction
) -> str:
    """Classify a scalar change using the crossed combinations.

    ``v01`` is the value with only the samples input changed, ``v10`` with
    only the baseline changed.  Both isolated effects are zero while the
    total changed cannot happen for a deterministic computation, so that
    case is reported as not attributable rather than mislabelled.
    """

    if v11 == v00:
        return "none"
    samples_effect = v01 != v00
    baseline_effect = v10 != v00
    if samples_effect and not baseline_effect:
        return "samples_only"
    if baseline_effect and not samples_effect:
        return "baseline_only"
    if samples_effect and baseline_effect:
        return "joint"
    return "unattributable"


def _attribution_for(
    *,
    baseline_changed: bool,
    samples_changed: bool,
    cross_samples_row: dict[str, Any] | None,
    cross_baseline_row: dict[str, Any] | None,
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    field: str,
    index: int,
    non_unique: bool,
) -> str:
    """Attribution for one component of one common sample."""

    # A non-unique optimum has no definite proportion/prediction/residual.
    # With both inputs moving there is no fact of the matter to assign, so
    # such a component is not attributable; with one input moving the cause
    # is unambiguous.
    if non_unique and baseline_changed and samples_changed:
        return "unattributable"

    def value_of(report_row: dict[str, Any], key: str, i: int) -> Fraction:
        return Fraction(report_row[key][i])

    # Single-side change: the cause is known without crossed recomputation.
    if baseline_changed and not samples_changed:
        v00 = value_of(old_row, field, index)
        v11 = value_of(new_row, field, index)
        return "baseline_only" if v11 != v00 else "none"
    if samples_changed and not baseline_changed:
        v00 = value_of(old_row, field, index)
        v11 = value_of(new_row, field, index)
        return "samples_only" if v11 != v00 else "none"
    if not baseline_changed and not samples_changed:
        return "none"

    # Both inputs changed: use the two crossed combinations.
    if cross_samples_row is None or cross_baseline_row is None:
        # pragma: no cover - both_changed always stores complete crosses
        return "unattributable"

    v00 = value_of(old_row, field, index)
    v11 = value_of(new_row, field, index)
    # b0s1 isolates the samples change; b1s0 isolates the baseline change.
    v01 = value_of(cross_samples_row, field, index)
    v10 = value_of(cross_baseline_row, field, index)
    return _scalar_attribution(v00, v01, v10, v11)


def _solution_component(
    field: str,
    range_field: str,
    index: int,
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    baseline_changed: bool,
    samples_changed: bool,
    cross_samples: dict[str, Any] | None,
    cross_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Descriptor for a weight / predicted / residual component."""

    both_changed = baseline_changed and samples_changed

    if old_row["unique"] and new_row["unique"]:
        attribution = _attribution_for(
            baseline_changed=baseline_changed,
            samples_changed=samples_changed,
            cross_samples_row=cross_samples,
            cross_baseline_row=cross_baseline,
            old_row=old_row,
            new_row=new_row,
            field=field,
            index=index,
            non_unique=False,
        )
        return _point_descriptor(
            old_row[field][index], new_row[field][index], attribution
        )

    # At least one side is non-unique: no definite number may be shown, so
    # only the exact attainable intervals are compared.  With a single
    # moving input the cause of an interval change is still unambiguous.
    old_pair = (
        Fraction(old_row[range_field][index][0]),
        Fraction(old_row[range_field][index][1]),
    )
    new_pair = (
        Fraction(new_row[range_field][index][0]),
        Fraction(new_row[range_field][index][1]),
    )
    if both_changed:
        attribution = "unattributable"
    elif baseline_changed:
        attribution = "baseline_only" if old_pair != new_pair else "none"
    elif samples_changed:
        attribution = "samples_only" if old_pair != new_pair else "none"
    else:
        attribution = "none"
    return _range_descriptor(
        old_row[range_field][index],
        new_row[range_field][index],
        attribution,
    )


def _sse_descriptor(
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    baseline_changed: bool,
    samples_changed: bool,
    cross_samples_row: dict[str, Any] | None,
    cross_baseline_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """SSE is always a definite scalar, even for non-unique optima."""

    v00 = Fraction(old_row["sse"])
    v11 = Fraction(new_row["sse"])
    if baseline_changed and samples_changed:
        v01 = Fraction(cross_samples_row["sse"])  # type: ignore[union-attr]
        v10 = Fraction(cross_baseline_row["sse"])  # type: ignore[union-attr]
        attribution = _scalar_attribution(v00, v01, v10, v11)
    elif baseline_changed:
        attribution = "baseline_only" if v11 != v00 else "none"
    elif samples_changed:
        attribution = "samples_only" if v11 != v00 else "none"
    else:
        attribution = "none"
    return _point_descriptor(
        old_row["sse"], new_row["sse"], attribution
    )


# ---------------------------------------------------------------------------
# Entry-level attribution summary
# ---------------------------------------------------------------------------

_ATTRIBUTION_RANK = {
    "none": 0,
    "baseline_only": 1,
    "samples_only": 1,
    "joint": 2,
    "unattributable": 3,
}


def _summarise_attribution(descriptors: list[dict[str, Any]]) -> str:
    labels = {d["attribution"] for d in descriptors}
    if "unattributable" in labels:
        return "unattributable"
    if "joint" in labels:
        return "joint"
    baseline_only = "baseline_only" in labels
    samples_only = "samples_only" in labels
    if baseline_only and samples_only:
        # Different components driven by different single inputs: the
        # sample-level effect is a joint one.
        return "joint"
    if baseline_only:
        return "baseline_only"
    if samples_only:
        return "samples_only"
    return "none"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_impact_list(
    old_report_id: int,
    new_report_id: int,
    old_payload: dict[str, Any],
    new_payload: dict[str, Any],
    *,
    baseline_changed: bool,
    samples_changed: bool,
    old_baseline: BaselineInput | None = None,
    old_samples: SamplesInput | None = None,
    new_baseline: BaselineInput | None = None,
    new_samples: SamplesInput | None = None,
) -> dict[str, Any]:
    """Build the full impact-list payload for one ordered report pair."""

    if old_report_id == new_report_id:
        raise DiffError("旧报告与新报告不能是同一份报告。")

    elements, ores = check_name_sets(old_payload, new_payload)
    old_rows = _canonical_results(old_payload, elements, ores)
    new_rows = _canonical_results(new_payload, elements, ores)

    both_changed = baseline_changed and samples_changed
    crosses: dict[str, dict[str, Any]] | None = None
    if both_changed:
        if None in (old_baseline, old_samples, new_baseline, new_samples):
            # pragma: no cover - endpoint always supplies every snapshot
            raise DiffError("两侧输入均变更时必须提供四份输入快照。")
        crosses = compute_crosses(
            old_baseline,  # type: ignore[arg-type]
            old_samples,  # type: ignore[arg-type]
            new_baseline,  # type: ignore[arg-type]
            new_samples,  # type: ignore[arg-type]
        )

    old_ids = list(old_rows)
    new_ids = list(new_rows)
    shared = [sample_id for sample_id in old_ids if sample_id in new_rows]
    added = [sample_id for sample_id in new_ids if sample_id not in old_rows]
    removed = [sample_id for sample_id in old_ids if sample_id not in new_rows]

    entries: list[dict[str, Any]] = []

    # Removed samples first (old report order), then shared, then added
    # (new report order) – a stable, reviewable ordering.
    for sample_id in removed:
        entries.append(
            _one_sided_entry(sample_id, "removed", old_rows[sample_id], samples_changed)
        )

    for sample_id in shared:
        entries.append(
            _shared_entry(
                sample_id,
                old_rows[sample_id],
                new_rows[sample_id],
                baseline_changed,
                samples_changed,
                crosses,
                len(elements),
                len(ores),
            )
        )

    for sample_id in added:
        entries.append(
            _one_sided_entry(sample_id, "added", new_rows[sample_id], samples_changed)
        )

    changed_entries = [entry for entry in entries if entry["has_changes"]]
    non_unique_pairs = sum(
        1
        for entry in entries
        if entry["status"] == "common"
        and (not entry["_old_unique"] or not entry["_new_unique"])
    )

    # The uniqueness flags were bookkeeping for the summary; the public
    # payload communicates the same information through range descriptors.
    for entry in entries:
        entry.pop("_old_unique", None)
        entry.pop("_new_unique", None)

    return {
        "schema_version": 1,
        "elements": elements,
        "ores": ores,
        "old_report_id": old_report_id,
        "new_report_id": new_report_id,
        "baseline_changed": baseline_changed,
        "samples_changed": samples_changed,
        "both_changed": both_changed,
        "crosses": crosses,
        "summary": {
            "total": len(entries),
            "added": len(added),
            "removed": len(removed),
            "shared": len(shared),
            "changed": len(changed_entries),
            "non_unique_pairs": non_unique_pairs,
        },
        "entries": entries,
    }


def _one_sided_entry(
    sample_id: str,
    status: str,
    row: dict[str, Any],
    samples_changed: bool,
) -> dict[str, Any]:
    """An added or removed sample exists only on one side."""

    # Presence/absence of a sample is driven by the samples input.
    attribution = "samples_only" if samples_changed else "none"
    entry: dict[str, Any] = {
        "sample_id": sample_id,
        "status": status,
        "attribution": attribution,
        "has_changes": True,
        "_old_unique": row["unique"] if status == "removed" else None,
        "_new_unique": row["unique"] if status == "added" else None,
        "measured": [],
        "weights": [],
        "predicted": [],
        "residuals": [],
        "sse": None,
    }
    for value in row["measured"]:
        entry["measured"].append(
            {
                "kind": "point",
                "old": value if status == "removed" else None,
                "new": value if status == "added" else None,
                "delta": None,
                "changed": True,
                "attribution": attribution,
            }
        )
    for index, _ore in enumerate(row["weight_ranges"]):
        if row["unique"]:
            entry["weights"].append(
                {
                    "kind": "point",
                    "old": row["weights"][index] if status == "removed" else None,
                    "new": row["weights"][index] if status == "added" else None,
                    "delta": None,
                    "changed": True,
                    "attribution": attribution,
                }
            )
        else:
            entry["weights"].append(
                {
                    "kind": "range",
                    "old_range": (
                        row["weight_ranges"][index] if status == "removed" else None
                    ),
                    "new_range": (
                        row["weight_ranges"][index] if status == "added" else None
                    ),
                    "changed": True,
                    "attribution": attribution,
                }
            )
    for index in range(len(row["predicted_ranges"])):
        if row["unique"]:
            entry["predicted"].append(
                {
                    "kind": "point",
                    "old": (
                        row["predicted"][index] if status == "removed" else None
                    ),
                    "new": row["predicted"][index] if status == "added" else None,
                    "delta": None,
                    "changed": True,
                    "attribution": attribution,
                }
            )
            entry["residuals"].append(
                {
                    "kind": "point",
                    "old": (
                        row["residuals"][index] if status == "removed" else None
                    ),
                    "new": (
                        row["residuals"][index] if status == "added" else None
                    ),
                    "delta": None,
                    "changed": True,
                    "attribution": attribution,
                }
            )
        else:
            entry["predicted"].append(
                {
                    "kind": "range",
                    "old_range": (
                        row["predicted_ranges"][index]
                        if status == "removed"
                        else None
                    ),
                    "new_range": (
                        row["predicted_ranges"][index]
                        if status == "added"
                        else None
                    ),
                    "changed": True,
                    "attribution": attribution,
                }
            )
            entry["residuals"].append(
                {
                    "kind": "range",
                    "old_range": (
                        row["residual_ranges"][index]
                        if status == "removed"
                        else None
                    ),
                    "new_range": (
                        row["residual_ranges"][index]
                        if status == "added"
                        else None
                    ),
                    "changed": True,
                    "attribution": attribution,
                }
            )
    entry["sse"] = {
        "kind": "point",
        "old": row["sse"] if status == "removed" else None,
        "new": row["sse"] if status == "added" else None,
        "delta": None,
        "changed": True,
        "attribution": attribution,
    }
    return entry


def _shared_entry(
    sample_id: str,
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    baseline_changed: bool,
    samples_changed: bool,
    crosses: dict[str, dict[str, Any]] | None,
    element_count: int,
    ore_count: int,
) -> dict[str, Any]:
    cross_samples = crosses["b0s1"] if crosses else None
    cross_baseline = crosses["b1s0"] if crosses else None
    # Cross rows only exist for samples present in both samples inputs; for
    # shared samples (present in both reports) that is always the case.
    cs_row = (
        cross_samples["results"].get(sample_id) if cross_samples else None
    )
    cb_row = (
        cross_baseline["results"].get(sample_id) if cross_baseline else None
    )
    if crosses and (cs_row is None or cb_row is None):  # pragma: no cover
        raise DiffError(f"交叉组合缺少样本 {sample_id} 的结果。")

    measured: list[dict[str, Any]] = []
    for index in range(element_count):
        old_value = old_row["measured"][index]
        new_value = new_row["measured"][index]
        # Measured values come from the samples input alone.
        attribution = (
            "samples_only"
            if samples_changed and Fraction(new_value) != Fraction(old_value)
            else "none"
        )
        measured.append(_point_descriptor(old_value, new_value, attribution))

    weights = [
        _solution_component(
            "weights",
            "weight_ranges",
            index,
            old_row,
            new_row,
            baseline_changed,
            samples_changed,
            cs_row,
            cb_row,
        )
        for index in range(ore_count)
    ]
    predicted = [
        _solution_component(
            "predicted",
            "predicted_ranges",
            index,
            old_row,
            new_row,
            baseline_changed,
            samples_changed,
            cs_row,
            cb_row,
        )
        for index in range(element_count)
    ]
    residuals = [
        _solution_component(
            "residuals",
            "residual_ranges",
            index,
            old_row,
            new_row,
            baseline_changed,
            samples_changed,
            cs_row,
            cb_row,
        )
        for index in range(element_count)
    ]
    sse = _sse_descriptor(
        old_row,
        new_row,
        baseline_changed,
        samples_changed,
        cs_row,
        cb_row,
    )

    all_descriptors = measured + weights + predicted + residuals + [sse]
    has_changes = any(d["changed"] for d in all_descriptors)
    attribution = (
        _summarise_attribution(all_descriptors)
        if has_changes
        else "none"
    )

    return {
        "sample_id": sample_id,
        "status": "common",
        "attribution": attribution,
        "has_changes": has_changes,
        "_old_unique": old_row["unique"],
        "_new_unique": new_row["unique"],
        "measured": measured,
        "weights": weights,
        "predicted": predicted,
        "residuals": residuals,
        "sse": sse,
    }
