"""End-to-end tests for report comparison and saved impact lists.

Covers the web workflow (select old report + latest non-stale report,
generate and view a sample-level impact list), the publication contract
(inputs or the latest report changing while a list is being generated must
block publication), write-failure preservation of previously saved lists,
and the persistence rules (one row per ordered report pair, still readable
after the inputs keep changing).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import Store, StoreError
from app.main import create_app

BASE_CSV = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

BASE_CSV_2 = """ore,Fe,SiO2,Al2O3,CaO
A,63.3,3.7,2.0,1.1
B,68.5,2.5,1.1,0.8
C,55.2,6.2,3.4,2.1
"""

SAMPLES_CSV_1 = """sample_id,Fe,SiO2,Al2O3,CaO
S1,62.74,3.71,2.02,1.13
S2,60,4,2,1
"""

SAMPLES_CSV_2 = """sample_id,Fe,SiO2,Al2O3,CaO
S1,63.0,3.8,2.1,1.1
S3,59,5,2.5,1.5
"""

NON_UNIQUE_BASE = """ore,Fe,SiO2,Al2O3,CaO
A,60,4,2,1
B,60,4,2,1
C,50,6,3,2
"""


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(str(tmp_path / "impact.db"))
    with TestClient(app) as test_client:
        test_client.app_state = app.state  # type: ignore[attr-defined]
        yield test_client


def upload(client: TestClient, baseline: str | None = None,
           samples: str | None = None):
    files = {}
    if baseline is not None:
        files["baseline_file"] = ("baseline.csv", baseline.encode(), "text/csv")
    if samples is not None:
        files["samples_file"] = ("samples.csv", samples.encode(), "text/csv")
    return client.post(
        "/inputs/upload", files=files, follow_redirects=False
    )


def compute(client: TestClient):
    return client.post("/compute", follow_redirects=False)


def generate(client: TestClient, old_id: int, new_id: int,
             follow_redirects: bool = False):
    return client.post(
        "/impacts/generate",
        data={"old_report_id": str(old_id), "new_report_id": str(new_id)},
        follow_redirects=follow_redirects,
    )


def make_report_chain(client: TestClient) -> None:
    """Publish report #1 (both inputs) then report #2 after samples change."""

    upload(client, BASE_CSV, SAMPLES_CSV_1)
    assert compute(client).status_code == 303
    upload(client, samples=SAMPLES_CSV_2)
    assert compute(client).status_code == 303


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def test_index_shows_comparison_section_and_empty_state(client: TestClient):
    html = client.get("/").text
    assert "报告对照" in html
    assert "至少需要两份报告" in html


def test_compare_form_lists_old_reports_and_only_fresh_latest(
    client: TestClient,
):
    make_report_chain(client)
    # A third upload makes report #2 stale; the fresh-latest selector must
    # then be disabled.
    upload(client, samples=SAMPLES_CSV_1)
    html = client.get("/").text
    assert "已过期，请先重新计算" in html
    assert 'name="old_report_id"' in html
    # Recompute to get a fresh latest.
    assert compute(client).status_code == 303
    html = client.get("/").text
    assert 'value="3"' in html
    assert "生成并保存影响清单" in html


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_generate_impact_list_end_to_end(client: TestClient):
    make_report_chain(client)
    response = generate(client, 1, 2)
    assert response.status_code == 303
    location = response.headers["location"]
    assert "impact-1" in location

    store: Store = client.app_state.store
    summaries = store.list_impact_lists()
    assert len(summaries) == 1
    summary = summaries[0].summary
    assert summary["added"] == 1       # S3
    assert summary["removed"] == 1     # S2
    assert summary["shared"] == 1      # S1

    detail = client.get("/impacts/1")
    assert detail.status_code == 200
    body = detail.text
    for sample_id in ("S1", "S2", "S3"):
        assert sample_id in body
    assert "新增" in body and "移除" in body
    assert "成品单侧改动" in body

    # The saved list appears on the index page table.
    home = client.get("/").text
    assert "已保存的影响清单" in home
    assert "报告 #1 → #2" in home


def test_impact_csv_export_contains_statuses_and_attribution(
    client: TestClient,
):
    make_report_chain(client)
    assert generate(client, 1, 2).status_code == 303
    response = client.get("/impacts/1/export.csv")
    assert response.status_code == 200
    body = response.content.decode("utf-8-sig")
    lines = body.strip().splitlines()
    header = lines[0].split(",")
    assert header[:4] == ["sample_id", "status", "attribution", "changed"]
    rows = {line.split(",", 4)[0]: line for line in lines[1:]}
    assert "新增" in rows["S3"] and "移除" in rows["S2"]
    # Weight columns are percent formatted for point values.
    assert any(column.startswith("weight_A_") for column in header)


def test_unknown_impact_list_returns_404(client: TestClient):
    make_report_chain(client)
    assert client.get("/impacts/1234").status_code == 404
    assert client.get("/impacts/1234/export.csv").status_code == 404


# ---------------------------------------------------------------------------
# Validation and refusal rules
# ---------------------------------------------------------------------------


def test_same_report_pair_is_rejected(client: TestClient):
    make_report_chain(client)
    response = generate(client, 2, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "同一份报告" in response.text
    assert client.app_state.store.list_impact_lists() == []


def test_missing_report_ids_are_rejected(client: TestClient):
    make_report_chain(client)
    response = generate(client, 99, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "不存在" in response.text


def test_stale_new_side_is_rejected(client: TestClient):
    make_report_chain(client)
    # Make report #2 stale without recomputing.
    upload(client, samples=SAMPLES_CSV_1)
    response = generate(client, 1, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "已过期" in response.text
    assert client.app_state.store.list_impact_lists() == []


def test_non_latest_new_side_is_rejected(client: TestClient):
    make_report_chain(client)
    # Recompute on the same current inputs: report #3 becomes the latest,
    # while report #2 is still non-stale (its versions remain current).
    assert compute(client).status_code == 303
    response = generate(client, 1, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "已不是最新报告" in response.text


def test_ore_name_mismatch_refuses_comparison(client: TestClient):
    make_report_chain(client)
    # Report #3 with an ore renamed.
    renamed = NON_UNIQUE_BASE  # placeholder, build real rename below
    renamed = """ore,Fe,SiO2,Al2O3,CaO
A,62.3,3.9,2.1,1.2
B,68.5,2.5,1.1,0.8
D,55.2,6.2,3.4,2.1
"""
    upload(client, baseline=renamed)
    assert compute(client).status_code == 303
    response = generate(client, 1, 3, follow_redirects=True)
    assert response.status_code == 400
    assert "名称集合不一致" in response.text
    assert client.app_state.store.find_impact_list(1, 3) is None


def test_both_side_change_runs_crosses_and_labels_joint(
    client: TestClient,
):
    upload(client, BASE_CSV, SAMPLES_CSV_1)
    assert compute(client).status_code == 303
    upload(client, BASE_CSV_2, SAMPLES_CSV_2)
    assert compute(client).status_code == 303
    assert generate(client, 1, 2).status_code == 303

    record = client.app_state.store.get_impact_list(1)
    payload = record.payload
    assert payload["both_changed"] is True
    assert set(payload["crosses"]) == {"b0s1", "b1s0"}
    detail = client.get("/impacts/1").text
    assert "交叉组合" in detail


def test_non_unique_pair_renders_ranges_and_unattributable(
    client: TestClient,
):
    ambiguous_samples_1 = """sample_id,Fe,SiO2,Al2O3,CaO
AMBIG,55,5,2.5,1.5
"""
    ambiguous_samples_2 = """sample_id,Fe,SiO2,Al2O3,CaO
AMBIG,56,5,2.5,1.5
"""
    upload(client, NON_UNIQUE_BASE, ambiguous_samples_1)
    assert compute(client).status_code == 303
    upload(client, samples=ambiguous_samples_2)
    assert compute(client).status_code == 303
    assert generate(client, 1, 2).status_code == 303

    record = client.app_state.store.get_impact_list(1)
    entry = record.payload["entries"][0]
    assert all(descriptor["kind"] == "range" for descriptor in entry["weights"])
    assert all(
        descriptor["attribution"] in ("none", "samples_only")
        for descriptor in entry["weights"]
    )
    html = client.get("/impacts/1").text
    assert "无法归因" in html or "成品单侧改动" in html
    csv_text = client.get("/impacts/1/export.csv").content.decode("utf-8-sig")
    # Ranges use " ~ " separators and never present a single chosen weight.
    assert "~" in csv_text


# ---------------------------------------------------------------------------
# Interleaving and persistence guarantees
# ---------------------------------------------------------------------------


def test_latest_report_changing_during_generation_blocks_publication(
    client: TestClient, monkeypatch
):
    import app.main as main_module

    make_report_chain(client)
    store: Store = client.app_state.store
    real_build = main_module.build_impact_list

    def build_then_publish(*args, **kwargs):
        payload = real_build(*args, **kwargs)
        # Another report lands before the impact list is saved.
        upload(client, samples=SAMPLES_CSV_1)
        compute(client)
        return payload

    monkeypatch.setattr(main_module, "build_impact_list", build_then_publish)
    response = generate(client, 1, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "发生了变化" in response.text
    assert store.list_impact_lists() == []


def test_input_changing_during_generation_blocks_publication(
    client: TestClient, monkeypatch
):
    import app.main as main_module

    make_report_chain(client)
    store: Store = client.app_state.store
    real_build = main_module.build_impact_list

    def build_then_upload(*args, **kwargs):
        payload = real_build(*args, **kwargs)
        upload(client, samples=SAMPLES_CSV_1)  # report #2 becomes stale
        return payload

    monkeypatch.setattr(main_module, "build_impact_list", build_then_upload)
    response = generate(client, 1, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "发生了变化" in response.text
    assert store.list_impact_lists() == []


def test_write_failure_keeps_previously_saved_list(
    client: TestClient, monkeypatch
):
    make_report_chain(client)
    store: Store = client.app_state.store
    assert generate(client, 1, 2).status_code == 303
    before = store.list_impact_lists()
    assert len(before) == 1

    def broken_save(*args, **kwargs):
        raise StoreError("磁盘只读（测试注入）")

    monkeypatch.setattr(store, "save_impact_list", broken_save)
    response = generate(client, 1, 2, follow_redirects=True)
    assert response.status_code == 400
    assert "写入失败" in response.text
    assert "保持不变" in response.text
    after = store.list_impact_lists()
    assert len(after) == 1
    assert after[0].impact_id == before[0].impact_id
    # The saved list remains fully viewable.
    assert client.get("/impacts/1").status_code == 200


def test_saved_list_survives_later_input_and_report_changes(
    client: TestClient,
):
    make_report_chain(client)
    assert generate(client, 1, 2).status_code == 303

    # Keep evolving the inputs; new reports come and go.
    upload(client, BASE_CSV_2, SAMPLES_CSV_1)
    assert compute(client).status_code == 303
    upload(client, samples=SAMPLES_CSV_2)
    assert compute(client).status_code == 303

    # The old saved list is still readable by its report pair and id.
    record = client.app_state.store.find_impact_list(1, 2)
    assert record is not None
    assert record.payload["old_report_id"] == 1
    assert record.payload["new_report_id"] == 2
    assert client.get(f"/impacts/{record.impact_id}").status_code == 200


def test_regenerating_same_pair_replaces_instead_of_duplicating(
    client: TestClient,
):
    make_report_chain(client)
    assert generate(client, 1, 2).status_code == 303
    assert generate(client, 1, 2).status_code == 303
    store: Store = client.app_state.store
    assert len(store.list_impact_lists()) == 1
    assert store.find_impact_list(1, 2) is not None
    assert store.find_impact_list(2, 1) is None


def test_original_compute_workflow_still_works(client: TestClient):
    """The new feature must not regress upload/compute/export behavior."""

    upload(client, BASE_CSV, SAMPLES_CSV_1)
    assert compute(client).status_code == 303
    report = client.get("/api/latest").json()["report"]
    assert report["stale"] is False
    export = client.get(f"/reports/{report['id']}/export.csv")
    assert export.status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}


def test_concurrent_impact_saves_never_corrupt_state(tmp_path: Path):
    """Hammer impact-list saves against concurrent uploads/reports.

    Every saved list must reference an existing ordered report pair whose
    new side was current at save time; no database errors or duplicate
    rows per pair are permitted.
    """

    import sqlite3
    import threading
    import time

    store = Store(str(tmp_path / "impact-race.db"))
    b_payload = _baseline_payload(BASE_CSV)
    s1_payload = _samples_payload(SAMPLES_CSV_1)
    s2_payload = _samples_payload(SAMPLES_CSV_2)
    store.save_inputs(
        {"baseline": (b_payload, "b.csv"), "samples": (s1_payload, "s1.csv")}
    )

    from app.compute import compute_report
    from app.diff import build_impact_list
    from app.inputs import BaselineInput, SamplesInput

    def publish(samples_payload):
        snapshot = store.snapshot()
        payload = compute_report(
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
        )
        return store.publish_report(snapshot, payload)

    report_ids = [publish(s1_payload)]
    store.save_inputs({"samples": (s2_payload, "s2.csv")})
    report_ids.append(publish(s2_payload))

    errors: list[BaseException] = []
    saved: list[int] = []

    def saver():
        try:
            for _ in range(40):
                old = store.get_report(report_ids[0])
                new = store.get_report(report_ids[1])
                impact = build_impact_list(
                    report_ids[0],
                    report_ids[1],
                    old.payload,
                    new.payload,
                    baseline_changed=False,
                    samples_changed=True,
                )
                result = store.save_impact_list(
                    report_ids[0], report_ids[1], impact
                )
                if result[0] is not None:
                    saved.append(result[0])
                time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    def uploader():
        try:
            time.sleep(0.01)
            for i in range(20):
                payload = s2_payload if i % 2 else s1_payload
                store.save_inputs({"samples": (dict(payload), f"s{i}.csv")})
                time.sleep(0.002)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=saver), threading.Thread(target=uploader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    # At least one save must have landed in a quiet window; the unique pair
    # index means all of them collapse onto a single row.
    assert saved
    with sqlite3.connect(tmp_path / "impact-race.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM impact_lists").fetchone()[0]
        pair_count = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM impact_lists "
            "GROUP BY old_report_id, new_report_id)"
        ).fetchone()[0]
    assert count == 1
    assert pair_count == 1


def _baseline_payload(csv_text: str) -> dict:
    from app.inputs import parse_baseline

    return parse_baseline(csv_text.encode()).to_payload()


def _samples_payload(csv_text: str) -> dict:
    from app.inputs import parse_samples

    return parse_samples(csv_text.encode()).to_payload()
