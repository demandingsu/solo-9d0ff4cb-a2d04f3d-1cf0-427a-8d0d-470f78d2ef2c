"""FastAPI application for ore-blend proportion reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from . import formatting
from .compute import ComputeError, compute_report
from .db import (
    ImpactListRecord,
    ReportRecord,
    Store,
    StoreError,
    VersionedInput,
)
from .diff import ATTRIBUTION_LABELS, DiffError, build_impact_list
from .inputs import (
    MAX_UPLOAD_BYTES,
    BaselineInput,
    InputValidationError,
    SamplesInput,
    parse_baseline,
    parse_samples,
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_ROW_LIMIT = 500
IMPACT_ROW_LIMIT = 500


def create_app(database_path: str | None = None) -> FastAPI:
    app = FastAPI(title="矿石配料反推与异常定位")
    store = Store(database_path or os.environ.get("ORE_DB_PATH", "data/app.db"))
    app.state.store = store

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["decimal"] = formatting.decimal
    templates.env.filters["percent"] = formatting.percent
    templates.env.filters["signed"] = formatting.signed
    templates.env.filters["ratio"] = formatting.ratio_text
    templates.env.filters["exact"] = formatting.exact

    app.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    # -- helpers ---------------------------------------------------------

    def redirect_notice(
        message: str, kind: str = "ok", anchor: str = ""
    ) -> RedirectResponse:
        query = urlencode({"notice": message, "notice_kind": kind})
        url = f"/?{query}"
        if anchor:
            url += f"#{anchor}"
        return RedirectResponse(url, status_code=303)

    def payload_to_inputs(
        snapshot,
    ) -> tuple[BaselineInput, SamplesInput]:
        return (
            BaselineInput.from_payload(snapshot.baseline.payload),
            SamplesInput.from_payload(snapshot.samples.payload),
        )

    def input_summary(value: VersionedInput | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if value.kind == "baseline":
            parsed = BaselineInput.from_payload(value.payload)
            return {
                "version": value.version_id,
                "filename": value.filename,
                "created_at": value.created_at,
                "elements": list(parsed.elements),
                "names": list(parsed.ores),
                "rows": [
                    [formatting.decimal(v, 8) for v in row] for row in parsed.rows
                ],
                "count": len(parsed.rows),
            }
        parsed_samples = SamplesInput.from_payload(value.payload)
        return {
            "version": value.version_id,
            "filename": value.filename,
            "created_at": value.created_at,
            "elements": list(parsed_samples.elements),
            "names": list(parsed_samples.sample_ids[:5]),
            "rows": [],
            "count": len(parsed_samples.sample_ids),
        }

    def report_view(record: ReportRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        payload = dict(record.payload)
        rows = payload.get("results", [])
        visible = rows[:REPORT_ROW_LIMIT]
        return {
            "id": record.report_id,
            "created_at": record.created_at,
            "baseline_version": record.baseline_version,
            "samples_version": record.samples_version,
            "stale": record.is_stale,
            "baseline_current": record.baseline_is_current,
            "samples_current": record.samples_is_current,
            "payload": payload,
            "rows": visible,
            "hidden_rows": max(0, len(rows) - len(visible)),
        }

    def impact_view(record: ImpactListRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        entries = record.payload.get("entries", [])
        visible = entries[:IMPACT_ROW_LIMIT]
        return {
            "id": record.impact_id,
            "created_at": record.created_at,
            "old_report_id": record.old_report_id,
            "new_report_id": record.new_report_id,
            "payload": record.payload,
            "entries": visible,
            "hidden_rows": max(0, len(entries) - len(visible)),
            "attribution_labels": ATTRIBUTION_LABELS,
        }

    def context(request: Request) -> dict[str, Any]:
        current = store.current_inputs()
        record = store.latest_report()
        all_reports = store.list_reports()
        fresh_reports = [item for item in all_reports if not item.is_stale]
        saved_impacts = store.list_impact_lists()
        return {
            "request": request,
            "current": {
                kind: input_summary(value) for kind, value in current.items()
            },
            "report": report_view(record),
            "all_reports": [report_view(item) for item in all_reports],
            "fresh_report_ids": {item.report_id for item in fresh_reports},
            "saved_impacts": [
                {
                    "id": item.impact_id,
                    "old_report_id": item.old_report_id,
                    "new_report_id": item.new_report_id,
                    "created_at": item.created_at,
                    "summary": item.summary,
                }
                for item in saved_impacts
            ],
            "notice": request.query_params.get("notice"),
            "notice_kind": request.query_params.get("notice_kind", "ok"),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(request, "index.html", context(request))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/inputs/upload")
    async def upload_inputs(
        request: Request,
        baseline_file: UploadFile | None = File(None),
        samples_file: UploadFile | None = File(None),
    ) -> Response:
        provided = {
            "baseline": baseline_file,
            "samples": samples_file,
        }
        if not any(file is not None and file.filename for file in provided.values()):
            return redirect_notice("请至少选择一个 CSV 文件再上传。", "error", "upload")

        parsed: dict[str, Any] = {}
        filenames: dict[str, str] = {}
        labels = {"baseline": "基准", "samples": "成品"}
        try:
            for kind, file in provided.items():
                if file is None or not file.filename:
                    continue
                raw = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise _LabeledError(
                        labels[kind],
                        f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                parser = parse_baseline if kind == "baseline" else parse_samples
                try:
                    parsed[kind] = await run_in_threadpool(parser, raw)
                except InputValidationError as exc:
                    raise _LabeledError(labels[kind], str(exc)) from exc
                filenames[kind] = (
                    file.filename.replace("\\", "/").rsplit("/", 1)[-1][:200]
                    or "未命名.csv"
                )
        except _LabeledError as exc:
            return _render_with_error(request, f"{exc.label}文件：{exc.message}")

        updates: dict[str, tuple[dict[str, Any], str]] = {}
        for kind, value in parsed.items():
            updates[kind] = (value.to_payload(), filenames[kind])

        had_report = await run_in_threadpool(
            lambda: store.latest_report() is not None
        )
        try:
            versions = await run_in_threadpool(store.save_inputs, updates)
        except StoreError as exc:
            return _render_with_error(
                request, f"输入写入失败，原有输入未被替换：{exc}"
            )

        replaced = []
        if "baseline" in versions:
            replaced.append(f"基准 v{versions['baseline']}")
        if "samples" in versions:
            replaced.append(f"成品 v{versions['samples']}")
        suffix = "。旧报告已标记为过期。" if had_report else "。"
        return redirect_notice(
            "已整版替换：" + "，".join(replaced) + suffix,
            "ok",
            "report",
        )

    def _render_with_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    def _render_impact_error(request: Request, message: str) -> HTMLResponse:
        ctx = context(request)
        ctx["notice"] = message
        ctx["notice_kind"] = "error"
        return templates.TemplateResponse(
            request, "index.html", ctx, status_code=400
        )

    @app.post("/compute")
    async def compute(request: Request) -> Response:
        snapshot = await run_in_threadpool(store.snapshot)
        if snapshot is None:
            return redirect_notice(
                "基准与成品输入都必须先上传，才能计算。", "error", "upload"
            )

        try:
            baseline, samples = payload_to_inputs(snapshot)
            payload = await run_in_threadpool(
                compute_report, baseline, samples
            )
        except (ComputeError, InputValidationError) as exc:
            return redirect_notice(str(exc), "error", "report")
        except Exception as exc:  # pragma: no cover - defensive
            return redirect_notice(
                f"计算失败，未产生新报告：{type(exc).__name__}: {exc}",
                "error",
                "report",
            )

        try:
            report_id = await run_in_threadpool(
                store.publish_report, snapshot, payload
            )
        except StoreError as exc:
            return redirect_notice(
                f"报告写入失败，仍显示上一份报告：{exc}", "error", "report"
            )

        if report_id is None:
            return redirect_notice(
                "计算期间输入已被更新，本次结果与当前输入不符，已丢弃，"
                "未发布报告。请基于新输入重新计算。",
                "error",
                "report",
            )

        return redirect_notice(
            f"报告 #{report_id} 已生成并发布（{payload['sample_count']} 个样本，"
            f"{payload['non_unique_count']} 个无法唯一确定）。",
            "ok",
            "report",
        )

    # -- impact lists ----------------------------------------------------

    @app.post("/impacts/generate")
    async def generate_impact(request: Request) -> Response:
        form = await request.form()
        try:
            old_id = int(str(form.get("old_report_id", "")).strip())
            new_id = int(str(form.get("new_report_id", "")).strip())
        except (TypeError, ValueError):
            return _render_impact_error(request, "请选择旧报告与新报告后再生成。")

        try:
            old_record, new_record = await run_in_threadpool(
                _load_report_pair, store, old_id, new_id
            )
        except DiffError as exc:
            return _render_impact_error(request, str(exc))

        if new_record.is_stale:
            return _render_impact_error(
                request,
                "新报告已过期（其基准或成品输入已被替换），不能作为对照的新侧；"
                "请先基于当前输入计算最新报告。",
            )
        latest = await run_in_threadpool(store.latest_report)
        if latest is None or latest.report_id != new_id:
            return _render_impact_error(
                request,
                f"报告 #{new_id} 已不是最新报告；新侧必须是当前最新、未过期的报告，"
                "请刷新页面后重新选择。",
            )

        baseline_changed = (
            new_record.baseline_version != old_record.baseline_version
        )
        samples_changed = (
            new_record.samples_version != old_record.samples_version
        )
        if not baseline_changed and not samples_changed:
            return _render_impact_error(
                request, "两份报告来自完全相同的输入版本，没有可对照的变化。"
            )

        old_baseline = old_samples = new_baseline = new_samples = None
        if baseline_changed and samples_changed:
            try:
                snapshots = await run_in_threadpool(
                    _load_four_inputs,
                    store,
                    old_record,
                    new_record,
                )
            except DiffError as exc:
                return _render_impact_error(request, str(exc))
            old_baseline, old_samples, new_baseline, new_samples = snapshots

        try:
            payload = await run_in_threadpool(
                build_impact_list,
                old_id,
                new_id,
                old_record.payload,
                new_record.payload,
                baseline_changed=baseline_changed,
                samples_changed=samples_changed,
                old_baseline=old_baseline,
                old_samples=old_samples,
                new_baseline=new_baseline,
                new_samples=new_samples,
            )
        except DiffError as exc:
            return _render_impact_error(request, str(exc))

        try:
            saved = await run_in_threadpool(
                store.save_impact_list, old_id, new_id, payload
            )
        except StoreError as exc:
            return _render_impact_error(
                request, f"影响清单写入失败，已保存的清单保持不变：{exc}"
            )

        impact_id, replaced = saved
        if impact_id is None:
            return _render_impact_error(
                request,
                "生成期间当前输入或最新报告发生了变化，本次清单针对的新侧已失效，"
                "已丢弃且未发布。请基于最新报告重新生成。",
            )

        verb = "已更新" if replaced else "已生成并保存"
        return redirect_notice(
            f"样本级影响清单 #{impact_id} {verb}（报告 #{old_id} → #{new_id}，"
            f"新增 {payload['summary']['added']}、移除 "
            f"{payload['summary']['removed']}、变化 "
            f"{payload['summary']['changed']}）。",
            "ok",
            f"impact-{impact_id}",
        )

    @app.get("/impacts/{impact_id}", response_class=HTMLResponse)
    def impact_detail(request: Request, impact_id: int) -> Response:
        record = store.get_impact_list(impact_id)
        if record is None:
            return PlainTextResponse("影响清单不存在", status_code=404)
        ctx = context(request)
        ctx["impact"] = impact_view(record)
        return templates.TemplateResponse(request, "impact.html", ctx)

    @app.get("/impacts/{impact_id}/export.csv")
    def impact_export(impact_id: int) -> Response:
        record = store.get_impact_list(impact_id)
        if record is None:
            return PlainTextResponse("影响清单不存在", status_code=404)
        return Response(
            "﻿" + build_impact_csv(record) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-impact-{impact_id}.csv"'
                )
            },
        )

    # -- exports ---------------------------------------------------------

    @app.get("/reports/{report_id}/export.csv")
    def export_report(report_id: int) -> Response:
        record = store.get_report(report_id)
        if record is None:
            return PlainTextResponse("报告不存在", status_code=404)
        payload = record.payload
        elements = payload["elements"]
        header = ["sample_id", "unique"]
        for ore in payload["ores"]:
            header.append(f"{ore}_weight")
        for ore in payload["ores"]:
            header.append(f"{ore}_weight_min")
            header.append(f"{ore}_weight_max")
        for element in elements:
            header.append(f"measured_{element}")
        for element in elements:
            header += [
                f"predicted_{element}",
                f"predicted_{element}_min",
                f"predicted_{element}_max",
            ]
        for element in elements:
            header += [
                f"residual_{element}",
                f"residual_{element}_min",
                f"residual_{element}_max",
            ]
        header += ["sse", "note"]

        def triple(point_value, interval):
            """One element's (value, min, max) cells.

            A non-unique optimum has no single value, so the value cell is
            left empty and only the exact attainable range is emitted.
            """

            low, high = interval
            return [
                "" if point_value is None else formatting.decimal(point_value, 10),
                formatting.decimal(low, 10),
                formatting.decimal(high, 10),
            ]

        rows = [",".join(_csv_cell(cell) for cell in header)]
        for result in payload["results"]:
            row = [result["sample_id"], "yes" if result["unique"] else "no"]
            if result["unique"]:
                row += [formatting.percent(v, 8) for v in result["weights"]]
            else:
                row += [""] * len(payload["ores"])
            for low, high in result["weight_ranges"]:
                row += [formatting.percent(low, 8), formatting.percent(high, 8)]
            row += [formatting.decimal(v, 10) for v in result["measured"]]
            for i in range(len(elements)):
                row += triple(
                    None if result["predicted"] is None else result["predicted"][i],
                    result["predicted_ranges"][i],
                )
            for i in range(len(elements)):
                row += triple(
                    None if result["residuals"] is None else result["residuals"][i],
                    result["residual_ranges"][i],
                )
            row.append(formatting.decimal(result["sse"], 12))
            row.append(
                ""
                if result["unique"]
                else "最优占比不唯一：占比、预测值与残差仅以下列 min/max 范围给出，"
                "不存在唯一的配料组合"
            )
            rows.append(",".join(_csv_cell(cell) for cell in row))
        return Response(
            "﻿" + "\r\n".join(rows) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ore-report-{report_id}.csv"'
                )
            },
        )

    @app.get("/inputs/{kind}/export.csv")
    def export_input(kind: str) -> Response:
        if kind not in ("baseline", "samples"):
            return PlainTextResponse("输入类型不存在", status_code=404)
        current = store.current_inputs().get(kind)
        if current is None:
            return PlainTextResponse("输入尚未上传", status_code=404)
        if kind == "baseline":
            value = BaselineInput.from_payload(current.payload)
            header = ["ore", *value.elements]
            rows = [
                [ore, *[formatting.decimal(x, 12) for x in row]]
                for ore, row in zip(value.ores, value.rows)
            ]
        else:
            value = SamplesInput.from_payload(current.payload)
            header = ["sample_id", *value.elements]
            rows = [
                [sample_id, *[formatting.decimal(x, 12) for x in row]]
                for sample_id, row in zip(value.sample_ids, value.rows)
            ]
        lines = [",".join(_csv_cell(cell) for cell in header)]
        lines += [",".join(_csv_cell(cell) for cell in row) for row in rows]
        return Response(
            "﻿" + "\r\n".join(lines) + "\r\n",
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{kind}.csv"'
            },
        )

    @app.get("/api/latest")
    def api_latest() -> JSONResponse:
        record = store.latest_report()
        current = store.current_inputs()
        if record is None:
            return JSONResponse(
                {
                    "report": None,
                    "inputs": {
                        kind: (
                            {
                                "version": value.version_id,
                                "filename": value.filename,
                                "created_at": value.created_at,
                            }
                            if value
                            else None
                        )
                        for kind, value in current.items()
                    },
                }
            )
        return JSONResponse(
            {
                "report": {
                    "id": record.report_id,
                    "created_at": record.created_at,
                    "baseline_version": record.baseline_version,
                    "samples_version": record.samples_version,
                    "stale": record.is_stale,
                    "payload": record.payload,
                }
            }
        )

    return app


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\r", "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _load_report_pair(
    store: Store, old_id: int, new_id: int
) -> tuple[ReportRecord, ReportRecord]:
    old_record = store.get_report(old_id)
    new_record = store.get_report(new_id)
    if old_record is None or new_record is None:
        raise DiffError("所选报告不存在，请刷新页面后重新选择。")
    if old_id == new_id:
        raise DiffError("旧报告与新报告不能选择同一份报告。")
    return old_record, new_record


def _load_four_inputs(
    store: Store, old_record: ReportRecord, new_record: ReportRecord
) -> tuple[BaselineInput, SamplesInput, BaselineInput, SamplesInput]:
    """Fetch every input snapshot needed for the two cross combinations."""

    versions = {
        "old_baseline": old_record.baseline_version,
        "old_samples": old_record.samples_version,
        "new_baseline": new_record.baseline_version,
        "new_samples": new_record.samples_version,
    }
    rows = {name: store.get_input_version(version) for name, version in versions.items()}
    missing = [name for name, row in rows.items() if row is None]
    if missing:
        raise DiffError("报告引用的输入版本已不存在，无法重算交叉组合。")
    return (
        BaselineInput.from_payload(rows["old_baseline"].payload),
        SamplesInput.from_payload(rows["old_samples"].payload),
        BaselineInput.from_payload(rows["new_baseline"].payload),
        SamplesInput.from_payload(rows["new_samples"].payload),
    )


_STATUS_LABELS = {
    "added": "新增",
    "removed": "移除",
    "common": "共有",
}


def _descriptor_cells(descriptor: dict[str, Any]) -> list[str]:
    """Render one change descriptor as CSV cells.

    Non-unique optima are emitted as ranges only – never as a chosen
    proportion – on both the old and new side.
    """

    attribution = ATTRIBUTION_LABELS[descriptor["attribution"]]
    if descriptor["kind"] == "range":
        old_range = descriptor["old_range"]
        new_range = descriptor["new_range"]
        old_text = (
            ""
            if old_range is None
            else f"{formatting.decimal(old_range[0], 10)} ~ "
            f"{formatting.decimal(old_range[1], 10)}"
        )
        new_text = (
            ""
            if new_range is None
            else f"{formatting.decimal(new_range[0], 10)} ~ "
            f"{formatting.decimal(new_range[1], 10)}"
        )
        return [old_text, new_text, "", attribution]

    old_value = descriptor["old"]
    new_value = descriptor["new"]
    delta = descriptor["delta"]
    return [
        "" if old_value is None else formatting.decimal(old_value, 10),
        "" if new_value is None else formatting.decimal(new_value, 10),
        "" if delta is None else formatting.decimal(delta, 10),
        attribution,
    ]


def build_impact_csv(record: ImpactListRecord) -> str:
    payload = record.payload
    elements = payload["elements"]
    ores = payload["ores"]

    header = [
        "sample_id",
        "status",
        "attribution",
        "changed",
    ]
    for ore in ores:
        header += [
            f"weight_{ore}_old",
            f"weight_{ore}_new",
            f"weight_{ore}_delta",
            f"weight_{ore}_attribution",
        ]
    for element in elements:
        header += [
            f"measured_{element}_old",
            f"measured_{element}_new",
            f"measured_{element}_delta",
            f"measured_{element}_attribution",
        ]
    for element in elements:
        header += [
            f"predicted_{element}_old",
            f"predicted_{element}_new",
            f"predicted_{element}_delta",
            f"predicted_{element}_attribution",
        ]
    for element in elements:
        header += [
            f"residual_{element}_old",
            f"residual_{element}_new",
            f"residual_{element}_delta",
            f"residual_{element}_attribution",
        ]
    header += ["sse_old", "sse_new", "sse_delta", "sse_attribution"]

    lines = [",".join(_csv_cell(cell) for cell in header)]
    for entry in payload["entries"]:
        row = [
            entry["sample_id"],
            _STATUS_LABELS[entry["status"]],
            ATTRIBUTION_LABELS[entry["attribution"]],
            "yes" if entry["has_changes"] else "no",
        ]
        for descriptor in entry["weights"]:
            cells = _descriptor_cells(descriptor)
            if descriptor["kind"] == "point" and cells[0]:
                cells[0] = formatting.percent(descriptor["old"], 8)
            if descriptor["kind"] == "point" and cells[1]:
                cells[1] = formatting.percent(descriptor["new"], 8)
            if descriptor["kind"] == "point" and cells[2]:
                cells[2] = formatting.percent(descriptor["delta"], 8)
            row += cells
        for descriptor in entry["measured"]:
            row += _descriptor_cells(descriptor)
        for descriptor in entry["predicted"]:
            row += _descriptor_cells(descriptor)
        for descriptor in entry["residuals"]:
            row += _descriptor_cells(descriptor)
        row += _descriptor_cells(entry["sse"])
        lines.append(",".join(_csv_cell(cell) for cell in row))
    return "\r\n".join(lines)


class _LabeledError(Exception):
    def __init__(self, label: str, message: str) -> None:
        super().__init__(message)
        self.label = label
        self.message = message


app = create_app()
