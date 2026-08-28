"""导出全局事件素材，并把可用事件和身份登记到产品数据库。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.catalog import CatalogService
from src.modules.database import ParticipantRecord, ProductDatabase
from src.modules.identity import CrossMaterialIdentityAssociator
from src.modules.materials import MaterialExporter


class MaterialFinalizationApplication:
    """编排素材导出、身份归并、目录构建和数据库登记。"""

    def __init__(
        self,
        *,
        exporter: MaterialExporter,
        catalog: CatalogService,
        database: ProductDatabase,
        associator: CrossMaterialIdentityAssociator | None = None,
    ) -> None:
        """保存已有模块；应用层不实现模型、FFmpeg 或 SQL 细节。"""
        self.exporter = exporter
        self.catalog = catalog
        self.database = database
        self.associator = associator or CrossMaterialIdentityAssociator()

    def run(
        self,
        *,
        source_video_id: str,
        source_video_path: str | Path,
        timeline_report: Mapping[str, Any],
        output_directory: str | Path,
        identity_reports: Mapping[str, Mapping[str, Any]] | None = None,
        replace_database_records: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """完成最终素材阶段；身份报告缺失时仍登记事件素材。"""
        raw_drafts = timeline_report.get("material_drafts", [])
        raw_events = timeline_report.get("events", [])
        if not isinstance(raw_drafts, list) or not isinstance(raw_events, list):
            raise ValueError("时间线报告必须包含 material_drafts 和 events 列表")

        exported = self.exporter.export(
            source_video=source_video_path,
            material_drafts=raw_drafts,
            output_directory=output_directory,
            dry_run=dry_run,
        )
        event_by_id = {
            str(value["event_id"]): value
            for value in raw_events
            if isinstance(value, Mapping) and value.get("event_id") is not None
        }
        association = self.associator.associate(
            source_video_id=source_video_id,
            identity_reports=identity_reports or {},
        )
        participants_by_material = association["material_participants"]

        registered_ids: list[str] = []
        for material in exported:
            related_events = [
                event_by_id[event_id]
                for event_id in material.event_ids
                if event_id in event_by_id
            ]
            references = participants_by_material.get(material.material_id, [])
            item = self.catalog.build_final_material(
                material_id=material.material_id,
                source_video_id=source_video_id,
                video_path=material.video_path,
                start_seconds=material.start_seconds,
                end_seconds=material.end_seconds,
                events=related_events,
                participants=references,
                metadata={
                    "event_ids": list(material.event_ids),
                    "boundary_source": "event_timeline",
                    "identity_available": bool(references),
                },
            )
            if dry_run:
                continue
            for reference in item.participants:
                self.database.save_participant(
                    ParticipantRecord(
                        participant_id=reference.participant_id,
                        display_name=reference.player_name,
                        jersey_color=reference.jersey_color,
                        jersey_number=reference.jersey_number,
                        metadata={"identity_status": reference.identity_status},
                    ),
                    replace=True,
                )
            self.database.save_material(item, replace=replace_database_records)
            registered_ids.append(item.material_id)

        return {
            "schema_version": "basketevent_material_finalization.v1",
            "source_video_id": source_video_id,
            "dry_run": dry_run,
            "exported_materials": [value.to_dict() for value in exported],
            "identity_association": association,
            "registered_material_ids": registered_ids,
        }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """读取一个 JSON 对象，并在格式错误时给出明确名称。"""
    with path.expanduser().open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label}根节点必须是对象：{path}")
    return value


def _read_identity_index(path: Path | None) -> dict[str, dict[str, Any]]:
    """读取“素材编号到身份报告路径”的可选索引文件。"""
    if path is None:
        return {}
    index_path = path.expanduser()
    index = _read_json_object(index_path, "身份索引")
    reports: dict[str, dict[str, Any]] = {}
    for material_id, report_value in index.items():
        report_path = Path(str(report_value)).expanduser()
        if not report_path.is_absolute():
            report_path = index_path.parent / report_path
        reports[str(material_id)] = _read_json_object(report_path, "身份报告")
    return reports


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析最终素材处理命令。"""
    parser = argparse.ArgumentParser(
        description="导出事件素材，并把事件和可选身份登记到 SQLite。"
    )
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--timeline-json", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument(
        "--identity-index-json",
        type=Path,
        default=None,
        help="可选 JSON：键为 material_id，值为对应身份报告路径。",
    )
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-database-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """从 JSON 文件构造最小模块并执行最终素材处理。"""
    args = parse_args(argv)
    timeline = _read_json_object(args.timeline_json, "时间线报告")
    identity_reports = _read_identity_index(args.identity_index_json)
    application = MaterialFinalizationApplication(
        exporter=MaterialExporter(args.ffmpeg_binary, args.overwrite),
        catalog=CatalogService(),
        database=ProductDatabase.open(args.database_root),
    )
    report = application.run(
        source_video_id=args.source_video_id,
        source_video_path=args.source_video,
        timeline_report=timeline,
        output_directory=args.output_directory,
        identity_reports=identity_reports,
        replace_database_records=args.replace_database_records,
        dry_run=args.dry_run,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
