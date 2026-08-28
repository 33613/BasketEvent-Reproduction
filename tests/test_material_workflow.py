"""测试事件素材导出、身份归并和数据库登记链路。"""

import tempfile
import unittest
from pathlib import Path

from src.application.finalize_materials import MaterialFinalizationApplication
from src.modules.catalog import CatalogService
from src.modules.database import ProductDatabase
from src.modules.identity import CrossMaterialIdentityAssociator
from src.modules.materials import ExportedMaterial, MaterialExporter


class FakeExporter:
    """为应用测试生成无需 FFmpeg 的确定性素材。"""

    def export(self, *, source_video, material_drafts, output_directory, dry_run=False):
        """按草稿创建空占位视频并返回导出记录。"""
        output_root = Path(output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        values = []
        for index, draft in enumerate(material_drafts):
            path = output_root / f"{index:05d}.mp4"
            if not dry_run:
                path.write_bytes(b"video")
            values.append(
                ExportedMaterial(
                    material_id=str(draft["material_id"]),
                    video_path=path,
                    start_seconds=float(draft["start_seconds"]),
                    end_seconds=float(draft["end_seconds"]),
                    event_ids=tuple(draft["event_ids"]),
                )
            )
        return tuple(values)


class MaterialExporterTest(unittest.TestCase):
    """验证素材导出的时间和文件名规则。"""

    def test_dry_run_builds_portable_output_and_ffmpeg_command(self) -> None:
        """业务编号包含冒号时仍应生成可移植文件名。"""
        with tempfile.TemporaryDirectory() as directory:
            exporter = MaterialExporter(ffmpeg_binary="ffmpeg-test")
            values = exporter.export(
                source_video=Path(directory) / "source.mp4",
                material_drafts=[
                    {
                        "material_id": "video:material:00001",
                        "start_seconds": 3.0,
                        "end_seconds": 8.5,
                        "event_ids": ["event-1"],
                    }
                ],
                output_directory=Path(directory) / "materials",
                dry_run=True,
            )

            self.assertEqual(len(values), 1)
            self.assertNotIn(":", values[0].video_path.name)
            command = exporter.build_command("source.mp4", "output.mp4", 3.0, 8.5)
            self.assertEqual(command[0], "ffmpeg-test")
            self.assertIn("5.500000", command)


class CrossMaterialIdentityAssociatorTest(unittest.TestCase):
    """验证确定身份跨素材合并，冲突身份保持隔离。"""

    def test_only_identified_jersey_is_merged(self) -> None:
        """相同球衣身份应合并，匿名轨迹不应跨素材合并。"""
        reports = {
            "material-a": {
                "resolutions": [
                    {
                        "track_id": "player_1",
                        "status": "identified",
                        "participant_id": "game:jersey:white:13",
                        "jersey_color": "white",
                        "jersey_number": "13",
                    },
                    {"track_id": "player_2", "status": "anonymous"},
                ]
            },
            "material-b": {
                "resolutions": [
                    {
                        "track_id": "player_8",
                        "status": "identified",
                        "participant_id": "game:jersey:white:13",
                        "jersey_color": "white",
                        "jersey_number": "13",
                    },
                    {"track_id": "player_2", "status": "anonymous"},
                ]
            },
        }

        result = CrossMaterialIdentityAssociator().associate(
            source_video_id="game", identity_reports=reports
        )

        stable = next(
            item
            for item in result["participants"]
            if item["participant_id"] == "game:jersey:white:13"
        )
        self.assertEqual(stable["appearance_count"], 2)
        anonymous_ids = {
            value["participant_id"]
            for values in result["material_participants"].values()
            for value in values
            if value["identity_status"] == "anonymous"
        }
        self.assertEqual(len(anonymous_ids), 2)


class MaterialFinalizationApplicationTest(unittest.TestCase):
    """验证最终素材可以登记并通过事件和人物查询。"""

    def test_event_identity_and_database_form_one_workflow(self) -> None:
        """一个进球素材应被保存，并可按事件和身份找回。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = ProductDatabase.open(root / "product")
            application = MaterialFinalizationApplication(
                exporter=FakeExporter(),
                catalog=CatalogService(),
                database=database,
            )
            timeline = {
                "events": [
                    {
                        "event_id": "game:event:1",
                        "event": "Made Shot",
                        "confidence": 0.91,
                        "global_track_id": None,
                        "evidence_start_seconds": 10.0,
                        "evidence_end_seconds": 11.0,
                    }
                ],
                "material_drafts": [
                    {
                        "material_id": "game:material:1",
                        "start_seconds": 5.0,
                        "end_seconds": 13.0,
                        "event_ids": ["game:event:1"],
                    }
                ],
            }
            identity_reports = {
                "game:material:1": {
                    "resolutions": [
                        {
                            "track_id": "player_3",
                            "status": "identified",
                            "participant_id": "game:jersey:black:17",
                            "jersey_color": "black",
                            "jersey_number": "17",
                        }
                    ]
                }
            }

            result = application.run(
                source_video_id="game",
                source_video_path=root / "source.mp4",
                timeline_report=timeline,
                output_directory=root / "materials",
                identity_reports=identity_reports,
            )

            self.assertEqual(result["registered_material_ids"], ["game:material:1"])
            by_event = database.find_materials(event="Made Shot")
            by_person = database.find_materials(participant_id="game:jersey:black:17")
            self.assertEqual(len(by_event), 1)
            self.assertEqual(len(by_person), 1)
            self.assertEqual(by_event[0].video_path.name, "00000.mp4")


if __name__ == "__main__":
    unittest.main()
