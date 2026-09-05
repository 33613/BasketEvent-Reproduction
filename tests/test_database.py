"""验证产品 SQLite 人物库、素材库和目录初始化。"""

import tempfile
import unittest
import sqlite3
from pathlib import Path

from src.modules.catalog import (
    CatalogItem,
    CatalogService,
    EventTag,
    ParticipantReference,
)
from src.modules.database import ParticipantRecord, ProductDatabase


class ProductDatabaseTest(unittest.TestCase):
    """验证产品数据能够跨对象重新打开，并与 BARD 路径解耦。"""

    def test_initialize_creates_only_product_runtime_layout(self):
        """初始化应创建数据库和三类媒体目录。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"

            product = ProductDatabase.open(root)

            self.assertTrue(product.storage.database_path.is_file())
            self.assertTrue(product.storage.uploads_dir.is_dir())
            self.assertTrue(product.storage.segments_dir.is_dir())
            self.assertTrue(product.storage.visualizations_dir.is_dir())
            self.assertEqual(product.status()["schema_version"], 2)

    def test_participant_record_survives_reopen(self):
        """最小人物档案应在重新打开数据库后仍可按球衣属性查询。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"
            product = ProductDatabase.open(root)
            profile = ParticipantRecord(
                participant_id="person_test_20",
                display_name=None,
                jersey_color="white",
                jersey_number="20",
                metadata={"source": "test"},
            )
            product.save_participant(profile)

            reopened = ProductDatabase.open(root)
            attribute_matches = reopened.find_participants_by_jersey("white", "20")

            self.assertEqual(attribute_matches[0].participant_id, "person_test_20")

    def test_version_one_database_is_upgraded_without_losing_events(self):
        """已有长视频实验数据库应原地增加事件主体字段。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"
            database_path = root / "database" / "basketevent.sqlite3"
            database_path.parent.mkdir(parents=True)
            connection = sqlite3.connect(database_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE schema_versions(version INTEGER PRIMARY KEY);
                    INSERT INTO schema_versions(version) VALUES (1);
                    CREATE TABLE materials(
                        material_id TEXT PRIMARY KEY,
                        source_video_id TEXT NOT NULL,
                        segment_id TEXT NOT NULL,
                        video_path TEXT NOT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL,
                        processing_status TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source_video_id, segment_id)
                    );
                    CREATE TABLE material_events(
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        material_id TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        player_id TEXT,
                        start_seconds REAL,
                        end_seconds REAL
                    );
                    INSERT INTO materials(
                        material_id, source_video_id, segment_id, video_path,
                        start_seconds, end_seconds, processing_status
                    ) VALUES ('m1', 'v1', 's1', 'm1.mp4', 0, 2, 'ready');
                    INSERT INTO material_events(
                        material_id, event_name, confidence, player_id
                    ) VALUES ('m1', 'Made Shot', 0.8, 'player_3');
                    """
                )
                connection.commit()
            finally:
                connection.close()

            database = ProductDatabase.open(root)

            self.assertEqual(database.schema_version(), 2)
            materials = database.find_materials(event="Made Shot")
            self.assertEqual(len(materials), 1)
            self.assertIsNone(materials[0].events[0].participant_id)

    def test_material_service_persists_identity_and_event_lookup(self):
        """产品素材应保留稳定人物编号，并支持事件与人物组合检索。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"
            product = ProductDatabase.open(root)
            service = CatalogService()
            prediction = {
                "player_predictions": [
                    {
                        "player_id": "player_0",
                        "participant_id": "person_test_20",
                        "jersey_color": "white",
                        "jersey_number": "20",
                        "event": "ast",
                        "confidence": 0.91,
                    }
                ],
                "temporal_events": [
                    {
                        "player_id": "player_0",
                        "event": "ast",
                        "confidence": 0.91,
                        "start_time": 1.6,
                        "end_time": 3.4,
                    }
                ],
            }

            item = service.build_material(
                source_video_id="upload_001",
                segment_id="segment_0001",
                video_path=product.storage.segments_dir / "segment_0001.mp4",
                start_seconds=10.0,
                end_seconds=22.0,
                prediction_report=prediction,
                metadata={"source": "user_upload"},
            )
            product.save_material(item)

            reopened = ProductDatabase.open(root)
            matches = reopened.find_materials(
                event="ast",
                participant_id="person_test_20",
                minimum_confidence=0.9,
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].material_id, "upload_001:segment_0001")
            self.assertEqual(
                matches[0].participants[0].participant_id, "person_test_20"
            )
            self.assertEqual(reopened.status()["counts"]["materials"], 1)

    def test_duplicate_material_is_rejected(self):
        """数据库默认拒绝覆盖同一素材编号。"""
        with tempfile.TemporaryDirectory() as directory:
            product = ProductDatabase.open(Path(directory) / "product_data")
            item = CatalogService().build_material(
                source_video_id="game-001",
                segment_id="clip-100",
                video_path="clips/100.mp4",
                start_seconds=0.0,
                end_seconds=12.0,
                prediction_report={"player_predictions": [], "temporal_events": []},
            )
            product.save_material(item)
            with self.assertRaises(ValueError):
                product.save_material(item)

    def test_combined_query_requires_same_event_actor(self):
        """素材中其他可见人物不能被误当作指定事件的主体。"""
        with tempfile.TemporaryDirectory() as directory:
            product = ProductDatabase.open(Path(directory) / "product_data")
            item = CatalogItem(
                material_id="material-1",
                source_video_id="video-1",
                segment_id="segment-1",
                video_path=Path("material-1.mp4"),
                start_seconds=0.0,
                end_seconds=5.0,
                processing_status="ready",
                events=(
                    EventTag(
                        event="Made Shot",
                        confidence=0.9,
                        participant_id="white-13",
                    ),
                    EventTag(
                        event="Foul",
                        confidence=0.8,
                        participant_id="black-5",
                    ),
                ),
                participants=(
                    ParticipantReference("white-13", "window/player_3"),
                    ParticipantReference("black-5", "window/player_8"),
                ),
            )
            product.save_material(item)

            self.assertEqual(
                len(
                    product.find_materials(event="Made Shot", participant_id="white-13")
                ),
                1,
            )
            self.assertEqual(
                product.find_materials(event="Made Shot", participant_id="black-5"),
                [],
            )


if __name__ == "__main__":
    unittest.main()
