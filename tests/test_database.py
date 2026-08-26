"""验证产品 SQLite 人物库、素材库和目录初始化。"""

import tempfile
import unittest
from pathlib import Path

from src.modules.catalog import MaterialCatalogService
from src.modules.database import ProductDatabase
from src.modules.identity.models import IdentityProfile


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
            self.assertEqual(product.status()["schema_version"], 1)

    def test_identity_profile_and_embedding_survive_reopen(self):
        """人物档案和 ReID 向量应在不同进程式对象之间持续存在。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"
            product = ProductDatabase.open(root)
            profile = IdentityProfile(
                participant_id="person_test_20",
                display_name=None,
                jersey_color="white",
                jersey_number="20",
                metadata={"source": "test"},
            )
            product.identity_gallery.save(profile)
            product.identity_gallery.add_embedding(
                profile.participant_id,
                [3.0, 4.0],
                model_name="test-reid",
                source_track_id="clip_100/player_8",
            )

            reopened = ProductDatabase.open(root)
            attribute_matches = reopened.identity_gallery.search_by_attributes(
                "white", "20"
            )
            embedding_matches = reopened.identity_gallery.search_by_embedding(
                [0.6, 0.8]
            )

            self.assertEqual(attribute_matches[0].participant_id, "person_test_20")
            self.assertEqual(embedding_matches[0].participant_id, "person_test_20")
            self.assertAlmostEqual(embedding_matches[0].score, 1.0)

    def test_material_service_persists_identity_and_event_lookup(self):
        """产品素材应保留稳定人物编号，并支持事件与人物组合检索。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "product_data"
            product = ProductDatabase.open(root)
            service = MaterialCatalogService(product.material_catalog)
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

            service.register_processed_clip(
                source_video_id="upload_001",
                segment_id="segment_0001",
                video_path=product.storage.segments_dir / "segment_0001.mp4",
                start_seconds=10.0,
                end_seconds=22.0,
                prediction_report=prediction,
                metadata={"source": "user_upload"},
            )

            reopened = ProductDatabase.open(root)
            matches = reopened.material_catalog.query(
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


if __name__ == "__main__":
    unittest.main()
