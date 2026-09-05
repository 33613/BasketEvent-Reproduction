"""无需 GPU 的评估回归测试：顺序、身份、漏检、重复、数据隔离。"""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.application.evaluate_bard import (
    build_bundle, draft_events, inside, load_product_predictions, materialize_bundle,
    read_json, run_bundle, score_bundle, score_tracks, sha256, validate_annotations, verify_bundle, write_json,
)
from src.modules.evaluation.metrics import (
    interval_overlap, lcs_pairs, player_pair_metrics, sequence_metrics, temporal_hits,
)


def event(label="Made Shot", actor="white#13"):
    """构造最小事件记录。"""
    return {"event": label, "actor": actor}


class MetricsTest(unittest.TestCase):
    def test_lcs_penalizes_order_and_duplicates(self):
        reference = [event(), event("Rebound")]
        self.assertEqual(len(lcs_pairs(reference[::-1], reference)), 1)
        result = sequence_metrics([{"sample_id": "x", "references": reference,
                                    "predictions": [event(), event(), event("Rebound")]}])
        self.assertEqual(result["full_event"]["matches"], 2)
        self.assertEqual(result["full_event"]["predicted"], 3)

    def test_unknown_identity_still_counts_as_prediction(self):
        result = sequence_metrics([{"sample_id": "x", "references": [event()],
                                    "predictions": [event(actor=None)]}])
        self.assertEqual(result["event_type"]["f1"], 1)
        self.assertEqual(result["full_event"]["f1"], 0)
        self.assertEqual(result["participant_accuracy"], 0)

    def test_micro_aggregates_counts_and_missing_clips(self):
        result = sequence_metrics([
            {"sample_id": "a", "references": [event()], "predictions": [event()]},
            {"sample_id": "b", "references": [event()] * 3, "predictions": []}])
        self.assertAlmostEqual(result["full_event"]["f1"], 0.4)

    def test_pair_set_deduplicates_and_uses_fixed_ten_classes(self):
        result = player_pair_metrics([{"sample_id": "a", "references": [event()],
                                      "predictions": [event(), event()]}])
        self.assertEqual(result["macro_f1_fixed_10"], 0.1)
        self.assertEqual(result["per_class"]["Made Shot"]["predicted"], 1)
        self.assertIsNone(result["recall_at_k"])

    def test_recall_ranks_pairs_globally_in_clip(self):
        result = player_pair_metrics([{"references": [event()], "predictions": [],
                                      "ranked_pairs": [dict(event("Foul"), confidence=.9),
                                                       dict(event(), confidence=.8)]}])
        self.assertEqual(result["recall_at_k"]["1"], 0)
        self.assertEqual(result["recall_at_k"]["3"], 0.1)

    def test_modified_overlap_is_not_union_iou(self):
        self.assertEqual(interval_overlap([0, 10], [4, 5], True), 1)
        self.assertEqual(interval_overlap([0, 10], [4, 5]), .1)

    def test_hit_threshold_strict_and_wrong_class_fails(self):
        targets = [{"reference": dict(event(), interval=[0, 10]),
                    "prediction": dict(event(), interval=[5, 15])},
                   {"reference": dict(event(), interval=[0, 10]), "prediction": None},
                   {"reference": dict(event(), interval=[0, 10]),
                    "prediction": dict(event("Foul"), interval=[0, 10])}]
        result = temporal_hits(targets)
        self.assertEqual(result["hit_at_0.5"], 0)
        self.assertAlmostEqual(result["hit_at_0.3"], 1 / 3)


class BundleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_drafts_keep_multiple_events_per_person(self):
        events, _ = draft_events({"actions": [
            {"action": "Rebound", "color": "white", "player": "13"},
            {"action": "2PT Shot", "result": True, "color": "white", "player": "13"}]})
        self.assertEqual([e["event"] for e in events], ["Rebound", "Made Shot"])

    def test_builder_does_not_require_clean_tracks_and_copy_is_verified(self):
        game = self.root / "data" / "game-0022400001"
        (game / "video").mkdir(parents=True)
        (game / "video" / "1.mp4").write_bytes(b"fixture-only-not-a-real-video")
        write_json(game / "description/action/1.json", {"actions": [
            {"action": "Rebound", "color": "white", "player": "00"}]})
        bundle = self.root / "bundle"
        args = argparse.Namespace(data_root=self.root / "data", output=bundle,
                                  games=None, game_count=1, seed=123, per_class=1,
                                  copy_videos=False, all_clips=False)
        build_bundle(args)
        manifest, annotations = read_json(bundle / "manifest.json"), read_json(bundle / "annotations.json")
        self.assertEqual(len(manifest["items"]), 1)
        self.assertEqual(annotations["samples"][0]["events"][0]["actor"], "white#00")
        with self.assertRaises(ValueError):
            validate_annotations(manifest, annotations)
        materialize_bundle(bundle, "hardlink")
        self.assertEqual(verify_bundle(bundle)["verified_videos"], 1)
        (bundle / manifest["items"][0]["video"]).write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            verify_bundle(bundle)

    def test_paths_cannot_escape_bundle(self):
        with self.assertRaises(ValueError):
            inside(self.root, "../elsewhere.mp4")

    def test_product_adapter_retains_anonymous_and_orders_by_evidence(self):
        write_json(self.root / "event_timeline.json", {"events": [
            {"event_id": "b", "event": "Foul", "evidence_start_seconds": 4,
             "evidence_end_seconds": 5, "confidence": .8},
            {"event_id": "a", "event": "Made Shot", "evidence_start_seconds": 1,
             "evidence_end_seconds": 2, "confidence": .9}]})
        write_json(self.root / "event_identity.json", {"event_resolutions": [
            {"event_id": "a", "status": "identified", "jersey_color": "white", "jersey_number": "00"}]})
        records = load_product_predictions(self.root)
        self.assertEqual(records[0]["actor"], "white#00")
        self.assertIsNone(records[1]["actor"])

    def test_score_counts_unrun_sample_as_missing(self):
        write_json(self.root / "manifest.json", {"items": [{"sample_id": "x"}]})
        write_json(self.root / "annotations.json", {"samples": [
            {"sample_id": "x", "reviewed": True, "reviewer": "人工", "events": [event()]}]})
        result = score_bundle(argparse.Namespace(bundle=self.root, run_root=self.root / "run",
                              annotations=None, allow_draft=False,
                              accept_bard_silver=False, output=None))
        self.assertEqual(result["full_event"]["reference"], 1)
        self.assertEqual(result["full_event"]["recall"], 0)

    def test_all_clips_includes_known_development_game_and_empty_action_clip(self):
        game = self.root / "data" / "bkn-vs-det-0022400861"
        (game / "video").mkdir(parents=True)
        (game / "description/action").mkdir(parents=True)
        (game / "video/1.mp4").write_bytes(b"one")
        (game / "video/2.mp4").write_bytes(b"two")
        write_json(game / "description/action/1.json", {"actions": []})
        bundle = self.root / "complete"
        result = build_bundle(argparse.Namespace(
            data_root=self.root / "data", output=bundle,
            games=[game.name], game_count=1, seed=1, per_class=3,
            copy_videos=False, all_clips=True))
        self.assertEqual(result["sample_count"], 2)
        manifest = read_json(bundle / "manifest.json")
        self.assertEqual(manifest["sampling"], "all_clips_from_explicit_games")
        self.assertEqual(manifest["known_development_games_included"], [game.name])

    def test_silver_score_disables_sequence_metric_and_reports_each_game(self):
        write_json(self.root / "manifest.json", {
            "games": ["g"], "items": [{"sample_id": "x", "game": "g"}]})
        write_json(self.root / "annotations.json", {"samples": [
            {"sample_id": "x", "reviewed": False, "reviewer": "", "events": [event()]}]})
        result = score_bundle(argparse.Namespace(
            bundle=self.root, run_root=self.root / "run", annotations=None,
            allow_draft=False, accept_bard_silver=True, output=None))
        report = read_json(Path(result["output"]))
        self.assertIsNone(report["q8_adapted"])
        self.assertIsNone(report["per_game"]["g"]["q8_adapted"])
        self.assertEqual(result["claim"], "bard_silver_unordered_evaluation")

    def test_score_tracks_has_missing_target_in_denominator(self):
        write_json(self.root / "prediction.json", {"player_predictions": [
            {"player_id": "player_0", "event": "Made Shot",
             "class_probabilities": {"Made Shot": .9, "blank": .1},
             "paper_gate_segment": {"start_time": 1, "end_time": 2}}]})
        targets = self.root / "targets.json"
        write_json(targets, {"reviewed": True, "reviewer": "人工", "samples": [
            {"sample_id": "x", "prediction_json": "prediction.json", "references": [
                dict(event(actor="player_0"), interval=[1, 2]),
                dict(event("Foul", "missing_0"), interval=[3, 4])]}]})
        result = score_tracks(argparse.Namespace(targets=targets, output=self.root / "scores.json"))
        self.assertEqual(result["temporal"]["hit_at_0.5"], .5)

    def test_runner_probes_fps_and_keeps_truth_out_of_command(self):
        """用替身验证应用编排，不冒充实际 GPU 推理验证。"""
        video = self.root / "videos" / "x.mp4"
        video.parent.mkdir()
        video.write_bytes(b"fixture")
        write_json(self.root / "manifest.json", {"items": [
            {"sample_id": "x", "video": "videos/x.mp4", "sha256": sha256(video)}]})
        settings = SimpleNamespace(event_checkpoint=video, sam3_checkpoint=video,
                                   timesformer_model=self.root, qwen_model=self.root)
        capture = SimpleNamespace(get=lambda _: 30.0, release=lambda: None)
        cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=5)
        commands = []

        def execute(command, **kwargs):
            if command[0] == "git":
                return SimpleNamespace(stdout="test-commit", returncode=0)
            commands.append(command)
            root = Path(command[command.index("--runtime-root") + 1])
            write_json(root / "job" / "job_state.json", {"status": "completed"})
            return SimpleNamespace(returncode=0)

        with patch.dict("sys.modules", {"cv2": cv2}), \
                patch("src.core.config.SETTINGS", settings), \
                patch("src.application.evaluate_bard.subprocess.run", side_effect=execute):
            result = run_bundle(argparse.Namespace(
                bundle=self.root, run_root=self.root / "run", pipeline_mode="product",
                limit=1, gpu=None, sam3_gpus="0,1", playnet_gpu=1,
                identity_gpus="1", ffmpeg_binary="ffmpeg"))
        self.assertEqual(result["results"]["x"]["status"], "completed")
        self.assertEqual(commands[0][commands[0].index("--fps-in") + 1], "30")
        self.assertEqual(commands[0][commands[0].index("--sam3-gpus") + 1], "0,1")
        self.assertEqual(commands[0][commands[0].index("--playnet-gpu") + 1], "1")
        self.assertEqual(commands[0][commands[0].index("--identity-gpus") + 1], "1")
        self.assertNotIn("annotations.json", " ".join(commands[0]))
        self.assertNotIn("--roster-json", commands[0])

    def test_old_predictions_cannot_fake_paper_metrics(self):
        write_json(self.root / "old.json", {"player_predictions": [
            {"player_id": "player_0", "event": "Made Shot"}]})
        path = self.root / "targets.json"
        write_json(path, {"reviewed": True, "reviewer": "人工", "samples": [
            {"sample_id": "x", "prediction_json": "old.json", "references": [event(actor="player_0")]}]})
        with self.assertRaisesRegex(ValueError, "旧缓存"):
            score_tracks(argparse.Namespace(targets=path, output=self.root / "scores.json"))


if __name__ == "__main__":
    unittest.main()
