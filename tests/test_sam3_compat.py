"""Tests for the BasketEvent SAM3 multi-GPU compatibility request."""

import unittest

from src.sam3_compat import install_object_limit_request


class FakeModel:
    """Provide the SAM3 attributes modified by the compatibility hook."""

    def __init__(self):
        """Initialize an unrestricted two-rank fake video model."""
        self.max_num_objects = 10_000
        self.num_obj_for_compile = 16
        self.rank = 0
        self.world_size = 2


class Sam3BasePredictor:
    """Mimic SAM3's standard request delegation target."""

    def __init__(self):
        """Attach a configurable fake model."""
        self.model = FakeModel()

    def handle_request(self, request):
        """Return a marker for requests not owned by the compatibility hook."""
        return {"delegated_type": request["type"]}


class Sam3VideoPredictorMultiGPU(Sam3BasePredictor):
    """Expose the concrete class name expected in the builder namespace."""


def fake_builder():
    """Stand in for SAM3's video predictor builder."""
    return Sam3VideoPredictorMultiGPU()


class Sam3CompatibilityTest(unittest.TestCase):
    """Verify synchronized object-limit request behavior."""

    @classmethod
    def setUpClass(cls):
        """Install the idempotent hook on the fake predictor hierarchy."""
        install_object_limit_request(fake_builder)
        install_object_limit_request(fake_builder)

    def test_object_limit_is_divided_across_gpu_ranks(self):
        """A global limit of ten should reserve five slots on each rank."""
        predictor = fake_builder()

        response = predictor.handle_request(
            {"type": "configure_object_limit", "max_num_objects": 10}
        )

        self.assertEqual(predictor.model.max_num_objects, 10)
        self.assertEqual(predictor.model.num_obj_for_compile, 5)
        self.assertEqual(response["world_size"], 2)

    def test_standard_requests_are_delegated(self):
        """The hook must leave SAM3's existing requests unchanged."""
        predictor = fake_builder()

        response = predictor.handle_request({"type": "start_session"})

        self.assertEqual(response, {"delegated_type": "start_session"})

    def test_non_positive_limit_is_rejected(self):
        """Invalid limits should fail before mutating model state."""
        predictor = fake_builder()

        with self.assertRaisesRegex(ValueError, "positive integer"):
            predictor.handle_request(
                {"type": "configure_object_limit", "max_num_objects": 0}
            )


if __name__ == "__main__":
    unittest.main()
