"""Runtime compatibility hooks for SAM3 on the BasketEvent server.

SAM3 constructs one predictor in each GPU worker process. Mutating only the
parent predictor after construction therefore leaves worker-side object limits
unchanged. This module adds a small request type to the exact predictor base
class used by the imported builder, allowing the existing multi-GPU dispatcher
to broadcast the limit to every rank.
"""

import math
from functools import wraps
from typing import Any, Callable


_CONFIGURE_OBJECT_LIMIT = "configure_object_limit"
_INSTALL_MARKER = "_basketevent_object_limit_request"


def install_object_limit_request(
    predictor_builder: Callable[..., Any],
) -> None:
    """Install an idempotent multi-GPU object-limit request handler.

    The SAM3 builder keeps the concrete multi-GPU predictor class in its global
    namespace. Resolving the base class through that concrete class guarantees
    that the hook patches the same class used by the builder, even when SAM3 is
    imported through different editable-package paths.

    This function is intentionally called at module import time by
    ``track_one_video.py``. Python's ``spawn`` multiprocessing mode imports the
    entry module inside every SAM3 worker, so all ranks install the same request
    handler before their command loops start.

    Args:
        predictor_builder: Imported SAM3 video-predictor builder function.

    Raises:
        RuntimeError: If the concrete predictor or its base request handler
            cannot be resolved from the supplied builder.
    """
    predictor_class = predictor_builder.__globals__.get("Sam3VideoPredictorMultiGPU")
    if predictor_class is None:
        raise RuntimeError(
            "Cannot locate Sam3VideoPredictorMultiGPU in the SAM3 builder"
        )

    base_class = next(
        (
            candidate
            for candidate in predictor_class.__mro__
            if candidate.__name__ == "Sam3BasePredictor"
        ),
        None,
    )
    if base_class is None or not hasattr(base_class, "handle_request"):
        raise RuntimeError("Cannot locate the SAM3 base request handler")

    original_handler = base_class.handle_request
    if getattr(original_handler, _INSTALL_MARKER, False):
        return

    @wraps(original_handler)
    def handle_request_with_object_limit(self, request):
        """Handle BasketEvent's limit request or delegate to standard SAM3."""
        if request.get("type") != _CONFIGURE_OBJECT_LIMIT:
            return original_handler(self, request)

        max_num_objects = int(request["max_num_objects"])
        if max_num_objects <= 0:
            raise ValueError("max_num_objects must be a positive integer")

        model = getattr(self, "model", None)
        required_attributes = ("max_num_objects", "num_obj_for_compile")
        if model is None or any(
            not hasattr(model, attribute) for attribute in required_attributes
        ):
            raise RuntimeError(
                "The installed SAM3 model does not expose configurable object limits"
            )

        world_size = max(int(getattr(model, "world_size", 1)), 1)
        rank = int(getattr(model, "rank", 0))
        model.max_num_objects = max_num_objects
        model.num_obj_for_compile = math.ceil(max_num_objects / world_size)
        print(
            "SAM3 rank object limit: "
            f"rank={rank}, global={model.max_num_objects}, "
            f"per_gpu_slots={model.num_obj_for_compile}",
            flush=True,
        )
        return {
            "is_success": True,
            "rank": rank,
            "world_size": world_size,
            "max_num_objects": model.max_num_objects,
            "num_obj_for_compile": model.num_obj_for_compile,
        }

    setattr(handle_request_with_object_limit, _INSTALL_MARKER, True)
    base_class.handle_request = handle_request_with_object_limit
