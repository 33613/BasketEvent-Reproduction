"""Runtime compatibility hooks for SAM3 on the BasketEvent server.

SAM3 constructs one predictor in each GPU worker process. Mutating only the
parent predictor after construction therefore leaves worker-side limits
unchanged. This module adds small request types to the exact predictor base
class used by the imported builder, allowing the existing multi-GPU dispatcher
to broadcast object limits, temporal-memory limits, and frame-cache offloading
to every rank.
"""

import math
from functools import wraps
from typing import Any, Callable


_CONFIGURE_OBJECT_LIMIT = "configure_object_limit"
_CONFIGURE_TRACKER_MEMORY = "configure_tracker_memory"
_CONFIGURE_FRAME_CACHE_OFFLOAD = "configure_frame_cache_offload"
_INSTALL_MARKER = "_basketevent_object_limit_request"
_FRAME_CACHE_MARKER = "_basketevent_frame_cache_offload"


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
        """Handle BasketEvent's limit requests or delegate to standard SAM3."""
        request_type = request.get("type")
        if request_type == _CONFIGURE_TRACKER_MEMORY:
            return _configure_tracker_memory(self, request)
        if request_type == _CONFIGURE_FRAME_CACHE_OFFLOAD:
            return _configure_frame_cache_offload(self)
        if request_type != _CONFIGURE_OBJECT_LIMIT:
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


def _move_nested(value, tensor_mover):
    """Recursively move tensor-like leaves while preserving containers.

    Args:
        value: Nested dictionaries, lists, tuples, or a leaf value.
        tensor_mover: Callable that receives each leaf value.

    Returns:
        A container with tensor-like leaves replaced by moved values.
    """
    if isinstance(value, dict):
        return {key: _move_nested(item, tensor_mover) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested(item, tensor_mover) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, tensor_mover) for item in value)
    return tensor_mover(value)


def _to_cpu_if_supported(value):
    """Move a tensor-like value to CPU and leave ordinary values unchanged."""
    cpu_method = getattr(value, "cpu", None)
    return cpu_method() if callable(cpu_method) else value


def _to_device_if_supported(value, device):
    """Move a tensor-like value to one device and preserve ordinary values."""
    to_method = getattr(value, "to", None)
    return to_method(device) if callable(to_method) else value


def _configure_frame_cache_offload(predictor):
    """Keep SAM3's per-frame detector output cache in CPU memory.

    SAM3's tracker-state offload does not cover ``cached_frame_outputs``. The
    detector appends object masks to that cache for every processed frame, so a
    long video can still exhaust a 24 GB GPU even when tracker state and source
    frames are offloaded. This instance-level wrapper moves new cache entries to
    CPU and restores them to the model device only when SAM3 reads them again.

    Args:
        predictor: Predictor instance owned by one GPU rank.

    Returns:
        Applied state and current rank for runtime diagnostics.

    Raises:
        RuntimeError: If this SAM3 version lacks the frame-cache methods or a
            model device needed to restore cached tensors.
    """
    model = getattr(predictor, "model", None)
    required_attributes = ("_cache_frame_outputs", "_build_tracker_output", "device")
    if model is None or any(
        not hasattr(model, attribute) for attribute in required_attributes
    ):
        raise RuntimeError(
            "The installed SAM3 model does not expose a compatible frame cache"
        )

    rank = int(getattr(model, "rank", 0))
    if not getattr(model, _FRAME_CACHE_MARKER, False):
        original_cache_frame_outputs = model._cache_frame_outputs
        original_build_tracker_output = model._build_tracker_output

        @wraps(original_cache_frame_outputs)
        def cache_frame_outputs_on_cpu(inference_state, frame_idx, *args, **kwargs):
            """Write one cache entry with all tensor leaves on CPU."""
            result = original_cache_frame_outputs(
                inference_state, frame_idx, *args, **kwargs
            )
            frame_cache = inference_state.get("cached_frame_outputs", {})
            if frame_idx in frame_cache:
                frame_cache[frame_idx] = _move_nested(
                    frame_cache[frame_idx], _to_cpu_if_supported
                )
            return result

        @wraps(original_build_tracker_output)
        def build_tracker_output_on_device(*args, **kwargs):
            """Restore cached tensor leaves before SAM3 consumes an output."""
            output = original_build_tracker_output(*args, **kwargs)
            return _move_nested(
                output,
                lambda value: _to_device_if_supported(value, model.device),
            )

        model._cache_frame_outputs = cache_frame_outputs_on_cpu
        model._build_tracker_output = build_tracker_output_on_device
        setattr(model, _FRAME_CACHE_MARKER, True)

    print(
        f"SAM3 rank frame cache: rank={rank}, device=cpu",
        flush=True,
    )
    return {"is_success": True, "rank": rank, "frame_cache_device": "cpu"}


def _configure_tracker_memory(predictor, request):
    """Apply temporal-memory limits before a SAM3 session is initialized.

    Args:
        predictor: Predictor instance owned by one GPU rank.
        request: Compatibility request containing the two memory limits.

    Returns:
        Applied values and the current rank for runtime diagnostics.

    Raises:
        ValueError: If either limit is less than one.
        RuntimeError: If the installed SAM3 tracker lacks the expected
            configurable attributes.
    """
    num_maskmem = int(request["num_maskmem"])
    max_cond_frames = int(request["max_cond_frames_in_attn"])
    if num_maskmem < 1:
        raise ValueError("num_maskmem must be at least one")
    if max_cond_frames < 2:
        raise ValueError("max_cond_frames_in_attn must be at least two")

    model = getattr(predictor, "model", None)
    tracker = getattr(model, "tracker", None)
    required_attributes = ("num_maskmem", "max_cond_frames_in_attn")
    if tracker is None or any(
        not hasattr(tracker, attribute) for attribute in required_attributes
    ):
        raise RuntimeError(
            "The installed SAM3 tracker does not expose temporal-memory limits"
        )

    rank = int(getattr(model, "rank", 0))
    tracker.num_maskmem = num_maskmem
    tracker.max_cond_frames_in_attn = max_cond_frames
    print(
        "SAM3 rank memory limits: "
        f"rank={rank}, num_maskmem={tracker.num_maskmem}, "
        f"max_cond_frames_in_attn={tracker.max_cond_frames_in_attn}",
        flush=True,
    )
    return {
        "is_success": True,
        "rank": rank,
        "num_maskmem": tracker.num_maskmem,
        "max_cond_frames_in_attn": tracker.max_cond_frames_in_attn,
    }
