"""调用 Qwen，把轨迹截图转换为逐帧身份观察。"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from src.modules.identity.models import (
    BallCandidate,
    IdentityObservation,
    TrackCrop,
)


def _normalize_color(value: Any) -> str | None:
    """规范化球衣颜色字段。"""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _normalize_number(value: Any) -> str | None:
    """规范化 0 至 99 的球衣号码，并保留 ``0`` 与 ``00`` 的区别。"""
    if value is None:
        return None
    normalized = str(value).strip()
    if not re.fullmatch(r"\d{1,2}", normalized):
        return None
    return normalized


def _normalize_confidence(value: Any) -> float:
    """把数值或 low/medium/high 置信度转换到 0 至 1。"""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
    return mapping.get(str(value).strip().lower(), 0.0)


def _decode_json_values(text: str) -> list[Any]:
    """从模型文本中恢复一个或多个完整JSON值。

    Qwen实际输出有时是标准对象，有时会返回顶层数组、连续多个对象，或在
    JSON前后添加Markdown代码围栏。这里先尝试严格解析；失败后再使用
    ``raw_decode`` 顺序扫描，避免原先的贪婪正则把多个对象拼在一起而触发
    ``Extra data``。
    """
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    if not cleaned:
        return []
    try:
        return [json.loads(cleaned)]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    values: list[Any] = []
    cursor = 0
    while cursor < len(cleaned):
        match = re.search(r"[\[{]", cleaned[cursor:])
        if match is None:
            break
        start = cursor + match.start()
        try:
            value, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        values.append(value)
        cursor = end
    return values


def _parse_json_object(text: str) -> dict[str, Any]:
    """把Qwen的常见JSON变体规范化为一个对象。"""
    values = _decode_json_values(text)
    if not values:
        raise ValueError("Qwen 输出中没有可解析的完整 JSON")

    # 标准的观察结果优先；若模型连续输出多个包装对象则合并其数组。
    wrapped_observations: list[Any] = []
    for value in values:
        if isinstance(value, Mapping) and isinstance(value.get("observations"), list):
            wrapped_observations.extend(value["observations"])
    if wrapped_observations:
        return {"observations": wrapped_observations}

    # 兼容顶层数组，以及逐行连续输出多个 observation 对象。
    observations: list[dict[str, Any]] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, Mapping) and (
                "image_index" in candidate
                or "is_on_court_player" in candidate
                or "jersey_number" in candidate
            ):
                observations.append(dict(candidate))
    if observations:
        return {"observations": observations}

    # 篮球候选等提示仍返回普通单对象，保持原有调用方式。
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    raise ValueError("Qwen 输出中没有可解析的 JSON 对象")


class QwenTrackObserver:
    """只负责视觉观察，不负责名单检索或轨迹保留决策。"""

    source = "qwen"

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        """保存已经加载的 Qwen 模型、处理器和推理设备。"""
        self.model = model
        self.processor = processor
        self.device = device
        self.last_output_text: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str,
        local_files_only: bool = True,
    ) -> "QwenTrackObserver":
        """以 4-bit 模式加载本地 Qwen；CPU 环境使用浮点权重。"""
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": {"": device},
                    "torch_dtype": torch.float16,
                }
            )
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        return cls(model=model, processor=processor, device=device)

    @staticmethod
    def build_prompt(crops: Sequence[TrackCrop]) -> str:
        """构造逐图观察提示词，明确禁止跨图多数投票和猜测姓名。"""
        frame_table = ", ".join(
            f"image {crop.image_index}=frame {crop.frame_index}" for crop in crops
        )
        return f"""You are reviewing chronological crops from one basketball tracking ID.
The tracker may switch from one person to another, so judge EACH image independently.
Do not identify a player name and do not force all images to share one jersey number.

Image to source-frame mapping: {frame_table}

For every image:
1. Decide whether the main boxed/cropped subject is an on-court basketball player.
2. Read jersey color when visible.
3. Read a 1-2 digit jersey number only when it is actually visible. Preserve 0 and 00.
4. Use null when uncertain; do not copy a number from another image.

Return STRICT JSON only:
{{
  "observations": [
    {{
      "image_index": 1,
      "is_on_court_player": true,
      "jersey_color": "white",
      "jersey_number": "13",
      "confidence": 0.9,
      "evidence": "short visual reason"
    }}
  ]
}}
"""

    def observe(self, crops: Sequence[TrackCrop]) -> list[IdentityObservation]:
        """一次调用观察同一轨迹的多张截图，并保留每张图的独立结论。"""
        if not crops:
            return []
        self.last_output_text = None
        content = [{"type": "image", "image": crop.image} for crop in crops]
        content.append({"type": "text", "text": self.build_prompt(crops)})
        output_text = self._generate([{"role": "user", "content": content}], 512)
        self.last_output_text = output_text
        return self.parse_observations(crops, output_text)

    @staticmethod
    def parse_observations(
        crops: Sequence[TrackCrop], output_text: str
    ) -> list[IdentityObservation]:
        """把模型 JSON 与原始帧索引对应，拒绝越界或重复的图像编号。"""
        document = _parse_json_object(output_text)
        values = document.get("observations")
        if not isinstance(values, list):
            values = [document]
        crops_by_index = {crop.image_index: crop for crop in crops}
        seen: set[int] = set()
        observations: list[IdentityObservation] = []
        for raw in values:
            if not isinstance(raw, Mapping):
                continue
            try:
                image_index = int(raw.get("image_index", crops[0].image_index))
            except (TypeError, ValueError):
                continue
            crop = crops_by_index.get(image_index)
            if crop is None or image_index in seen:
                continue
            seen.add(image_index)
            validity = raw.get("is_on_court_player")
            if not isinstance(validity, bool):
                validity = None
            observations.append(
                IdentityObservation(
                    track_id=crop.track_id,
                    image_index=image_index,
                    frame_index=crop.frame_index,
                    is_on_court_player=validity,
                    jersey_color=_normalize_color(raw.get("jersey_color")),
                    jersey_number=_normalize_number(raw.get("jersey_number")),
                    confidence=_normalize_confidence(raw.get("confidence")),
                    evidence=str(raw.get("evidence") or "") or None,
                    raw=dict(raw),
                )
            )
        return observations

    def select_ball(self, candidates: Sequence[BallCandidate]) -> dict[str, Any]:
        """从多个 SAM3 篮球候选中选择真实比赛用球。"""
        from src.modules.identity.sampling import build_ball_contact_sheet

        if not candidates:
            return {"selected_ball_id": None, "reason": "没有有效篮球候选"}
        if len(candidates) == 1:
            return {
                "selected_ball_id": candidates[0].track_id,
                "confidence": 1.0,
                "reason": "只有一个非空篮球候选",
            }
        sheet = build_ball_contact_sheet(candidates)
        statistics = [dict(candidate.statistics) for candidate in candidates]
        prompt = f"""Each row contains one candidate basketball track. Red boxes mark the candidate.
Choose the physical basketball used in play. Reject logos, shoes, players and overlays.
Candidate statistics: {json.dumps(statistics, ensure_ascii=False)}
Return STRICT JSON only:
{{"real_ball_id": "ball id or null", "confidence": 0.0, "reason": "short reason"}}
"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sheet},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output_text = self._generate(messages, 192)
        try:
            result = _parse_json_object(output_text)
        except (ValueError, json.JSONDecodeError) as error:
            return {
                "selected_ball_id": None,
                "confidence": 0.0,
                "reason": f"Qwen 输出无法解析：{error}",
                "raw_output": output_text,
            }
        allowed = {candidate.track_id for candidate in candidates}
        selected = result.get("real_ball_id")
        selected = str(selected) if selected is not None else None
        if selected not in allowed:
            selected = None
        return {
            "selected_ball_id": selected,
            "confidence": _normalize_confidence(result.get("confidence")),
            "reason": result.get("reason"),
            "raw_output": output_text,
        }

    def _generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        """执行一次确定性多模态生成，并只返回新增文本。"""
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
