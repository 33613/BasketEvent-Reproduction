"""把全局时间线中的待剪范围导出为独立视频素材。"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _safe_filename(value: str) -> str:
    """把业务编号转换为 Windows 和 Linux 都可用的文件名。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return normalized.strip("._") or "material"


@dataclass(frozen=True)
class ExportedMaterial:
    """记录一段已经规划或实际导出的事件素材。"""

    material_id: str
    video_path: Path
    start_seconds: float
    end_seconds: float
    event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为可以写入 JSON 的字典。"""
        return {
            "material_id": self.material_id,
            "video_path": str(self.video_path),
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.end_seconds - self.start_seconds,
            "event_ids": list(self.event_ids),
        }


class MaterialExporter:
    """使用 FFmpeg 导出事件时间线生成的素材范围。"""

    def __init__(self, ffmpeg_binary: str = "ffmpeg", overwrite: bool = False) -> None:
        """保存 FFmpeg 路径和覆盖策略。"""
        self.ffmpeg_binary = ffmpeg_binary
        self.overwrite = overwrite

    def build_command(
        self,
        source_video: str | Path,
        destination: str | Path,
        start_seconds: float,
        end_seconds: float,
    ) -> list[str]:
        """构造一条可独立解码的 MP4 导出命令。"""
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise ValueError("素材时间范围必须满足 0 <= start < end")
        return [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self.overwrite else "-n",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(Path(source_video).expanduser()),
            "-t",
            f"{end_seconds - start_seconds:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(Path(destination).expanduser()),
        ]

    def export(
        self,
        *,
        source_video: str | Path,
        material_drafts: Sequence[Mapping[str, Any]],
        output_directory: str | Path,
        dry_run: bool = False,
    ) -> tuple[ExportedMaterial, ...]:
        """按时间顺序导出素材；已有非空文件默认复用。"""
        source = Path(source_video).expanduser()
        if not dry_run and not source.is_file():
            raise FileNotFoundError(f"源视频不存在：{source}")
        output_root = Path(output_directory).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        exported: list[ExportedMaterial] = []
        for index, draft in enumerate(material_drafts):
            material_id = str(draft.get("material_id") or f"material_{index:05d}")
            start = float(draft.get("start_seconds") or 0.0)
            end = float(draft.get("end_seconds") or 0.0)
            event_ids = tuple(str(value) for value in draft.get("event_ids", []))
            destination = output_root / f"{index:05d}_{_safe_filename(material_id)}.mp4"
            command = self.build_command(source, destination, start, end)
            if not dry_run and not (
                destination.is_file() and destination.stat().st_size > 0
            ):
                subprocess.run(command, check=True)
                if not destination.is_file() or destination.stat().st_size <= 0:
                    raise RuntimeError(f"FFmpeg 未生成有效素材文件：{destination}")
            exported.append(
                ExportedMaterial(
                    material_id=material_id,
                    video_path=destination,
                    start_seconds=start,
                    end_seconds=end,
                    event_ids=event_ids,
                )
            )
        return tuple(exported)
