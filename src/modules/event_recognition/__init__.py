"""提供球员级事件识别及源视频全局时间线整理。"""

from src.modules.event_recognition.labels import LABEL_MAP, SUPPORTED_LABELS
from src.modules.event_recognition.timeline import EventTimelineService

__all__ = ["EventTimelineService", "LABEL_MAP", "SUPPORTED_LABELS"]
