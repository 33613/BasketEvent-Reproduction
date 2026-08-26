# BasketEvent Reproduction

BasketEvent Reproduction is being reorganized from a paper-reproduction
repository into a modular basketball-video processing backend. The current
system preserves the verified SAM3 → Qwen → TimeSformer/PlayNet inference path
while introducing simple product-facing modules for user-video ingestion,
long-video segmentation, and material statistics.

The immediate product goal is:

> Accept a basketball game video, split it into model-sized clips, identify
> tracked players by jersey color and number, classify player-level events, and
> organize the resulting clips as searchable materials.

Automatic highlight editing, databases, Web APIs, and user interfaces are not
implemented yet.

## Processing flow

```text
User video
  -> ingestion and media metadata
  -> long-video segmentation
  -> SAM3 player/ball tracking
  -> Qwen jersey recognition and trajectory filtering
  -> TimeSformer + PlayNet player-event recognition
  -> visualization and material statistics
```

`ingestion` is not a dataset downloader or a neural network. It is the product
boundary that validates an uploaded video, reads its FPS, duration, resolution,
and frame count, and assigns an internal `video_id`. Later storage or API code
can change where uploaded files live without changing downstream modules.

## Source layout

```text
src/
├── core/
│   └── config.py                 # Paths and runtime configuration
├── modules/
│   ├── ingestion/                # User-video metadata and legacy BARD adapters
│   ├── segmentation/             # Fixed-window long-video baseline
│   ├── tracking/                 # SAM3 tracking and TITAN compatibility
│   ├── identity/                 # Qwen player/jersey recognition
│   ├── event_recognition/        # Dataset, inference, training, solver, labels
│   │   └── playnet/              # PlayNet model assembly and layers
│   └── materials/                # Visualization and material statistics
└── application/
    ├── process_clip.py           # Existing complete single-clip workflow
    └── process_video.py          # User-video ingestion and segmentation use case
```

The application layer coordinates public module contracts. It does not contain
SAM3, Qwen, or PlayNet algorithms. An implementation can therefore be improved
inside its module without changing the stage ordering in the application.

The historical root scripts remain as compatibility entry points:

- `track_one_video.py`
- `recognize.py`
- `inference.py`
- `train.py`
- `local_script/process_one_video.py`
- `local_script/visualize_qwen_tracks.py`

New code should import from `src.modules` and `src.application` instead.

## Repository setup

Clone the repository together with the SAM3 submodule before running tracking:

```bash
git clone --recurse-submodules git@github.com:33613/BasketEvent-Reproduction.git
cd BasketEvent-Reproduction
git submodule update --init --recursive
```

The Python environment and model weights remain external to Git. Install the
project dependencies in an isolated environment and make sure `ffmpeg` is on
`PATH` before exporting long-video clips.

## Configuration

Central configuration lives in `src/core/config.py`. The root `settings.py`
module is retained only for backward compatibility.

Important environment variables:

| Variable | Default on the lab server |
| --- | --- |
| `BASKETEVENT_DATA_ROOT` | `/home/fangzilin/data/basket` |
| `BASKETEVENT_ARTIFACTS_ROOT` | `/home/fangzilin/data/basket_artifacts` |
| `BASKETEVENT_RUNTIME_ROOT` | `/home/fangzilin/data/basket_runtime` |
| `BASKETEVENT_MODEL_ROOT` | `/home/fangzilin/models` |
| `BASKETEVENT_GPU_IDS` | `0` |
| `BASKETEVENT_HF_LOCAL_FILES_ONLY` | `true` |

Expected model locations below `BASKETEVENT_MODEL_ROOT`:

```text
models/
├── sam3/sam3.pt
├── Qwen2.5-VL-7B-Instruct/
├── timesformer-base-finetuned-k400/
└── basketevent/playnet.pt
```

## Existing single-clip workflow

The verified server workflow remains available:

```bash
source /home/fangzilin/tools/miniconda3/etc/profile.d/conda.sh
conda activate /home/fangzilin/envs/basketevent
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

python -u local_script/process_one_video.py \
  --game bkn-vs-det-0022400861 \
  --clip 100
```

The application launches each GPU-heavy module in a separate process so SAM3,
Qwen, and PlayNet do not occupy GPU memory at the same time. Existing non-empty
artifacts are reused by default.

Each stage can also be invoked through its canonical module:

```bash
python -m src.modules.tracking.sam3_tracker --help
python -m src.modules.identity.qwen_recognizer --help
python -m src.modules.event_recognition.inference --help
python -m src.modules.materials.visualization --help
```

## Long-video baseline

The first long-video strategy creates overlapping fixed windows. It preserves
the exact source-video timestamps in a manifest and can optionally materialize
MP4 clips through FFmpeg.

```python
from pathlib import Path

from src.application.process_video import (
    LongVideoProcessingApplication,
    LongVideoProcessingConfig,
)
from src.modules.ingestion import VideoIngestionService
from src.modules.segmentation import LongVideoSegmenter

application = LongVideoProcessingApplication(
    ingestion=VideoIngestionService(),
    segmenter=LongVideoSegmenter(
        window_seconds=12,
        overlap_seconds=2,
    ),
    config=LongVideoProcessingConfig(
        output_root=Path("data/processed"),
        export_clips=False,
    ),
)

report = application.run("game.mp4")
```

Set `export_clips=True` after confirming that `ffmpeg` is available. This is a
baseline rather than the final event-aware segmentation method. Future replay,
scoreboard, audio, or shot-boundary strategies should keep the same `plan`
contract.

## Material statistics

PlayNet prediction reports can already be summarized without knowing real
player names:

```python
from src.modules.materials import MaterialStatisticsService

statistics = MaterialStatisticsService().summarize_files(
    ["clip_100_events.json", "clip_101_events.json"]
)
print(statistics.to_dict())
```

Participants are grouped with labels such as `white #20`. Database-backed
cataloging and cross-clip person re-identification are later milestones.

## Current compatibility limits

- The production Qwen module still follows the original one-pass roster-based
  filtering logic. Making the roster optional and adding temporal identity
  aggregation are the next identity-module changes.
- Fixed-window segmentation does not yet detect true possessions, replays, or
  broadcast transitions.
- The author checkpoint is useful for method validation, but predictions on
  BARD or arbitrary user videos can differ from the paper because the original
  training videos were not released.
- Training, diagnostics, and BARD conversion utilities are retained for
  reproducibility but are not part of the future upload workflow.

## Tests

Run the CPU-compatible suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Runtime diagnostic scripts under `tests/` may additionally require local model
weights and a CUDA GPU.

## Attribution

This repository is derived from the original BasketEvent research code and is
being extended for reproducibility and product prototyping. External datasets,
model weights, and upstream submodules retain their respective licenses and
attribution requirements.
