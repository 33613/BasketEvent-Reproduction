# BasketEvent

[*Yu Zhang*](https://github.com/zhangyu2003/),
[*Jiayuan Rao*](https://jyrao.github.io/),
[*Haoning Wu*](https://haoningwu3639.github.io/),
[*Weidi Xie*](https://weidixie.github.io/)

<div style="line-height: 1;">
  <a href="https://zhangyu2003.github.io/BasketEvent/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website🌐-BasketEvent-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<img src="images/overview.png" alt="BasketEvent overview" width="100%">
BasketEvent is an automatic player-level event-recognition project for basketball game
videos. Starting from raw broadcast videos, it first uses SAM3 for open-vocabulary
segmentation and tracking of players and the ball, then uses Qwen2.5-VL to clean
the tracking results, identify valid players, and select the real ball trajectory.
Finally, a TimeSformer-based model predicts player-level
basketball events.

The project implements the following functions:

- Track on-court players and ball candidates from basketball videos.
- Filter invalid tracks such as referees, audience members, bench players, staff,
  and false detections.
- Map valid player tracks to real player names using jersey color, jersey number,
  and roster information.
- Predict player-level events using video clips, player bounding boxes, and ball
  bounding boxes.
- Support training, evaluation, and single-video inference.


## 1. Quickstart

### 1.1 Environment Setup

SAM3 recommends Python 3.12, PyTorch 2.7, and CUDA 12.6. All Python
dependencies are listed in `requirements.txt`.

```bash
cd /path/to/BasketEvent

conda create -n sam3 python=3.12
conda activate sam3

pip install -r requirements.txt
```

If `torchvision.io.read_video` cannot read videos, make sure FFmpeg is installed
on the system.

### 1.2 Local path configuration

`settings.py` centralizes server paths. The checked-in defaults are:

```text
/home/fangzilin/data/basket             BARD staging data
/home/fangzilin/data/basket_artifacts   reusable SAM3/Qwen outputs
/home/fangzilin/data/basket_runtime     videos/train/valid/test runtime layout
/home/fangzilin/models                  downloaded and trained model weights
```

The corresponding model paths are:

```text
/home/fangzilin/models/sam3/sam3.pt
/home/fangzilin/models/Qwen2.5-VL-7B-Instruct/
/home/fangzilin/models/timesformer-base-finetuned-k400/
/home/fangzilin/models/basketevent/playnet.pt
```

Defaults can be changed without editing source code:

```bash
export BASKETEVENT_DATA_ROOT=/home/fangzilin/data/basket
export BASKETEVENT_ARTIFACTS_ROOT=/home/fangzilin/data/basket_artifacts
export BASKETEVENT_RUNTIME_ROOT=/home/fangzilin/data/basket_runtime
export BASKETEVENT_MODEL_ROOT=/home/fangzilin/models
export BASKETEVENT_GPU_IDS=0
export BASKETEVENT_HF_LOCAL_FILES_ONLY=true
```

Entry-point options override these environment variables. Run, for example,
`python inference.py --help` to see path overrides for one experiment. Importing
`settings.py` only reads configuration; output directories are created by the
entry point that needs them.

### 1.3 SAM3 Checkpoint

SAM3 checkpoint:

```bash
hf auth login
huggingface-cli download facebook/sam3
```

URL:

```text
https://huggingface.co/facebook/sam3
```

### 1.4 Qwen2.5-VL Checkpoint

Qwen2.5-VL checkpoint:

```bash
hf auth login
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /home/fangzilin/models/Qwen2.5-VL-7B-Instruct
```

URL:

```text
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

## 2. Data Storage Format

The code organizes videos and annotation files using a two-level
`game_id/video_name` structure. During training and evaluation,
`src/dataset.py` scans `bbox_dir/{game_id}/*.json` and automatically finds the
corresponding video:

```text
video_path = video_dir/{game_id}/{video_name}.mp4
bbox_path  = bbox_dir/{game_id}/{video_name}.json
```

Recommended directory layout:

```text
/DB/data/yuzhang/basketball/
├── videos/
│   └── {game_id}/
│       └── {video_name}.mp4
├── train/
│   └── {game_id}/
│       └── {video_name}.json
├── valid/
│   └── {game_id}/
│       └── {video_name}.json
└── test/
    └── {game_id}/
        └── {video_name}.json
```

Directory meanings:

- `videos/` stores raw basketball game videos.
- `train/`, `valid/`, and `test/` store SAM3/Qwen-cleaned trajectories plus
  deterministically mapped event labels.

### 2.1 BARD Scheme-A annotation pipeline

BARD source data remains read-only under `Settings.data_root`. Reusable model
outputs and deterministic labels are kept separately:

```text
/home/fangzilin/data/basket_artifacts/
└── {bard_game}/
    ├── metadata/
    │   ├── recognize_roster.json
    │   └── roster_report.json
    ├── tracks/
    │   ├── raw/{clip}.json
    │   └── clean/{clip}.json
    ├── annotations/{clip}.json
    └── reports/{clip}.json
```

`src/bard/labeling.py` generates labels using fixed rules only. It maps BARD
2PT/3PT results to `Made Shot` or `Missed Shot`, preserves the direct event
classes supported by `src/labels.py`, and generates `ast` for the
`other_player` of an assisted shot. Unknown actions are reported rather than
silently changed to `blank`.

Scheme A accepts a clip only when every event actor matches exactly one cleaned
track by `(jersey_color, jersey_number)` and each actor has at most one distinct
event class. Every excluded clip still receives a machine-readable report.
Repeated instances of the same class are collapsed and reported as a warning.

First create the lightweight output roots on the server:

```bash
mkdir -p /home/fangzilin/data/basket_artifacts
mkdir -p /home/fangzilin/data/basket_runtime/{videos,train,valid,test,cache}
```

Copy the team-color template outside Git and fill it by independently inspecting
each game. Never derive these colors from the BARD action labels:

```bash
cp local/bard_team_colors.example.json \
  /home/fangzilin/data/basket_artifacts/team_colors.json

python local_script/build_bard_annotations.py rosters \
  --team-colors /home/fangzilin/data/basket_artifacts/team_colors.json
```

For one clip, write SAM3 and Qwen outputs to the artifact paths:

```bash
GAME=bkn-vs-det-0022400861
CLIP=100

python track_one_video.py \
  --video_path "/home/fangzilin/data/basket/$GAME/video/$CLIP.mp4" \
  --json_save_path "/home/fangzilin/data/basket_artifacts/$GAME/tracks/raw/$CLIP.json" \
  --gpus_to_use "0,1" \
  --offload-video-to-cpu \
  --offload-state-to-cpu \
  --max-num-objects 10 \
  --max-ball-objects 2 \
  --sam3-num-maskmem 3 \
  --sam3-max-cond-frames 2

python recognize.py \
  --video_path "/home/fangzilin/data/basket/$GAME/video/$CLIP.mp4" \
  --bbox_json_path "/home/fangzilin/data/basket_artifacts/$GAME/tracks/raw/$CLIP.json" \
  --json_save_path "/home/fangzilin/data/basket_artifacts/$GAME/tracks/clean/$CLIP.json" \
  --roster_json "/home/fangzilin/data/basket_artifacts/$GAME/metadata/recognize_roster.json"
```

The recommended server entry point now runs SAM3, Qwen, PlayNet, and the
diagnostic visualization with the same paths and memory-safe defaults:

```bash
source /home/fangzilin/tools/miniconda3/etc/profile.d/conda.sh
conda activate /home/fangzilin/envs/basketevent
cd /home/fangzilin/project/BasketEvent

python -u local_script/process_one_video.py \
  --game bkn-vs-det-0022400861 \
  --clip 130
```

The runner sets `PYTHONNOUSERSITE=1` and `TOKENIZERS_PARALLELISM=false` for
every child process, creates all output directories, and reuses non-empty
intermediate files after an interruption. Use
`--start-at qwen --no-resume` to retry Qwen without repeating SAM3, or
`--start-at visualize --no-resume` to regenerate an overlay after visualization
code changes. A per-clip status report is written
to `basket_artifacts/{game}/reports/{clip}_pipeline.json`. If Qwen retains no
players, PlayNet is skipped and a SAM3/Qwen diagnostic overlay is still made.

Render all raw SAM3 candidates together with Qwen's retained jersey numbers
before generating labels. Green player boxes were retained by Qwen, orange player
boxes were rejected, yellow is the selected basketball, and gray boxes are
unselected basketball candidates:

```bash
mkdir -p "/home/fangzilin/data/basket_artifacts/$GAME/visualizations"

python local_script/visualize_qwen_tracks.py \
  --video_path "/home/fangzilin/data/basket/$GAME/video/$CLIP.mp4" \
  --raw_json_path "/home/fangzilin/data/basket_artifacts/$GAME/tracks/raw/${CLIP}_dual_cachecpu.json" \
  --clean_json_path "/home/fangzilin/data/basket_artifacts/$GAME/tracks/clean/$CLIP.json" \
  --prediction_json_path "/home/fangzilin/data/basket_artifacts/$GAME/predictions/${CLIP}_events.json" \
  --output_video_path "/home/fangzilin/data/basket_artifacts/$GAME/visualizations/${CLIP}_qwen_overlay.mp4"
```

The renderer writes a JSON report beside the output MP4. New clean files also
include `source_track_id` for direct raw-to-clean traceability; older files
without that field are matched by their copied trajectories. The overlay MP4
is diagnostic and intentionally has no audio. The optional prediction JSON
adds a bottom timeline: each colored segment is a sampled clip that most
strongly supported a final non-background player event, and the white line is
the current video time. These segments are model-evidence windows rather than
manually supervised event boundaries. The legend maps each color to a jersey
number and predicted event; multiple evidence windows for the same pair always
use the same color.

On the two pre-Ampere TITAN RTX GPUs, `track_one_video.py` broadcasts each
object limit to all SAM3 ranks, divides the tracker slots between the GPUs, and
removes low-confidence initial candidates before propagation. The flags above
therefore limit both the initial prompt batch and detections added later. The
two memory flags retain less temporal history than upstream SAM3 (3 versus 7
mask-memory frames and 2 versus 4 conditioning frames) to bound attention
memory without reducing the ten-player object limit. When state offloading is
enabled, the compatibility layer also keeps SAM3's per-frame detector mask
cache on CPU; otherwise that separate cache grows throughout a long clip even
though tracker state is already offloaded.

Generate the fixed-rule annotation and inspect its report:

```bash
python local_script/build_bard_annotations.py labels \
  --games "$GAME" \
  --clips "$CLIP"

python -m json.tool \
  "/home/fangzilin/data/basket_artifacts/$GAME/reports/$CLIP.json"
```

After processing enough clips, split complete games and export only accepted
annotations to the original BasketEvent runtime layout:

```bash
python local_script/convert_bard_subset.py make-split

python local_script/convert_bard_subset.py export \
  --allow-missing-annotations \
  --materialize hardlink
```

The split is game-level to prevent clips from one game appearing in multiple
sets. `hardlink` avoids a second physical copy when the source and runtime roots
are on the same filesystem.

A single trajectory JSON file uses the following format. Bounding boxes are in
`xywh` format:

```json
{
  "player_0": {
    "jersey_number": 23,
    "jersey_color": "white",
    "player_name": "Player Name",
    "trajectory": [[x, y, w, h], null, [x, y, w, h]],
    "event": {
      "actionType": "Made Shot"
    }
  },
  "ball": {
    "trajectory": [[x, y, w, h], null, [x, y, w, h]]
  }
}
```

The training code uses `event.actionType` as the event label. Supported classes
are defined in `src/labels.py`:

Qwen2.5-VL player recognition requires a roster JSON. Example:

```json
{
  "jersey_color": {
    "Home Team": "white",
    "Away Team": "blue"
  },
  "players": [
    {
      "team_name": "Home Team",
      "jersey": "23",
      "name": "Player Name"
    }
  ]
}
```

## 3. Pipeline Steps

Run the following commands from the `BasketEvent` root directory.

### Step 1: SAM3 Tracking

File: `track_one_video.py`

Purpose: take a raw video as input, use SAM3 with text prompts to track on-court
players and the basketball, and export raw candidate trajectories.

```bash
python track_one_video.py \
  --video_path examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4 \
  --json_save_path examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720_raw.json \
  --gpus_to_use 0
```

Example output:

```text
examples/{video_name}_raw.json
```

### Step 2: Qwen2.5-VL Track Cleaning and Player Identification

File: `recognize.py`

Purpose: read the raw SAM3 JSON, filter invalid player tracks, identify players
using roster information, and select the real ball trajectory from multiple ball
candidates.

```bash
python recognize.py \
  --video_path examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4 \
  --bbox_json_path examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720_raw.json \
  --json_save_path examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json \
  --roster_json /path/to/{game_id}.json \
  --gpus_to_use 0
```

Example output:

```text
examples/{video_name}.json
```

### Step 3: Train the Event Recognition Model

File: `train.py`

Purpose: read cleaned trajectory JSON files and corresponding videos, construct
multi-clip bags, and train `PlayerEventModel`.

We'll provide detailed annotation json files to skip the first two steps. For the test dataset, we manually labeled the relationships between trajectories and events to make sure the data is correct.

```bash
torchrun --nproc_per_node=4 train.py \
  --bbox_dir data/train \
  --video_dir data/videos \
  --cache_dir cache \
  --save_dir ckpt_train \
  --bag_clips 4 \
  --clip_len 8 \
  --fps_in 25 \
  --fps_out 4 \
  --batch_size 1 \
  --epochs 10
```


### Step 4: Evaluate the Model

File: `test.py`

Purpose: evaluate top-k classification accuracy on the test set. When
`event_time_labels.csv` is available, the script can also evaluate temporal
localization-related metrics.

```bash
torchrun --nproc_per_node=4 test.py \
  --ckpt ckpt/epoch_best.pt \
  --test_dir data/test \
  --video_dir data/videos \
  --cache_dir cache \
  --time_csv data/event_time_labels.csv \
  --bag_clips 12 \
  --clip_len 8 \
  --fps_in 25 \
  --fps_out 4
```

### Step 5: Single-Video Inference

File: `inference.py`

Purpose: given one video and its cleaned trajectory JSON, output predicted
player-event pairs whose event class is not `blank`.

```bash
python inference.py \
  --video examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4 \
  --traj_json examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json \
  --checkpoint ckpt.pt \
  --bag_clips 12 \
  --clip_len 8 \
  --fps_in 25 \
  --fps_out 4 \
  --prediction_json_path outputs/events.json \
  --timeline_topk 2
```

When `--prediction_json_path` is supplied, inference exports final player
events plus their strongest time-aligned clip evidence. The visualization
command above consumes this report to draw the event timeline.
