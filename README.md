# BasketEvent 视频素材理解系统

本仓库以 BasketEvent 论文方法为基础，目标是把一场篮球比赛视频自动处理成可检索、可分类的短片素材。当前阶段先保证数据处理链路清晰、可验证，再逐步接入长视频切分、身份归并、事件分类和素材管理。

## 当前处理流程

```text
用户视频
  → 视频接入与元数据检查
  → 长视频切分
  → SAM3 球员与篮球轨迹
  → 轨迹截图取样
  → Qwen 逐帧视觉观察
  → 身份解析与跨片段归并
  → TimeSformer + PlayNet 事件识别
  → 可视化与素材统计
```

模型权重、原始视频和生成的中间数据均不提交到 Git。代码通过集中配置读取它们。

## 目录结构

```text
BasketEvent/
├── config/
│   ├── bard_team_colors.example.json
│   └── environment/                 # 服务器环境固化文件
├── sam3/                            # SAM3 Git 子模块
├── src/
│   ├── application/                 # 只编排业务流程，不实现模型细节
│   │   ├── process_clip.py           # 单个短片完整处理
│   │   └── process_video.py          # 长视频接入与切分
│   ├── core/
│   │   └── config.py                 # 路径和运行配置
│   └── modules/
│       ├── ingestion/                # 视频与 BARD 数据接入
│       ├── segmentation/             # 长视频切分
│       ├── tracking/                 # SAM3 轨迹生成
│       ├── identity/                 # Qwen 观察与球员身份处理
│       ├── event_recognition/        # TimeSformer + PlayNet
│       └── materials/                # 可视化与素材统计
├── tests/                            # 单元测试和受控诊断脚本
├── requirements.txt
└── README.md
```

仓库不再保留根目录脚本或 `src` 根目录兼容文件。每项功能只有一份实现，统一通过 `python -m 完整模块路径` 调用。

## Identity 模块

```text
src/modules/identity/
├── sampling.py       # TrackSampler：按每条轨迹的有效帧独立取样
├── qwen_observer.py  # QwenTrackObserver：生成逐帧视觉观察
├── resolver.py       # IdentityResolver：聚合证据、可选名单检索和命令入口
├── clustering.py     # CrossClipIdentityClusterer：跨片段人物素材归并
└── models.py         # 阶段之间传递的中间数据结构
```

Identity 阶段遵循以下边界：

- `TrackSampler` 只负责读取视频和轨迹、产生带原始帧号的截图，不复制短轨迹末帧。
- `QwenTrackObserver` 只描述每张图是不是场上球员、球衣颜色和号码，不猜球员姓名，也不决定是否删除整条轨迹。
- `IdentityResolver` 把逐帧证据解析成 `stable`、`mixed`、`unresolved` 或 `invalid`。
- `mixed` 表示 SAM3 轨迹发生身份切换，必须先按时间拆分；不能用多数票覆盖冲突。
- `unresolved` 表示已确认是场上球员但号码不可读，仍保留给事件识别，避免旧版硬过滤造成漏检。
- 比赛名单是可选输入。没有名单时使用“球衣颜色 + 号码”作为人物标识。

身份处理会生成两个文件：

- 指定的 `json_save_path`：供 PlayNet 使用的干净轨迹。
- 同目录下的 `*_identity.json`：逐帧观察、解析状态和篮球选择的审计报告。

## 环境准备

服务器当前验证环境位于：

```bash
source /home/fangzilin/tools/miniconda3/etc/profile.d/conda.sh
conda activate /home/fangzilin/envs/basketevent
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
```

首次克隆后初始化 SAM3：

```bash
git submodule update --init --recursive
```

模型默认目录：

```text
/home/fangzilin/models/
├── sam3/sam3.pt
├── Qwen2.5-VL-7B-Instruct/
├── timesformer-base-finetuned-k400/
└── basketevent/playnet.pt
```

路径可通过以下环境变量覆盖：

- `BASKETEVENT_DATA_ROOT`
- `BASKETEVENT_ARTIFACTS_ROOT`
- `BASKETEVENT_RUNTIME_ROOT`
- `BASKETEVENT_MODEL_ROOT`
- `BASKETEVENT_GPU_IDS`
- `BASKETEVENT_HF_LOCAL_FILES_ONLY`

## 运行单个视频片段

完整流程入口位于 `src.application.process_clip`：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --sam3-gpus 0,1 \
  --qwen-gpus 0 \
  --playnet-gpu 0
```

已有中间结果时可从后续阶段继续：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --start-at qwen
```

只重新生成可视化：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --visualize-only
```

也可以单独运行 Identity 阶段：

```bash
python -m src.modules.identity.resolver \
  --video_path /home/fangzilin/data/basket/GAME/video/CLIP.mp4 \
  --bbox_json_path /home/fangzilin/data/basket_artifacts/GAME/tracks/raw/CLIP.json \
  --json_save_path /home/fangzilin/data/basket_artifacts/GAME/tracks/clean/CLIP.json \
  --roster_json /home/fangzilin/data/basket_artifacts/GAME/metadata/recognize_roster.json \
  --qwen_model /home/fangzilin/models/Qwen2.5-VL-7B-Instruct \
  --gpus_to_use 0
```

不提供 `--roster_json` 时，系统不会尝试映射真实姓名。

## 长视频基础切分

当前长视频模块先提供固定时长、带重叠窗口的可靠基线：

```bash
python -m src.application.process_video INPUT.mp4 \
  --output-root /home/fangzilin/data/video_jobs \
  --window-seconds 12 \
  --overlap-seconds 2 \
  --export-clips
```

后续可以在不修改应用层的前提下，把固定窗口替换为镜头边界、记分牌时间或比赛事件驱动的切分器。

## BARD 数据工具

BARD 相关工具已经归入数据接入模块：

```bash
python -m src.modules.ingestion.bard.prepare --help
python -m src.modules.ingestion.bard.annotations_cli --help
```

环境固化文件位于 `config/environment/`，比赛球衣颜色配置示例位于 `config/bard_team_colors.example.json`。

## 测试

```bash
python -m unittest discover -s tests -v
```

`tests/qwen_tests_runtime/` 和 `tests/track_segment_runtime/` 只保存运行时诊断结果的目录约定，生成的大文件不会提交。

## 当前限制

- TITAN RTX 不支持 Ampere Flash Attention，SAM3 使用兼容注意力路径，速度较慢。
- 跨片段聚类目前以球衣颜色和号码作为确定性基线，尚未接入人物 ReID 特征。
- `mixed` 轨迹已经能被发现，但自动确定身份切换边界仍是下一步工作。
- 长视频切分目前是固定窗口基线，还没有使用比赛时钟、文字播报或镜头语义。
- 作者未公开原始训练视频与完整标签生成过程，作者 checkpoint 仅用于方法链路验证，不能代表在 BARD 上的最终准确率。
