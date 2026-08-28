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
  → 固定规则身份解析
  → TimeSformer + PlayNet 事件识别
  → 素材登记、检索与统计
```

模型权重、原始视频和生成的中间数据均不提交到 Git。代码通过集中配置读取它们。

## 目录结构

```text
BasketEvent/
├── config/
│   ├── bard_team_colors.example.json
│   └── environment/                 # 服务器环境固化文件
├── sam3/                            # SAM3 Git 子模块
├── product_data/                    # 产品数据库与用户媒体（真实内容不提交）
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
│       ├── event_recognition/        # TimeSformer + PlayNet 推理
│       ├── database/                 # SQLite 产品人物库和素材库
│       ├── catalog/                  # 素材登记、查询和统计
│       └── materials/                # 处理结果可视化
├── training/                         # Dataset、Solver 和训练入口
├── tests/                            # 单元测试和受控诊断脚本
├── requirements.txt
└── README.md
```

仓库不再保留根目录脚本或 `src` 根目录兼容文件。每项功能只有一份实现，统一通过 `python -m 完整模块路径` 调用。

## 接入、切分与追踪

这三个模块按“先确定输入，再决定片段，最后生成结构化轨迹”的顺序工作：

- `ingestion` 是输入边界。它检查文件是否存在、能否读取，使用视频探测工具读取时长、帧率和分辨率，并为源视频生成稳定编号。BARD 转换也属于接入，而不是模型推理。
- `segmentation` 把长视频变成事件模型能接受的短片。当前实现使用固定时长和重叠窗口，先生成包含原视频时间范围的清单，再按需调用 FFmpeg 导出真实片段。未来可以替换成镜头、比赛时钟或语义切分，而不改变应用层调用。
- `tracking` 对每个短片调用 SAM3，以球员和篮球提示生成掩码并沿时间传播，再把掩码转换为逐帧边界框 JSON。该模块还集中处理 TITAN RTX 的兼容注意力、视频与状态卸载等运行策略。

三者的职责不能混合：接入不理解比赛内容，切分不识别球员，追踪也不决定人物身份或事件类别。

## 事件识别与训练

`src/modules/event_recognition/` 只保留产品推理需要的内容：

- `playnet/`：PlayNet 网络层和模型组装；
- `inference.py`：TimeSformer 全局特征、人物 ROI/轨迹特征与 PlayNet 事件推理；
- `trajectory.py`：训练和推理共用的轨迹缩放；
- `labels.py`：事件标签定义。

训练专用的 `VideoBagClipsDataset`、`Solver` 和训练命令已经迁移到仓库根目录的 `training/`。这样产品部署无需依赖训练生命周期，模型训练也不会挤占业务模块。训练入口为：

```bash
python -m training.train --help
```

## 素材目录

`src/modules/catalog/` 当前实现一个不依赖数据库的最小业务闭环：

1. 把片段时间范围、Identity 结果和 PlayNet 预测组合成统一 `CatalogItem`；
2. 使用“球衣颜色 + 号码”生成人物检索键；
3. 支持按事件、人物和置信度查询；
4. 汇总事件数量、人物数量和平均置信度；
5. 保持素材检索与人物身份检索相互独立。

素材接口同时提供内存实现和 SQLite 实现。内存实现用于单元测试，产品运行使用 `src.modules.database.SQLiteMaterialCatalog`。

## 产品数据库

产品数据库与 BARD 数据集完全分开：BARD 只用于模型训练和方法验证；`product_data/` 只保存用户视频处理产生的产品数据。

```text
product_data/
├── database/basketevent.sqlite3
└── media/
    ├── uploads/
    ├── segments/
    └── visualizations/
```

SQLite 保存最小人物档案、素材、事件和素材人物关系。MP4 文件仍保存在媒体目录，数据库只记录路径。实际数据库和媒体均被 Git 忽略。

初始化或查看数据库状态：

```bash
python -m src.modules.database init
python -m src.modules.database status
```

数据库模块通过 `SQLiteParticipantRepository` 和 `SQLiteMaterialCatalog` 封装持久化操作。以后迁移 PostgreSQL 时替换仓库实现，不修改 PlayNet 或素材服务。

## Identity 模块

```text
src/modules/identity/
├── sampling.py          # 单视频轨迹取样
├── qwen_observer.py     # Qwen 逐帧属性观察
├── resolver.py          # 固定身份解析规则和可选名单查询
├── service.py           # 对外流程接口与命令入口
└── models.py            # 阶段之间传递的中间数据结构
```

Identity 阶段遵循以下边界：

- `TrackSampler` 只负责读取视频和轨迹、产生带原始帧号的截图，不复制短轨迹末帧。
- `QwenTrackObserver` 只描述每张图的场上人物、球衣颜色和号码，不作保留或删除决策。
- `IdentityResolver` 使用固定规则：唯一颜色号码组合为 `identified`，多个组合为 `conflicting`，没有完整组合为 `anonymous`。
- 三种结果都保留 SAM3 轨迹；确定身份使用“比赛 + 颜色 + 号码”作为编号，其他情况使用“比赛 + 片段 + 轨迹”匿名编号。
- 可选名单只补充姓名，不决定是否保留轨迹。
- `IdentityService` 按“取样 → Qwen 观察 → 规则解析”的顺序编排，不包含尚未落地的 ReID 或向量融合逻辑。

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
- `BASKETEVENT_PRODUCT_DATA_ROOT`
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
  --start-at identity
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
python -m src.modules.identity.service \
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
- 当前最小方案不包含 ReID 和跨片段外观归并；先以比赛内球衣颜色与号码生成检索编号。
- `conflicting` 轨迹已经能被发现并保留，但自动确定身份切换边界仍是下一步工作。
- 长视频切分目前是固定窗口基线，还没有使用比赛时钟、文字播报或镜头语义。
- 作者未公开原始训练视频与完整标签生成过程，作者 checkpoint 仅用于方法链路验证，不能代表在 BARD 上的最终准确率。
