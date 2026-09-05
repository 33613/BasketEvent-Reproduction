# BasketEvent 篮球视频素材理解系统

本项目以 BasketEvent 论文方法为基础，把篮球长视频处理成可查询的事件素材。当前目标是先跑通一条可复现、可恢复的最小产品链路：固定窗口只是模型输入，最终交付素材由事件结果重新导出。

## 当前可运行链路

```text
长视频
  → 读取视频元数据
  → 固定时长重叠切窗
  → 对每个窗口运行 SAM3 轨迹追踪
  → 不经过身份硬过滤，整理 PlayNet 输入轨迹
  → TimeSformer + PlayNet 识别球员级事件
  → 映射到长视频全局时间并合并重叠事件
  → 按事件时间范围导出最终素材
  → 默认登记“事件 → 素材”SQLite 索引
  → 可选：沿事件轨迹引用复用窗口 SAM3 bbox
  → 只为非 blank 事件主体抽帧并运行 Qwen 身份观察
  → 登记“人物 → 事件 → 素材”SQLite 索引
```

身份处理是事件识别后的可选增强。球衣号码没有识别出来时，事件素材仍然会保留并以事件级匿名人物登记，不会再次出现“Qwen 过滤掉轨迹后，PlayNet 看不到关键球员”的问题。身份阶段不重新运行 SAM3，也不观察同一窗口里的无关球员。

## 项目结构

```text
BasketEvent/
├── config/
│   ├── bard_team_colors.example.json
│   └── environment/                    # 服务器环境固化文件
├── product_data/                       # 正式产品数据库和媒体（不提交大文件）
├── sam3/                               # SAM3 Git 子模块
├── src/
│   ├── application/                    # 跨模块应用流程
│   │   ├── process_long_video.py       # 长视频调度、重试、续跑和最终入库
│   │   ├── evaluate_bard.py            # 独立 BARD 抽样、链路实验与评分
│   │   ├── process_clip.py             # 单窗口完整模型链路
│   │   ├── process_video.py            # 视频接入与切窗的可复用用例
│   │   ├── finalize_materials.py       # 最终素材导出和数据库登记
│   │   └── search_materials.py         # 产品素材查询
│   ├── core/
│   │   └── config.py                   # 默认路径和模型配置
│   └── modules/
│       ├── ingestion/                  # 用户视频与 BARD 数据接入
│       ├── segmentation/               # 固定重叠窗口规划和 FFmpeg 导出
│       ├── tracking/                   # SAM3 和事件输入轨迹准备
│       ├── event_recognition/          # PlayNet 推理和全局事件时间线
│       ├── materials/                  # 最终事件素材导出与可视化
│       ├── identity/                   # 事件轨迹取样、Qwen观察和规则解析
│       ├── catalog/                    # 数据库存储前的素材对象整理
│       ├── evaluation/                 # 论文核心指标，不参与推理决策
│       └── database/                   # SQLite 保存和查询
├── training/                           # Dataset、Solver 和训练入口
├── tests/
│   ├── long_video_runtime/             # 长视频端到端测试输入与运行产物
│   ├── bard_eval_runtime/              # 独立 BARD 测试包与评估说明
│   ├── qwen_tests_runtime/             # Qwen 诊断产物
│   └── track_segment_runtime/           # 人工轨迹片段验证产物
├── requirements.txt
└── README.md
```

## 独立测试与论文指标

目前已经跑通素材整理链路，但“运行完成”不等于识别可靠。新增 BARD 评估入口将抽样、人工标注、完整链路运行和评分分开；不依据 Qwen 或 SAM3 成功与否筛测试样本，GT 不进入推理。

评估分为两条：完整产品结果用 BasketballBench Q8 的顺序事件 F1、事件＋身份 F1 和条件身份正确率；人工核验轨迹后的模型结果用 BasketEvent 的人物—事件 Macro-F1、Recall@K 和最高 gate 时间段 Hit。推理报告增加完整类别概率和独立的最高 gate 证据，不改变现有素材生成逻辑。

这是自建 BARD 测试集上的**指标适配**，不是官方 benchmark 复现。数据划分、身份表示和协议差异在报告及说明中明确记录。

完整抽样、标注、校园网直传、服务器运行和评分命令见 [BARD 评估说明](tests/bard_eval_runtime/README.md)。

## 各模块现有逻辑

### application / process_long_video：长视频总调度

长视频处理的应用入口，依次调用视频接入、固定窗口切分、窗口模型处理、全局时间线、素材导出、身份处理和数据库登记。`job_state.json` 记录每个窗口的状态、重试次数和各阶段结果，SSH 中断后可以复用已有结果继续运行。

### ingestion：视频接入

读取输入视频路径与媒体元数据，并根据视频内容生成稳定的 `video_id`。源视频是后续窗口、全局时间线、最终素材和数据库记录的共同来源。BARD 转换工具也位于该模块，但 BARD 属于研究数据，不进入产品数据库。

### segmentation：固定重叠分析窗口

按照固定时长与重叠时间规划模型输入窗口，使用 FFmpeg 从源视频导出窗口 MP4，并写出 `segments.json` 窗口清单。每个窗口记录源视频中的全局起止时间和帧号。当前测试配置为 12 秒窗口、2 秒重叠；窗口只用于运行模型，不是最终交付素材。

### tracking：SAM3 轨迹追踪

`sam3_tracker.py` 在每个窗口上运行 SAM3，追踪画面中的球员和篮球，保存 `player_N`、`ball_N` 形式的逐帧 bbox 轨迹 JSON。窗口成功后，轨迹文件作为缓存供事件识别和后续身份观察共同复用。TITAN RTX 使用兼容注意力与 CPU 卸载策略。

### tracking / preparation：事件输入轨迹准备

`preparation.py` 检查轨迹结构，保留所有具有有效 bbox 的人物轨迹，选择篮球候选，并生成 PlayNet 所需的 clean 轨迹文件。此阶段不读取 Qwen 结果，也不以身份是否识别成功作为人物轨迹保留条件。

### event_recognition：球员级事件推理

`inference.py` 使用 TimeSformer 提取全局视频特征和人物局部特征，再由 PlayNet 对每条人物轨迹进行事件推理。窗口预测包含人物级事件、置信度和基于 MIL 权重的诊断性时间片段；`blank` 不进入全局事件素材。`playnet/` 保存网络层和模型组装，`labels.py` 保存事件标签定义。

### event_recognition / timeline：全局事件时间线

`timeline.py` 将窗口内事件起止时间加上窗口偏移，映射回原长视频时间；随后合并重叠窗口中的同类重复事件，形成 `event_timeline.json`，并根据事件类型增加素材前后文范围。当前事件边界来自 MIL 权重与窗口预测，属于诊断性时间定位。

### materials：最终事件素材导出

`exporter.py` 读取时间线中的 `material_drafts`，使用 FFmpeg 从原长视频重新剪出最终事件素材，并保存到 `final_materials/`。已经存在且非空的素材文件默认复用。

### identity：事件主体身份观察

身份阶段读取非 `blank` 事件的 `track_references`，定位已有窗口 MP4 和 SAM3 bbox，只对事件主体的唯一“窗口/人物轨迹”最多均匀抽取 10 帧。Qwen 逐帧记录球衣颜色、号码和置信度，固定规则形成 `identified`、`conflicting` 或 `anonymous` 结论。逐帧证据保存在 `event_identity_tracks/`，同一轨迹被多个事件引用时只处理一次。

Qwen 只提供可审计的身份信息，不决定轨迹是否进入 PlayNet。当前没有实现 ReID；跨窗口与跨素材暂时只按确定的球衣颜色和号码生成稳定人物编号。

### catalog：素材业务对象整理

将最终素材文件、事件标签、原视频事件时间、人物编号和身份状态整理为统一的 `CatalogItem` 业务对象，使应用层不依赖 SQLite 的具体表结构。

### database：产品数据保存与查询

使用 SQLite 保存产品数据，包括素材路径、原视频时间、事件类别、置信度、`participant_id`、球衣属性和事件—人物关系。MP4 仍保存在文件系统，数据库只保存元数据和路径，并支持按事件、人物以及两者组合查询素材。

研究数据与产品数据分离：

```text
BARD 数据集：/home/fangzilin/data/basket
产品或测试数据：product_data/ 或 tests/long_video_runtime/<video_id>/product_data
```

## 服务器环境

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

默认模型目录：

```text
/home/fangzilin/models/
├── sam3/sam3.pt
├── Qwen2.5-VL-7B-Instruct/
├── timesformer-base-finetuned-k400/
└── basketevent/playnet.pt
```

## 长视频端到端测试

所有长视频测试数据放在服务器项目的：

```text
/home/fangzilin/project/BasketEvent/tests/long_video_runtime/
```

建议把待测视频放在 `input/`：

```bash
cd /home/fangzilin/project/BasketEvent
mkdir -p tests/long_video_runtime/input
```

运行一条 3～10 分钟测试视频：

```bash
python -u -m src.application.process_long_video \
  tests/long_video_runtime/input/test_game.mp4 \
  --runtime-root tests/long_video_runtime \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --window-seconds 12 \
  --overlap-seconds 2 \
  --max-attempts-per-run 2 \
  --sam3-gpus 0,1 \
  --playnet-gpu 0
```

默认流程不运行身份识别。这样可以先验证长视频切窗、轨迹、事件、时间线、最终素材和 SQLite 登记。事件阶段完成后，重新运行相同命令并增加以下参数，即可复用全部成功窗口并只执行事件主体身份阶段：

```bash
  --with-identity --identity-gpus 0
```

`--identity-num-crops` 默认是10。`--identity-pad-ratio` 默认是0，不扩张SAM3人物框。

如果通过 `CUDA_VISIBLE_DEVICES=1` 只暴露物理 GPU1，进程内它会映射为逻辑 GPU0，因此仍应写 `--identity-gpus 0`：

```bash
export CUDA_VISIBLE_DEVICES=1

python -u -m src.application.process_long_video \
  tests/long_video_runtime/input/test_game.mp4 \
  --runtime-root tests/long_video_runtime \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --window-seconds 12 \
  --overlap-seconds 2 \
  --max-attempts-per-run 2 \
  --sam3-gpus 0 \
  --playnet-gpu 0 \
  --with-identity \
  --identity-gpus 0 \
  --identity-num-crops 10
```

名单不是必需输入；不提供名单时保存球衣颜色、号码或匿名身份，不映射真实姓名。

### 失败重试与断点续跑

- 每个失败窗口在本次运行中最多尝试 `--max-attempts-per-run` 次；
- SSH 中断后，重新运行完全相同的命令即可继续；
- 已成功窗口、已有最终素材和已有身份报告会被复用；
- 默认只要有窗口最终失败，任务就停止在时间线生成前；
- 只有明确接受不完整结果时才使用 `--allow-partial`；
- 已有任务的切窗或模型参数发生变化时，程序会拒绝混用旧结果。

如需放弃旧状态并从头运行，必须同时显式指定：

```bash
  --no-resume --overwrite-windows
```

### 运行产物

```text
tests/long_video_runtime/<video_id>/
├── job_state.json             # 窗口状态、尝试次数和错误信息
├── segments.json              # 固定窗口清单
├── windows/                   # 导出的模型输入窗口
├── window_artifacts/          # 每个窗口的 SAM3、PlayNet 和可视化结果
├── event_timeline.json        # 源视频全局事件时间线
├── final_materials/           # 重新从长视频导出的最终事件素材
├── event_identity.json        # 可选的事件主体身份汇总
├── event_identity_tracks/     # 唯一窗口轨迹的Qwen证据缓存
├── review_visualizations/     # 最终素材的人工复核标注视频
├── finalization_report.json   # 最终导出和入库报告
└── product_data/
    └── database/basketevent.sqlite3
```

`tests/long_video_runtime/` 中除说明文件外的输入和产物均被 Git 忽略。

### 生成人工复核视频

长视频任务完成后，可直接复用事件时间线、事件身份和已有 SAM3 轨迹，为
`final_materials` 中的每段素材生成复核版视频，不会重新运行 SAM3、Qwen 或
PlayNet。画面只框出事件主体，并显示事件类别、置信度、身份状态、球衣颜色与
号码、素材内时间、源长视频全局时间和事件证据时间条。

先用一条素材检查显示效果：

```bash
python -u -m src.application.visualize_final_materials \
  --job-root tests/long_video_runtime/<video_id> \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --limit 1
```

确认后生成全部复核视频；已有非空结果会自动复用：

```bash
python -u -m src.application.visualize_final_materials \
  --job-root tests/long_video_runtime/<video_id> \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg
```

输出位于 `review_visualizations/`，其中 `review_report.json` 记录每条素材、
事件、身份状态、轨迹引用和主体框覆盖帧数。需要重绘时增加 `--overwrite`。

## 单窗口诊断

保留单窗口入口用于 BARD 片段复现和问题定位：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --sam3-gpus 0,1 \
  --playnet-gpu 0
```

已有 SAM3 结果时，可从轨迹准备继续：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --start-at prepare
```

## 查询素材

长视频测试任务使用自己的数据库目录：

```bash
python -m src.application.search_materials \
  --database-root tests/long_video_runtime/<video_id>/product_data \
  --event "Made Shot" \
  --minimum-confidence 0.7
```

身份阶段完成后，可以按人物，或按“同一个人物完成的指定事件”查询：

```bash
python -m src.application.search_materials \
  --database-root tests/long_video_runtime/<video_id>/product_data \
  --participant-id "<video_id>:jersey:white:13"

python -m src.application.search_materials \
  --database-root tests/long_video_runtime/<video_id>/product_data \
  --participant-id "<video_id>:jersey:white:13" \
  --event "Made Shot"
```

## BARD、训练与测试

BARD 只用于方法验证与未来训练：

```bash
python -m src.modules.ingestion.bard.prepare --help
python -m src.modules.ingestion.bard.annotations_cli --help
python -m training.train --help
```

作者未公开原始训练视频和完整标签生成过程。作者 checkpoint 只能验证方法链路，不能代表在 BARD 或用户视频上的最终准确率。

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 当前已实现

- 长视频元数据读取与固定重叠窗口导出；
- 自动遍历全部窗口；
- 每窗口失败重试、状态记录和断点续跑；
- SAM3 轨迹生成及 TITAN RTX 兼容路径；
- 不依赖身份硬过滤的 PlayNet 轨迹准备；
- 球员级事件推理、全局时间映射和重叠事件合并；
- 根据事件时间重新导出最终素材；
- 默认登记事件素材，支持事件和置信度查询；
- 复用窗口SAM3轨迹、只观察事件主体的Qwen身份阶段；
- SQLite v2事件—人物直接绑定，支持人物、事件及组合查询。

## 当前边界

- 固定窗口还没有替换为智能切分；
- 事件边界来自模型的诊断性时间证据，尚未达到人工剪辑精度；
- 当前没有 ReID，跨素材只使用确定的球衣颜色和号码；
- TITAN RTX 不支持 Ampere Flash Attention，SAM3 使用兼容路径，速度较慢；
- 已完成一条150.7秒真实视频的15窗口事件链路测试；事件准确率和事件边界仍需后续优化；
- 事件主体身份新流程已完成代码和隔离单元测试，仍需在服务器GPU上验证Qwen输出与运行时间。
