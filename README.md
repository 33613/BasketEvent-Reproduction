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
│       └── database/                   # SQLite 保存和查询
├── training/                           # Dataset、Solver 和训练入口
├── tests/
│   ├── long_video_runtime/             # 长视频端到端测试输入与运行产物
│   ├── qwen_tests_runtime/             # Qwen 诊断产物
│   └── track_segment_runtime/           # 人工轨迹片段验证产物
├── requirements.txt
└── README.md
```

## 主要模块

### ingestion：视频接入

检查输入文件，读取时长、帧率、分辨率和帧数，并为同一份输入生成稳定的视频编号。BARD 转换工具也位于此模块，但 BARD 是研究数据，不进入产品数据库。

### segmentation：模型输入切窗

当前按固定时长切出带重叠的窗口。每个窗口记录源视频中的全局起止时间和帧号。窗口仅用于运行模型，不是最终交付给用户的素材。

### tracking：轨迹处理

- `sam3_tracker.py` 调用 SAM3 生成球员和篮球逐帧边界框；
- `preparation.py` 校验轨迹、保留所有具有有效边界框的人物并选择篮球候选；
- TITAN RTX 使用兼容注意力和 CPU 卸载策略。

事件识别前不读取 Qwen 的人物判断。

### event_recognition：事件识别与时间线

- `playnet/`：PlayNet 网络层和模型组装；
- `inference.py`：TimeSformer 全局特征、人物局部特征和事件推理；
- `timeline.py`：把窗口内时间映射回源视频、合并重叠窗口的重复事件，并生成待剪范围；
- `labels.py`：事件标签定义。

当前事件时间来自 MIL 权重与窗口预测，是产品原型中的诊断性定位，不是人工标注的精确动作边界。

### materials：最终素材

`exporter.py` 根据全局事件时间线，使用 FFmpeg 从原长视频导出最终事件素材。导出的现有非空文件默认复用。

### identity：可选身份增强

```text
读取非 blank 事件的 track_references
  → 定位已有窗口 MP4 和 SAM3 bbox JSON
  → 只对事件主体轨迹最多均匀抽取 10 帧
  → Qwen 逐帧观察球衣颜色和号码
  → 固定规则解析 identified / conflicting / anonymous
  → 把 participant_id 直接绑定到事件和最终素材
```

同一个“窗口/人物轨迹”被多个事件引用时只观察一次，逐帧证据保存在 `event_identity_tracks/`，中断后可以复用。Qwen 只提供可审计证据，不决定事件轨迹是否进入 PlayNet。当前没有实现 ReID；跨素材只按确定的球衣颜色和号码生成稳定人物编号。

### catalog 与 database：产品登记

`catalog` 把素材文件、事件和可选人物整理成统一业务对象；`database` 使用 SQLite 保存素材、事件、人物及其关系。MP4 保存在文件系统，SQLite 只保存元数据和路径。

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
├── finalization_report.json   # 最终导出和入库报告
└── product_data/
    └── database/basketevent.sqlite3
```

`tests/long_video_runtime/` 中除说明文件外的输入和产物均被 Git 忽略。

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
