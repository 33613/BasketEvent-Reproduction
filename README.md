# BasketEvent 篮球视频素材理解系统

本仓库以 BasketEvent 论文方法为基础，目标是把一场篮球比赛长视频自动处理成可检索、可分类的事件素材。项目当前关注完整产品链路，而不是孤立优化某一个模型：模型识别失败必须被记录，但不能阻塞素材生成。

## 当前工作流

```text
用户比赛视频
  → 视频接入与元数据检查
  → 固定重叠窗口（保留源视频全局时间）
  → SAM3 球员与篮球轨迹
  → 轨迹结构准备（不按身份过滤）
  → TimeSformer + PlayNet 事件识别
  → 局部事件映射到源视频时间轴
  → 重叠窗口事件消重
  → 按事件类型生成待剪范围
  → FFmpeg 导出事件素材
  → 对事件素材执行身份观察与规则解析
  → 跨素材人物归并（只使用确定证据）
  → SQLite 登记、检索与统计
```

工作流分成两个弱耦合阶段：

1. **事件发现**回答“什么时候发生了什么”，不依赖球衣号码是否可读；
2. **素材增强**再回答“素材里有谁”，身份无法确定时保留匿名人物。

因此 Qwen、号码识别或后续 ReID 都不会成为事件识别前的硬过滤器。

## 项目结构

```text
BasketEvent/
├── config/
│   ├── bard_team_colors.example.json
│   └── environment/                  # 服务器环境固化文件
├── product_data/                     # 产品数据库与用户媒体，内容不提交
├── sam3/                             # SAM3 Git 子模块
├── src/
│   ├── application/                  # 编排模块，不实现模型和 SQL 细节
│   │   ├── process_video.py           # 长视频接入与固定窗口规划
│   │   ├── process_clip.py            # 单窗口追踪和事件识别
│   │   ├── finalize_materials.py       # 导出、归并并登记最终素材
│   │   └── search_materials.py         # 查询产品素材
│   ├── core/
│   │   └── config.py                  # 路径和运行配置
│   └── modules/
│       ├── ingestion/                 # 视频与 BARD 数据接入
│       ├── segmentation/              # 固定重叠窗口规划和导出
│       ├── tracking/                  # SAM3 追踪与模型输入轨迹准备
│       ├── event_recognition/         # PlayNet 推理与全局事件时间线
│       ├── materials/                 # 最终素材导出和视频可视化
│       ├── identity/                  # 身份观察、解析与跨素材归并
│       ├── catalog/                   # 素材对象整理与统计
│       └── database/                  # SQLite 保存与查询
├── training/                          # Dataset、Solver 和训练入口
├── tests/                             # 单元测试和受控诊断脚本
├── requirements.txt
└── README.md
```

根目录不再保留旧入口或 `src` 根目录兼容副本。每项功能只有一份实现，统一通过 `python -m 完整模块路径` 调用。

## 模块职责

### 1. ingestion：接收视频

检查输入文件并读取时长、帧率、分辨率、帧数等元数据，为一次用户上传生成稳定视频编号。BARD 转换工具也放在这里，但 BARD 只用于模型验证，不属于产品数据库。

### 2. segmentation：生成分析窗口

当前使用固定时长、带重叠的窗口。每个 `VideoSegment` 保存：

- 源视频编号；
- 全局起止秒数；
- 全局起止帧号；
- 导出文件名。

固定窗口是事件模型的分析单位，不是最终交付给用户的素材。

### 3. tracking：生成轨迹

- `sam3_tracker.py` 调用 SAM3 生成球员和篮球逐帧边界框；
- `preparation.py` 检查轨迹结构、保留所有有效人物并选择篮球候选；
- TITAN RTX 使用兼容注意力和 CPU 卸载策略。

`preparation.py` 不读取 Qwen 结论。只要人物轨迹具有有效边界框，就会进入 PlayNet。

### 4. event_recognition：事件发现与全局时间线

- `playnet/`：PlayNet 网络层和模型组装；
- `inference.py`：TimeSformer 全局特征、人物 ROI/轨迹特征与事件推理；
- `trajectory.py`：训练和推理共用的轨迹缩放；
- `labels.py`：事件标签定义；
- `timeline.py`：局部时间映射、重叠窗口消重和待剪范围生成。

时间线包含三层结果：

- `candidates`：每个固定窗口中的原始时间证据；
- `events`：映射到源视频并完成窗口消重的事件；
- `material_drafts`：按事件类型增加前后文后的待剪范围。

当前时间定位来自 MIL 权重，只用于第一版产品剪辑，不等同于动作级人工真值。

### 5. materials：导出与可视化

- `exporter.py` 使用 FFmpeg 从源视频导出 `material_drafts`；
- `visualization.py` 绘制 SAM3 轨迹、模型输入轨迹和事件时间线。

导出的文件名同时兼容 Windows 和 Linux。已有非空素材默认复用。

### 6. identity：素材后处理身份

```text
identity/
├── sampling.py          # 从一条轨迹选择不同时间位置的截图
├── qwen_observer.py     # Qwen 逐帧观察球衣颜色和号码
├── resolver.py          # 固定规则生成 identified/conflicting/anonymous
├── association.py       # 跨素材保守人物归并
├── service.py           # 单素材身份处理入口
└── models.py            # 阶段间数据结构
```

身份规则：

- Qwen 只提供可审计观察，不决定轨迹是否保留；
- 唯一完整颜色号码组合为 `identified`；
- 多个组合为 `conflicting`；
- 没有完整组合为 `anonymous`；
- 名单只补充姓名，不决定是否保留；
- 跨素材仅合并 `identified` 且颜色号码完整的结果；
- 匿名和冲突人物始终保持素材隔离，避免错误合并。

当前没有实现 ReID。以后可以增加外观向量作为新证据，但不能改变上述保守回退规则。

### 7. catalog：整理素材对象

`CatalogService` 不连接数据库：

- `build_final_material()` 把最终视频路径、全局事件和可选人物组成 `CatalogItem`；
- `build_material()` 保留给旧的单片段实验；
- `summarize_reports()` 汇总事件、人物和置信度。

### 8. database：保存产品数据

产品数据和 BARD 数据集完全分开：

```text
product_data/
├── database/basketevent.sqlite3
└── media/
    ├── uploads/
    ├── segments/
    └── visualizations/
```

SQLite 保存人物、素材、事件和素材人物关系；MP4 仍保存在文件系统，数据库只记录路径。支持按事件、人物和最低置信度组合查询。

## 环境准备

服务器当前验证环境：

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

常用环境变量：

- `BASKETEVENT_DATA_ROOT`
- `BASKETEVENT_ARTIFACTS_ROOT`
- `BASKETEVENT_RUNTIME_ROOT`
- `BASKETEVENT_PRODUCT_DATA_ROOT`
- `BASKETEVENT_MODEL_ROOT`
- `BASKETEVENT_GPU_IDS`
- `BASKETEVENT_HF_LOCAL_FILES_ONLY`

## 运行方式

### 规划长视频固定窗口

```bash
python -m src.application.process_video INPUT.mp4 \
  --output-root /home/fangzilin/data/video_jobs \
  --window-seconds 12 \
  --overlap-seconds 2 \
  --export-clips
```

### 处理一个固定窗口

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --sam3-gpus 0,1 \
  --playnet-gpu 0
```

已有 SAM3 结果时从轨迹准备继续：

```bash
python -m src.application.process_clip \
  --game bkn-vs-det-0022400861 \
  --clip 100 \
  --start-at prepare
```

### 单独处理素材身份

```bash
python -m src.modules.identity.service \
  --video_path /path/to/material.mp4 \
  --bbox_json_path /path/to/material_raw_tracks.json \
  --json_save_path /path/to/material_identity_tracks.json \
  --qwen_model /home/fangzilin/models/Qwen2.5-VL-7B-Instruct \
  --gpus_to_use 0
```

身份服务需要该最终素材自身的 SAM3 轨迹。名单可选；不提供名单时不映射真实姓名。

### 导出并登记最终素材

```bash
python -m src.application.finalize_materials \
  --source-video-id GAME_OR_UPLOAD_ID \
  --source-video /path/to/full_game.mp4 \
  --timeline-json /path/to/event_timeline.json \
  --output-directory product_data/media/segments/GAME_OR_UPLOAD_ID \
  --database-root product_data \
  --identity-index-json /path/to/identity-index.json \
  --report-json /path/to/finalization_report.json
```

身份索引是可选文件。没有它时，事件素材仍会被导出并登记：

```json
{
  "GAME_OR_UPLOAD_ID:material_00000": "identity/material_00000_identity.json",
  "GAME_OR_UPLOAD_ID:material_00001": "identity/material_00001_identity.json"
}
```

### 查询素材

```bash
python -m src.application.search_materials \
  --database-root product_data \
  --event "Made Shot" \
  --minimum-confidence 0.7
```

也可以增加 `--participant-id` 查询同一确定人物出现的素材。

### 初始化或查看数据库

```bash
python -m src.modules.database init
python -m src.modules.database status
```

## BARD 和训练工具

BARD 只用于方法验证与未来训练，不进入产品数据库：

```bash
python -m src.modules.ingestion.bard.prepare --help
python -m src.modules.ingestion.bard.annotations_cli --help
python -m training.train --help
```

作者未公开原始训练视频与完整标签生成过程。作者 checkpoint 只能验证方法链路，不能代表在 BARD 或用户视频上的最终准确率。

## 测试

```bash
python -m unittest discover -s tests -v
```

`tests/qwen_tests_runtime/` 和 `tests/track_segment_runtime/` 只约定运行时诊断目录，大文件不会提交。

## 当前完成度

- [x] 长视频元数据读取与固定重叠窗口规划
- [x] SAM3 轨迹生成与 TITAN RTX 兼容路径
- [x] 不依赖身份过滤的 PlayNet 输入准备
- [x] 球员级事件推理与诊断性时间证据
- [x] 源视频全局时间映射与重叠窗口事件消重
- [x] 事件类型上下文规则与最终素材导出
- [x] 单素材身份观察、固定规则解析和保守跨素材归并
- [x] SQLite 素材登记、事件/人物组合检索与统计
- [ ] 自动遍历整场比赛的全部窗口并管理失败重试
- [ ] 最终素材上的 SAM3 与身份处理批量调度
- [ ] 更精确的事件边界和 ReID/外观证据实验

## 当前限制

- TITAN RTX 不支持 Ampere Flash Attention，SAM3 使用兼容注意力路径，速度较慢；
- 固定窗口不是最终素材，最终素材由全局事件时间线重新生成；
- `conflicting` 轨迹会被保留，但身份切换边界仍需后续处理；
- 现有模块已覆盖完整数据流，但尚未提供自动循环全部窗口的作业调度器；
- 最终素材的 SAM3 和身份处理目前调用已有命令，尚未自动批量执行；
- 时间边界来自模型诊断证据，需要通过更多比赛评估剪辑效果。
