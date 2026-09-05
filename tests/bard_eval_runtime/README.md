# BARD 独立评估实验

## 要回答的问题

产品目标不是篮球知识问答，而是：输入视频，输出可按“事件、球员”查找的、有时间来源的素材。
一次全链路运行完成，只说明软件能工作；下面的独立评估才检验输出是否可靠。

先分三层，不用一个“准确率”概括全部：

1. **模型层**：人工核验目标轨迹后，PlayNet 是否识别正确事件、找到时间证据？
2. **系统层**：不人工修正预测，让追踪、事件、合并、身份的完整链路输出事件列表，再评分。
3. **素材层**：人工检查导出片段有没有包含完整动作、是否冗余、适不适合使用。论文的事件指标不能代替素材可用性。

## 论文关系与评估口径

[BasketEvent 第 5.2 节](https://arxiv.org/html/2607.21267v1#S5.SS2)针对人物—事件识别和粗时间证据。
[Towards Comprehensive Basketball Understanding](https://arxiv.org/html/2608.23435v1)把篮球理解扩展成多任务问答；其中 Q8 最接近我们的“谁做了什么”输出。

| 指标 | 本项目实现与用途 | 限制 |
| --- | --- | --- |
| Q8 Event-type F1 | 按事件顺序做 LCS；跨视频累计匹配数后计算 micro P/R/F1 | 不检查球员身份，也不检查精确时间 |
| Q8 Full-event F1 | LCS 同时要求类别、身份相同 | 当前使用比赛内颜色＋号码，论文使用球队＋号码，是适配版 |
| Q8 Participant Accuracy | Type-only LCS 对齐后，身份正确的比例 | 条件指标；漏检不直接进入 PA 分母，不能单独汇报 |
| BasketEvent Macro-F1 | 每视频人物—事件集合，逐类 F1 再平均 | `score-tracks` 使用人工轨迹编号；`score` 使用自动颜色号码，二者不可混报 |
| BasketEvent Recall@1/3/5 | 每视频全体人物的非背景类别概率组成排名，计算逐类召回并宏平均 | 排名粒度是本实现的明确约定；尚未通过作者官方评估代码逐项对齐 |
| BasketEvent Hit@0.3/0.5 | 人工目标对应轨迹的最高 gate 片段，类别正确且 mIoU 严格大于阈值 | 必须人工标事件起止；不能用扩展素材边界或挑最吻合 GT 的候选 |

公式：`P=M/Npred`，`R=M/Ngt`，`F1=2M/(Npred+Ngt)`。LCS 是最长公共子序列，事件错序、重复、漏检都会影响匹配。
Type-only LCS 的平局固定先跳过预测项，不能拿身份正确与否打破平局；论文没有公布此实现细节。

论文时间 `mIoU=交集/较短区间长度`，不是常规 `IoU=交集/并集`。例如预测 `[0,10]`、GT `[4,5]`，前者为 1，后者仅 0.1。我们的模型层报告保留二者，Hit 按论文 mIoU 计算。

新论文 Q7 是**已知查询事件的单个时刻**评估（误差不超过 1 秒），并非事件检测或素材起止边界。本链路没有该任务的独立时刻预测器，所以不把 MIL 中点包装成 Q7 复现分数。
同样不计算篮球知识、比分 OCR、投篮区域等尚未实现的任务。

**这不是作者官方测试集复现。** 当前官方 checkpoint 的训练比赛与本地 BARD 比赛交集尚未核清；数据、采样与身份键均有差别。
公式实现相同，只允许说“在自建 BARD 测试集上采用论文指标评估”，不能据此宣布达到或未达到作者基准。
严格复现还需要作者测试划分、轨迹—事件 GT、时间 GT、原始推理设置及完整排名/平局约定。

## 文件和职责

- `src/modules/evaluation/metrics.py`：纯指标计算，不加载模型、不读取数据库。
- `src/application/evaluate_bard.py`：抽样、媒体校验、调用现有流水线、适配输出并评分。
- `src/modules/event_recognition/inference.py`：额外保存 `class_probabilities` 和 `paper_gate_segment`，不改变产品推理决策。
- `tests/test_evaluation.py`：CPU 回归测试。
- 本目录下每个 `pilot_*` 是独立测试包，每个 `run_*` 是一次协议固定的运行目录。数据不提交 Git。

```text
pilot_v1/
├── manifest.json          # 冻结抽样、源路径、SHA256、类别覆盖
├── annotations.json       # 人工核验的全部事件、身份、顺序，可选起止时间
├── sources/               # 原始 BARD action JSON，只用于标注
└── videos/                # 可传输的 MP4
run_pilot_v1/
├── evaluation_config.json # 协议、代码 commit、checkpoint 摘要
├── run_report.json        # 每条视频的状态、命令和 job_root
├── <sample_id>/           # pipeline.log 及完整产品运行产物
└── scores.json            # 总分、逐类计数、逐样本对齐、失败列表
```

## 第一步：构建小型先导集（本机 PowerShell）

在仓库根目录执行：

```powershell
python -m src.application.evaluate_bard build `
  --data-root "D:\数据集\basket" `
  --output tests/bard_eval_runtime/pilot_v1 `
  --game-count 6 --per-class 3 --seed 20260905
```

默认排除反复调试的 `bkn-vs-det-0022400861` 整场比赛。不根据 Qwen 成功、SAM3 质量或最终预测筛样本。
先随机选比赛，再优先补齐稀有事件；同一片段可含多类事件，**每类 3 条不是共 30 条**。
源描述中无法映射的内容会留在候选统计/标注警告里，不自动变为 blank。
这是按类别覆盖抽取的先导集，不是对真实长视频事件频率的无偏估计。

本机首次实际抽样得到 **6 场比赛、13 条视频、88.6 MiB**，覆盖 9 类，未抽到 Jump Ball。
具体比赛/视频和源标签分布见冻结清单；这不是经过人工确认的类别分布。
因此这个先导集不能代表完整十类基准，固定十类 Macro-F1 和有 GT 支持类别的 Macro-F1 会分别报告。

检查清单后复制媒体并校验（已存在且摘要一致的媒体会跳过）：

```powershell
python -m src.application.evaluate_bard copy --bundle tests/bard_eval_runtime/pilot_v1
python -m src.application.evaluate_bard verify --bundle tests/bard_eval_runtime/pilot_v1
```

`build` 拒绝覆盖旧目录。扩大正式测试集时用新名称和新的比赛列表 `--games GAME1 GAME2 ...`，建议最终每类至少 20～30 条有人工 GT 的样本；正式测试比赛不要与先导调参比赛重合。稀有事件不足必须明确披露，不能复制样本充数。
另增加真实无事件、回放和遮挡场景作为产品负例集；BARD 事件中心片段不能代表整场录像里的这些场景。

## 第二步：人工核验 GT（不看模型预测）

打开 `videos` 中原片段，修改 `annotations.json`：

1. 核验所有可见事件，不只确认 BARD 的主动作；补漏标、删掉片外描述。
2. 按真实发生顺序排列 `events`，例如助攻传球通常早于进球。不能直接相信草稿顺序。
3. `actor` 使用 `white#13` 等明确颜色号码；`0` 和 `00` 不同。跳球两名参与人需按字典序写为 `black#0|white#8`；目前系统没有完整双人跳球输出，这会如实暴露为身份不匹配。
4. 标注者无法确认身份时不要猜。暂不设 `reviewed`，扩大画面或请第二人复核；若最终不可判读，另制定并披露不可判读子集协议，不静默删除样本。
5. 可选填写 `start_seconds/end_seconds`：相对当前原片段，不是比赛计时。做时间评估时再统一定义动作起止规则。
6. 确认全片后填写 `reviewer`，设 `reviewed: true`。没有事件的真实负例使用 `events: []`。

草稿示例（修改顺序并核验之后才能改为 true）：

```json
{
  "sample_id": "比赛名__片段号",
  "reviewed": false,
  "reviewer": "",
  "events": [
    {"event": "ast", "actor": "white#20", "start_seconds": null, "end_seconds": null},
    {"event": "Made Shot", "actor": "white#13", "start_seconds": null, "end_seconds": null}
  ]
}
```

不再调用训练用 `annotations_cli labels` 构建评测 GT：它依赖 Qwen-clean 轨迹并按 Scheme A 排除多标签人物，会产生筛选偏差。

## 第三步：直连传输（本机 PowerShell）

在本机仓库根目录执行；校园网直连，不使用旧跳板机别名：

```powershell
$Server = "fangzilin@10.195.234.58"
$RemoteRoot = "/home/fangzilin/project/BasketEvent/tests/bard_eval_runtime"
ssh $Server "mkdir -p $RemoteRoot"
tar -cf tests/bard_eval_runtime/pilot_v1.tar -C tests/bard_eval_runtime pilot_v1
scp tests/bard_eval_runtime/pilot_v1.tar "${Server}:${RemoteRoot}/"
ssh $Server "tar -xf $RemoteRoot/pilot_v1.tar -C $RemoteRoot"
```

如果无法直连，先用 `Test-NetConnection 10.195.234.58 -Port 22` 检查校园网络连通性；不要自行启用旧跳板配置。
再次传输同名包前确认服务器内人工标注/输入没有新的修改，以免覆盖。更改已冻结测试集应使用新版本目录。

## 第四步：服务器先跑一条，再整批续跑

```bash
cd /home/fangzilin/project/BasketEvent
git pull --ff-only origin server/titan-bootstrap
source /home/fangzilin/tools/miniconda3/etc/profile.d/conda.sh
conda activate /home/fangzilin/envs/basketevent
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m unittest tests.test_evaluation tests.test_inference_timeline -v
python -m src.application.evaluate_bard verify --bundle tests/bard_eval_runtime/pilot_v1

python -u -m src.application.evaluate_bard run \
  --bundle tests/bard_eval_runtime/pilot_v1 \
  --run-root tests/bard_eval_runtime/run_pilot_v1 \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --sam3-gpus 0,1 --playnet-gpu 0 --identity-gpus 0 --limit 1
```

此命令要求两张物理卡都空闲：SAM3 用逻辑 `0,1` 分担轨迹，PlayNet 和 Qwen 身份阶段依次使用逻辑 GPU 0；各阶段并非同时常驻。若只开放物理 GPU1，可改回 `CUDA_VISIBLE_DEVICES=1` 和兼容参数 `--gpu 0`。主入口自动读取每个输入视频的 FPS，并显式传给 PlayNet，避免把 30 FPS 视频当成 60 FPS。
当前接口只接受整数帧率；遇到 29.97 等输入会报错，需单独准备并记录 CFR 版本，不能悄悄改写原测试数据。

先看第一条 `pipeline.log`、`job_state.json` 和实际导出素材；确认工程链路没问题，再去掉 `--limit 1`：

```bash
nohup python -u -m src.application.evaluate_bard run \
  --bundle tests/bard_eval_runtime/pilot_v1 \
  --run-root tests/bard_eval_runtime/run_pilot_v1 \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --sam3-gpus 0,1 --playnet-gpu 0 --identity-gpus 0 \
  > tests/bard_eval_runtime/pilot_v1.log 2>&1 &
echo $! > tests/bard_eval_runtime/pilot_v1.pid
```

同一命令重跑会让现有调度器复用缓存、重试失败；按顺序执行，不并发争抢两张卡。
每个视频独立完成切窗、SAM3、PlayNet、时间线、素材、Qwen、SQLite。**运行不读取 annotations.json 或 sources。**
`--pipeline-mode clip` 将分析窗口设为 3600 秒（BARD 短片整体作为单窗口），用于与 `product` 的 12 秒重叠切窗做对照；必须使用另一个 `run-root`。它仍不是作者完整原版流程。
代码 commit、PlayNet 权重或协议变化时拒绝复用旧实验目录，另建目录保存对照；其他模型记录路径而非完整目录摘要，目录内容不要原地替换。

## 第五步：评分

人工标注完成后再执行：

```bash
python -m src.application.evaluate_bard score \
  --bundle tests/bard_eval_runtime/pilot_v1 \
  --run-root tests/bard_eval_runtime/run_pilot_v1
```

所有清单样本进入分母，未运行或失败任务按空预测计入；身份 unknown/conflicting 不过滤事件，也不能与 GT 身份匹配。
失败项同时单独列出，便于区别工程失败和模型错误。Q8 对预测按模型时间证据中点排序，不按 GT 重排。
报告包含规范化预测与 GT 以及 LCS 对齐索引，可以逐条复核。
只有想测试评分脚本能否运行时才加 `--allow-draft`，输出名为 `scores_draft.json`，不能用来评价模型。

## 可选：原论文模型层指标

手工核验每个分析窗口的 SAM3 轨迹，准备 `track_targets.json`。不要用 Qwen 预测直接填 GT。示例：

```json
{
  "reviewed": true,
  "reviewer": "标注者",
  "samples": [{
    "sample_id": "窗口唯一编号",
    "prediction_json": "相对本文件的窗口_events.json路径",
    "references": [
      {"actor": "player_3", "event": "Made Shot", "interval": [3.1, 4.2]},
      {"actor": "missing_0", "event": "ast", "interval": [1.8, 2.8]}
    ]
  }]
}
```

`interval` 相对窗口；未追踪到的 GT 球员用独立 `missing_N`，计入漏检；混合身份轨迹需标记并单独审查，不能强行当成干净的 oracle。
每窗口补齐真实参与事件的人物；没事件的人物不写入 references，其误报仍计 FP。
新推理 JSON 含全部类别概率及最高 gate 片段；旧缓存没有这两个字段时脚本拒绝给出伪造的排名/时间分数，须只重跑对应 PlayNet 推理。

```bash
python -m src.application.evaluate_bard score-tracks \
  --targets tests/bard_eval_runtime/track_targets.json \
  --output tests/bard_eval_runtime/track_scores.json
```

这层回答“已有轨迹的事件推理能否工作”，不是完整身份识别性能。原论文 Recall 的排名粒度仍需与官方实现确认。

## 得分后再选下一步

- 模型层好、系统层差：优先检查轨迹身份切换、身份绑定和窗口合并，不急着训练新网络。
- 类型 F1 好、Full-event F1 差：先处理事件主体与身份关联；比较更好的抽帧、OCR、规则或其他身份检索方案。
- 人工正确轨迹上仍频繁错事件：优先检查 FPS、模型输入、标签映射和数据分布，再比较其他事件方法。
- 事件对、素材不好用：单独改时间定位和剪辑边界；不要期待 Qwen 号码识别解决时间问题。
- 工程失败率高：先消除运行失败，否则模型比较会混入资源和环境因素。

先导集用于排查协议和选择方向。方向选定后，在未参与调参的新比赛上冻结正式测试集，报告样本量、逐类支持数和比赛级置信区间；小样本暂不下“达到 90%”结论。

## 两场完整比赛实验

先导集只验证协议。当前完整实验固定为：

- `bkn-vs-det-0022400861`：246 条，已经用于多轮问题定位，只作为回归场，不能冒充未见测试集。
- `okc-vs-mia-0022400375`：232 条，约 1.43 GiB，未参与已有调试，作为主要留出场。

两场共 478 条、约 2990.1 MiB。BARD 草稿的片段覆盖为：Missed Shot 167、Made Shot 140、Free Throw 96、Foul 86、Turnover 62、Rebound 152、steal 31、block 24、ast 92、Jump Ball 0。一个片段可覆盖多类，以上数字不能相加当作视频数，也尚不是人工核验后的事件实例数。

本地 60 场 BARD 的结构化动作均没有映射出 Jump Ball，因此本实验对该类没有评估能力，指标报告会同时给固定十类 Macro-F1 与仅 GT 支持类别的 Macro-F1。

先把新增比赛上传到服务器数据区。Windows PowerShell：

```powershell
$Game = "okc-vs-mia-0022400375"
$DataRoot = "D:\数据集\basket"
$Archive = "$DataRoot\$Game.tar"
$Server = "fangzilin@10.195.234.58"
$RemoteDataRoot = "/home/fangzilin/data/basket"

tar -cf $Archive -C $DataRoot $Game
Get-FileHash -Algorithm SHA256 $Archive
ssh $Server "mkdir -p $RemoteDataRoot"
scp $Archive "${Server}:${RemoteDataRoot}/"
```

服务器先比较 `sha256sum /home/fangzilin/data/basket/okc-vs-mia-0022400375.tar` 与 PowerShell 输出，再解包：

```bash
tar -xf /home/fangzilin/data/basket/okc-vs-mia-0022400375.tar \
  -C /home/fangzilin/data/basket

find /home/fangzilin/data/basket/okc-vs-mia-0022400375/video \
  -maxdepth 1 -type f -name '*.mp4' | wc -l
```

预期为 232。确认无误后再决定是否保留传输归档；不要在校验前删除。

在服务器从两场原始数据构建冻结测试包。`--all-clips` 必须搭配显式 `--games`，防止误打包全部 60 场：

```bash
cd /home/fangzilin/project/BasketEvent

python -m src.application.evaluate_bard build \
  --data-root /home/fangzilin/data/basket \
  --output tests/bard_eval_runtime/two_games_full_v1 \
  --games bkn-vs-det-0022400861 okc-vs-mia-0022400375 \
  --all-clips

python -m src.application.evaluate_bard copy \
  --bundle tests/bard_eval_runtime/two_games_full_v1 \
  --method hardlink

python -m src.application.evaluate_bard verify \
  --bundle tests/bard_eval_runtime/two_games_full_v1
```

硬链接复用 `/home/fangzilin/data/basket` 的视频内容，不额外占用约 2.9 GiB；若两目录不在同一文件系统，命令会明确失败，此时改用 `--method copy` 并先确认磁盘空间。

当前单条双 GPU 试运行成功后，再启动 478 条完整实验：

```bash
nohup python -u -m src.application.evaluate_bard run \
  --bundle tests/bard_eval_runtime/two_games_full_v1 \
  --run-root tests/bard_eval_runtime/run_two_games_full_v1 \
  --ffmpeg-binary /home/fangzilin/tools/ffmpeg-full/bin/ffmpeg \
  --sam3-gpus 0,1 --playnet-gpu 0 --identity-gpus 0 \
  > tests/bard_eval_runtime/two_games_full_v1.log 2>&1 &

echo $! > tests/bard_eval_runtime/two_games_full_v1.pid
```

先用一条成功任务的实际耗时估算 478 条总时长；不要未经估算就让共享 GPU 连续运行数天。运行报告和最终得分按两场分别输出，同时给总体结果。

对全部 478 条，可先使用 BARD 结构化动作进行 **silver label** 无序人物—事件集合评估：

```bash
python -m src.application.evaluate_bard score \
  --bundle tests/bard_eval_runtime/two_games_full_v1 \
  --run-root tests/bard_eval_runtime/run_two_games_full_v1 \
  --accept-bard-silver
```

该模式输出 `scores_silver.json`，明确不计算 Q8 顺序 LCS：BARD 草稿未经过逐片核验，而且自动加入的助攻顺序不能当作严格视频事件顺序。要报告 Q8 Event-type/Full-event F1，仍需人工核验对应样本。推荐从新比赛再冻结分层人工 gold 子集，而不是人工标完 478 条后才发现协议问题。
