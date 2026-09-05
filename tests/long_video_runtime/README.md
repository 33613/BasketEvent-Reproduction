# 长视频运行测试目录

这个目录只存放长视频端到端测试产生的数据，包括：

- 待测试的 3～10 分钟输入视频；
- 固定重叠窗口；
- SAM3 轨迹和 PlayNet 预测；
- 全局事件时间线；
- 最终事件素材和身份报告；
- 测试使用的 SQLite 数据库；
- `job_state.json` 断点续跑状态。

除本说明外，目录内所有运行数据都不会提交到 Git。

建议把输入视频放在：

```text
tests/long_video_runtime/input/
```

每次任务的输出会按稳定视频编号保存在：

```text
tests/long_video_runtime/{video_id}/
```
