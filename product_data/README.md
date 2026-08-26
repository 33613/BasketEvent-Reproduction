# 产品运行数据目录

该目录只保存 BasketEvent 产品运行时的数据，不属于 BARD 训练数据集。

程序初始化后会创建：

```text
product_data/
├── database/
│   └── basketevent.sqlite3    # 人物库与素材库
└── media/
    ├── uploads/               # 用户上传的原始视频
    ├── segments/              # 从长视频切出的片段
    └── visualizations/        # 可视化和产品导出文件
```

除本说明外，目录中的数据库、视频和生成文件均被 Git 忽略。数据库保存结构化元数据和媒体路径，不把 MP4 文件本身写入 SQLite。

默认服务器位置为：

```text
/home/fangzilin/project/BasketEvent/product_data
```

可通过环境变量 `BASKETEVENT_PRODUCT_DATA_ROOT` 修改。
