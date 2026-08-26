"""提供轨迹取样、Qwen 观察、身份解析和跨片段聚类。

各类应从其所属子模块显式导入。这里不提前导入模型依赖，避免运行
``python -m src.modules.identity.resolver`` 时重复加载命令模块。
"""
