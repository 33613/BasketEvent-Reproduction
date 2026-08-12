# Local reproduction metadata

此目录用于固化 BasketEvent 的本地复现环境，可以随代码提交；不保存模型权重、数据集、访问令牌或个人缓存。

## 文件说明

- `environment.yml`：新建 Conda 环境的基础定义。
- `requirements-lock.txt`：本次验证通过的关键 Python 包版本。
- `system-info.md`：当前操作系统、GPU、驱动和模型版本快照。
- `verify_environment.ps1`：环境、GPU和本地模型文件的一键验收脚本。
- `git-workflow.md`：作者上游仓库与个人复现仓库的 Git 管理方式。

## 重建环境

```powershell
conda env create -f local/environment.yml
conda activate sam3
$env:PYTHONNOUSERSITE = "1"
python -m pip install -r local/requirements-lock.txt
python -m pip install -e ".\sam3[notebooks,train,dev]"
```

模型权重需要单独下载到：

```text
checkpoints/sam3/sam3.pt
checkpoints/timesformer-base-finetuned-k400/pytorch_model.bin
Qwen2.5-VL-7B-Instruct/
```

验收：

```powershell
powershell -ExecutionPolicy Bypass -File .\local\verify_environment.ps1
```

注意：`requirements-lock.txt` 固定的是本次成功环境的关键运行依赖，而不是所有 Jupyter/开发工具的传递依赖。安装结束后必须运行 `pip check` 和验收脚本。
