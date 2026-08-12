# Verified local system snapshot

Snapshot date: 2026-08-12

| Item | Verified value |
|---|---|
| OS | Windows 11 / PowerShell 5.1.26100.8894 |
| Conda | 24.11.3, environment `sam3` |
| Python | 3.12.11 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU memory | 8151 MiB (PyTorch reports 7.96 GiB) |
| Driver | 591.86 |
| Compute capability | 12.0 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| CUDA runtime used by PyTorch | 12.8 |
| Transformers | 5.15.0 |
| bitsandbytes | 0.50.0 |
| triton-windows | 3.4.0.post21 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |

## Verified models

| Model | Verification |
|---|---|
| SAM3 | `sam3.pt` parsed and `Sam3VideoPredictorMultiGPU` initialized on `cuda:0` |
| Qwen2.5-VL-7B-Instruct | five safetensors shards present and readable; configured for NF4 4-bit/BF16 |
| TimeSformer K400 | `pytorch_model.bin` size 486,348,721 bytes; 247 weight groups loaded |

TimeSformer reports `classifier.weight` and `classifier.bias` as unexpected because BasketEvent loads `TimesformerModel` without the original Kinetics-400 classification head. This is expected.

## Important environment notes

- Set `$env:PYTHONNOUSERSITE="1"` before installation and CLI work. This prevents packages under `%APPDATA%\Python` from shadowing the Conda environment.
- The environment was repaired from Conda/pip mixing. `pip check` currently reports no broken requirements.
- SciPy is the PyPI/OpenBLAS build. Combined with the deterministic visualization palette, this avoids Windows OMP Error #15 caused by duplicate Intel OpenMP runtimes.
- The repository's original `requirements.txt` targets Torch 2.7/CUDA 12.6; this snapshot uses Torch 2.8/CUDA 12.8 for the RTX 5060.
- Model weights and Hugging Face tokens are intentionally excluded from Git.

## Source revisions

- BasketEvent base commit: `8a313f3ad4476735ddac38543578e19c1bccebd5`
- SAM3 base commit: `5dd401d1c5c1d5c3eedff06d41b77af824517619`
