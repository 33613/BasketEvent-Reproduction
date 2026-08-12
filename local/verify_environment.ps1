$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE ".conda\envs\sam3\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "sam3 Python not found: $python"
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed" }

& $python -c "import torch, torchvision, transformers, bitsandbytes, triton; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)); print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); print('torchvision:', torchvision.__version__); print('transformers:', transformers.__version__); print('bitsandbytes:', bitsandbytes.__version__); print('triton:', triton.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python/CUDA import check failed" }

$requiredFiles = @(
    (Join-Path $projectRoot "checkpoints\sam3\sam3.pt"),
    (Join-Path $projectRoot "checkpoints\timesformer-base-finetuned-k400\pytorch_model.bin"),
    (Join-Path $projectRoot "Qwen2.5-VL-7B-Instruct\model.safetensors.index.json")
)

foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required local model file is missing: $path"
    }
    $item = Get-Item -LiteralPath $path
    Write-Output ("MODEL_FILE_OK {0} ({1} bytes)" -f $item.FullName, $item.Length)
}

Push-Location $projectRoot
try {
    & $python -c "from transformers import TimesformerModel; TimesformerModel.from_pretrained(r'checkpoints\timesformer-base-finetuned-k400', local_files_only=True); print('TIMESFORMER_LOAD_OK')"
    if ($LASTEXITCODE -ne 0) { throw "TimeSformer local load failed" }

    & $python -c "import ast, pathlib; files=['track_one_video.py','recognize.py','inference.py','src/model.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in files]; print('PROJECT_SYNTAX_OK')"
    if ($LASTEXITCODE -ne 0) { throw "Project syntax check failed" }
}
finally {
    Pop-Location
}

Write-Output "ENVIRONMENT_VERIFICATION_OK"
