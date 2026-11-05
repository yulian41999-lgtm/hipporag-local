param(
    [ValidateSet('embedding', 'sample', 'reload', 'probe')]
    [string]$Mode = 'sample',
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [ValidateSet('sample', 'chinese_demo')]
    [string]$Dataset = 'sample'
)
$ErrorActionPreference = 'Stop'
$pythonExe = if ($env:HIPPO_PYTHON) { $env:HIPPO_PYTHON } elseif ($env:CONDA_DEFAULT_ENV -eq 'corecoder') { 'python' } elseif (Test-Path -LiteralPath 'D:\Anaconda3\envs\corecoder\python.exe') { 'D:\Anaconda3\envs\corecoder\python.exe' } else { 'python' }
if ($Mode -eq 'probe') {
    & $pythonExe (Join-Path $PSScriptRoot 'probe_llm.py')
} else {
    & $pythonExe (Join-Path $PSScriptRoot 'run.py') $Mode --device $Device --dataset $Dataset
}
exit $LASTEXITCODE
