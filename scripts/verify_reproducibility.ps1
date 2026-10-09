# Bit-reproducibility check for the five-seed PhysPrior protocol.
#
# The main re-run (scripts/rerun_grp.ps1, NL_CKPT_TAG=grp) writes
# output/physprior_5seed_results_grp.json. This script re-runs the identical
# protocol in a second process under a different output tag, then compares the
# two result files field by field.
#
# Run AFTER the main re-run finishes: both trainings want the GPU.
#
# Before determinism was pinned, two runs of this protocol with the same seeds
# disagreed on individual seeds by up to 4 pp. Expected outcome now: PASS.

$ErrorActionPreference = 'Continue'
$ROOT = 'E:\Claude code\project\noise-label-cloud'
Set-Location $ROOT

$env:NL_SPLIT_TAG = 'grp'
$env:NL_SEED      = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$main = Join-Path $ROOT 'output\physprior_5seed_results_grp.json'
$repl = Join-Path $ROOT 'output\physprior_5seed_results_detcheck.json'

if (-not (Test-Path $main)) {
    Write-Output "ABORT: main result $main missing - run scripts\rerun_grp.ps1 first"
    exit 1
}

# --- fingerprint the first run ----------------------------------------------
$hashA = (Get-FileHash $main -Algorithm SHA256).Hash
Write-Output "main run   : $main"
Write-Output "sha256     : $hashA"

# --- second run, isolated output tag ----------------------------------------
$env:NL_CKPT_TAG = 'detcheck'
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'output\logs_detcheck') | Out-Null

Write-Output ""
Write-Output "=== re-running the same five seeds in a second process (tag=detcheck) ==="
$t0 = Get-Date
& python -X utf8 scripts\train_physprior_5seeds.py *>&1 |
    Tee-Object -FilePath (Join-Path $ROOT 'output\logs_detcheck\repro_check.log')
$code = $LASTEXITCODE
Write-Output ("--- exit={0}  elapsed={1:hh\:mm\:ss}" -f $code, ((Get-Date) - $t0))
if ($code -ne 0) {
    Write-Output "ABORT: the replay run failed; see output\logs_detcheck\repro_check.log"
    exit $code
}

# --- compare ----------------------------------------------------------------
Write-Output ""
python -X utf8 scripts\verify_reproducibility.py $main $repl
$verdict = $LASTEXITCODE

$hashB = (Get-FileHash $repl -Algorithm SHA256).Hash
Write-Output ""
Write-Output "sha256 A   : $hashA"
Write-Output "sha256 B   : $hashB"
if ($hashA -eq $hashB) {
    Write-Output "result files are byte-identical"
} else {
    Write-Output "result files differ byte-wise (float formatting or ordering); see the field comparison above"
}

exit $verdict
