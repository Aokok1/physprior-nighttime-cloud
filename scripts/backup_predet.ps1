# Archive the pre-determinism re-run artifacts before the deterministic re-run.
#
# The 2026-09-24 re-run produced every number currently in the manuscript, but
# stages 01-06 and 10 ran with no RNG seeding at all and stages 07-09 ran with
# cuDNN autotuning on. Those artifacts are kept here verbatim so the old-vs-new
# comparison in the run report can be re-derived from disk.
#
# Nothing is deleted: artifacts are MOVED into _predet_backup_<stamp>\.

$ErrorActionPreference = 'Stop'
$ROOT  = 'E:\Claude code\project\noise-label-cloud'
$STAMP = '20260924'
$DEST  = Join-Path $ROOT "output\_predet_backup_$STAMP"

New-Item -ItemType Directory -Force -Path $DEST | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DEST 'logs_grp') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DEST 'checkpoints_grp') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DEST 'frozen_predictions') | Out-Null

$moved = 0

# --- tagged JSON results in output/ -----------------------------------------
Get-ChildItem (Join-Path $ROOT 'output') -File -Filter '*grp*.json' | ForEach-Object {
    Move-Item $_.FullName (Join-Path $DEST $_.Name) -Force
    $moved++
}
# artifacts the tagged stages write without a _grp suffix in their name
foreach ($n in @('physprior_moderate_stats.json', 'paper_tables_grp.json', 'derived_tables_grp.json')) {
    $p = Join-Path $ROOT "output\$n"
    if (Test-Path $p) { Move-Item $p (Join-Path $DEST $n) -Force; $moved++ }
}

# --- logs and checkpoints ---------------------------------------------------
foreach ($d in @('logs_grp', 'checkpoints_grp')) {
    Get-ChildItem (Join-Path $ROOT "output\$d") -File | ForEach-Object {
        Move-Item $_.FullName (Join-Path $DEST "$d\$($_.Name)") -Force
        $moved++
    }
}

# --- frozen predictions -----------------------------------------------------
Get-ChildItem (Join-Path $ROOT 'output\frozen_predictions') -File -Filter '*grp*.json' -ErrorAction SilentlyContinue |
    ForEach-Object { Move-Item $_.FullName (Join-Path $DEST "frozen_predictions\$($_.Name)") -Force; $moved++ }

Write-Output "archived $moved artifacts -> $DEST"

# --- confirm the driver's skip-artifacts are gone ---------------------------
$checks = @(
    'output\loss_correction_redo_grp.json',
    'output\checkpoints_grp\baseline_best.pth',
    'output\checkpoints_grp\coteaching_a_best.pth',
    'output\checkpoints_grp\gce_q0.7_best.pth',
    'output\checkpoints_grp\mixup_best.pth',
    'output\checkpoints_grp\physprior_moderate_best.pth',
    'output\checkpoints_grp\physprior_coteaching_long_a_best.pth',
    'output\2x2_ablation_grp.json',
    'output\physprior_5seed_results_grp.json',
    'output\coteaching_5seed_results_grp.json'
)
$left = $checks | Where-Object { Test-Path (Join-Path $ROOT $_) }
if ($left) {
    Write-Output 'STILL PRESENT (driver would skip these stages):'
    $left | ForEach-Object { Write-Output "  $_" }
    exit 1
}
Write-Output 'all 10 stage artifacts cleared - the driver will re-run every stage'
