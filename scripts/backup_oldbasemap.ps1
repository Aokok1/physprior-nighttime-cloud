# Archive the run made with the original hand-picked basemaps.
#
# `base_map/` was built by basemap_make.py from hand-selected "TrueClear"
# candidates that searched both Train/ and Test/ source directories. The
# candidate list is gone, so whether a test scene contributed cannot be
# determined. The basemaps were rebuilt from train+val clear scenes only
# (`base_map_trainval/`, scripts/rebuild_basemap.py), which changes the third
# input channel and therefore every trained number.
#
# Nothing is deleted: artifacts are MOVED into _oldbasemap_backup_<stamp>\.

$ErrorActionPreference = 'Stop'
$ROOT  = 'E:\Claude code\project\noise-label-cloud'
$STAMP = '20260924'
$DEST  = Join-Path $ROOT "output\_oldbasemap_backup_$STAMP"

New-Item -ItemType Directory -Force -Path $DEST | Out-Null
foreach ($d in @('logs_grp', 'checkpoints_grp', 'frozen_predictions')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DEST $d) | Out-Null
}

$moved = 0
Get-ChildItem (Join-Path $ROOT 'output') -File -Filter '*grp*.json' | ForEach-Object {
    Move-Item $_.FullName (Join-Path $DEST $_.Name) -Force; $moved++
}
foreach ($n in @('physprior_moderate_stats.json', 'paper_tables_grp.json', 'derived_tables_grp.json')) {
    $p = Join-Path $ROOT "output\$n"
    if (Test-Path $p) { Move-Item $p (Join-Path $DEST $n) -Force; $moved++ }
}
foreach ($d in @('logs_grp', 'checkpoints_grp')) {
    Get-ChildItem (Join-Path $ROOT "output\$d") -File | ForEach-Object {
        Move-Item $_.FullName (Join-Path $DEST "$d\$($_.Name)") -Force; $moved++
    }
}
Get-ChildItem (Join-Path $ROOT 'output\frozen_predictions') -File -Filter '*grp*.json' -ErrorAction SilentlyContinue |
    ForEach-Object { Move-Item $_.FullName (Join-Path $DEST "frozen_predictions\$($_.Name)") -Force; $moved++ }

Write-Output "archived $moved artifacts -> $DEST"

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
