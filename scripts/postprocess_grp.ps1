# Rebuild every derived artifact from the finished re-run, then gate the manuscript.
#
# Run AFTER scripts\rerun_grp.ps1 completes.
#
#   1. freeze_predictions_grp.py   checkpoints + five-seed results -> per-sample records
#   2. rebuild_paper_tables.py     per-sample records -> Tables I and V, five-seed blocks
#   3. derive_station_season_tables.py  -> Tables III/IV and the rule-vs-model diagnostic
#   4. compare_predet.py           archived pre-determinism run vs this one
#   5. verify_manuscript_numbers.py     manuscript <-> artifacts gate
#
# Step 5 exits non-zero when any number in the manuscript does not trace to an
# artifact, so this script's exit code is the answer to "is the paper consistent".

$ErrorActionPreference = 'Continue'
$ROOT = 'E:\Claude code\project\noise-label-cloud'
Set-Location $ROOT

$env:NL_SPLIT_TAG = 'grp'
$env:NL_CKPT_TAG  = 'grp'
$env:NL_SEED      = '42'
# Must match scripts\rerun_grp.ps1: the checkpoints were trained with the
# rebuilt composites, so evaluation has to read the same ones.
$env:NL_BASEMAP_DIR = 'E:\Data\Unet_Dataset\base_map_trainval'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:\NL_ARTIFACT_DIR -ErrorAction SilentlyContinue

$steps = @(
    @{ n = '01_freeze_predictions'; c = 'scripts\freeze_predictions_grp.py' },
    @{ n = '02_rebuild_tables';     c = 'scripts\rebuild_paper_tables.py' },
    @{ n = '03_derive_tables';      c = 'scripts\derive_station_season_tables.py' },
    @{ n = '04_compare_predet';     c = 'scripts\compare_predet.py' },
    @{ n = '05_manuscript_gate';    c = 'scripts\verify_manuscript_numbers.py' }
)

$results = @()
foreach ($s in $steps) {
    Write-Output ''
    Write-Output ('=' * 78)
    Write-Output ("  {0}   ({1})" -f $s.n, $s.c)
    Write-Output ('=' * 78)
    & python -X utf8 $s.c *>&1 | Tee-Object -FilePath (Join-Path $ROOT ("output\logs_grp\post_" + $s.n + '.log'))
    $code = $LASTEXITCODE
    Write-Output ("--- {0} exit={1}" -f $s.n, $code)
    $results += [pscustomobject]@{ step = $s.n; exit = $code }
}

Write-Output ''
Write-Output '=== post-processing summary ==='
$results | Format-Table -AutoSize | Out-String | Write-Output
$results | ConvertTo-Json | Set-Content (Join-Path $ROOT 'output\logs_grp\postprocess_summary.json')

$failed = @($results | Where-Object { $_.exit -ne 0 })
if ($failed) {
    Write-Output ("FAILED steps: " + (($failed | ForEach-Object { $_.step }) -join ', '))
    exit 1
}
Write-Output 'all post-processing steps succeeded; the manuscript traces to the artifacts'
exit 0
