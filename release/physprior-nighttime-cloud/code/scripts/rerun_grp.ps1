# Rerun every experiment that the audit flagged, on the group-disjoint split.
#
#   NL_SPLIT_TAG=grp  -> Train_grp / Val_grp  (see scripts/build_split_grp.py)
#   NL_CKPT_TAG=grp   -> output/checkpoints_grp, output/logs_grp
#   NL_SEED=42        -> consumed by determinism.seed_everything in every entry point
#
# Old artifacts are untouched; every stage writes tagged outputs.
# A failing stage is logged and skipped so that one bad run cannot abort the rest.
#
# Every stage is deterministic: the entry points call seed_everything(42), which
# also pins cuDNN to deterministic kernels and disables autotuning. Re-running
# this script from an empty output/ must reproduce every number bit-for-bit.

$ErrorActionPreference = 'Continue'
$env:NL_SPLIT_TAG = 'grp'
$env:NL_CKPT_TAG  = 'grp'
$env:NL_SEED      = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
# Basemap rebuilt from train+val clear scenes only (scripts/rebuild_basemap.py).
# base_map/ was built from a lost hand-picked candidate list that searched Test/
# as well, so this run uses a composite whose construction is reproducible.
$env:NL_BASEMAP_DIR = 'E:\Data\Unet_Dataset\base_map_trainval'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$ROOT = 'E:\Claude code\project\noise-label-cloud'
$LOGD = Join-Path $ROOT 'output\logs_grp'
New-Item -ItemType Directory -Force -Path $LOGD | Out-Null
Set-Location $ROOT

$stages = @(
  @{ n = '01_loss_correction';  c = @('scripts\redo_loss_correction.py');                    a = 'output\loss_correction_redo_grp.json' },
  @{ n = '02_standard';         c = @('train.py', '--method', 'baseline');                   a = 'output\checkpoints_grp\baseline_best.pth' },
  @{ n = '03_coteaching';       c = @('train.py', '--method', 'coteaching');                 a = 'output\checkpoints_grp\coteaching_a_best.pth' },
  @{ n = '04_gce';              c = @('train.py', '--method', 'gce', '--gce-q', '0.7');      a = 'output\checkpoints_grp\gce_q0.7_best.pth' },
  @{ n = '05_mixup';            c = @('train.py', '--method', 'mixup');                      a = 'output\checkpoints_grp\mixup_best.pth' },
  @{ n = '06_physprior_dl';     c = @('train.py', '--method', 'phys-mod');                   a = 'output\checkpoints_grp\physprior_moderate_best.pth' },
  @{ n = '07_ablation_2x2';     c = @('scripts\train_2x2_ablation.py');                      a = 'output\2x2_ablation_grp.json' },
  @{ n = '08_physprior_5seed';  c = @('scripts\train_physprior_5seeds.py');                  a = 'output\physprior_5seed_results_grp.json' },
  @{ n = '09_coteaching_5seed'; c = @('scripts\train_coteaching_5seeds.py');                 a = 'output\coteaching_5seed_results_grp.json' },
  @{ n = '10_cot_ensemble';     c = @('scripts\train_physprior_coteaching_long.py');         a = 'output\checkpoints_grp\physprior_coteaching_long_a_best.pth' }
)

$results = @()
$t0 = Get-Date
Write-Output "=== rerun on group-disjoint split, start $t0 ==="

# A stage that dies part-way still leaves an artifact behind, and the skip test
# below would then treat it as done. The two multi-seed stages are the ones this
# actually bit (a killed run left 2 of 5 seeds and the stage was skipped), so
# they are checked for completeness rather than mere existence.
function Test-StageDone($relPath) {
    $p = Join-Path $ROOT $relPath
    if (-not (Test-Path $p)) { return $false }
    if ($relPath -match '5seed_results') {
        try {
            $j = Get-Content $p -Raw | ConvertFrom-Json
            $n = @($j).Count
            if ($n -lt 5) {
                Write-Output ("      (artifact has {0} of 5 seeds - treating as incomplete)" -f $n)
                return $false
            }
        } catch { return $false }
    }
    return $true
}

foreach ($s in $stages) {
    $log = Join-Path $LOGD ($s.n + '.log')
    $ts = Get-Date
    if ($s.a -and (Test-StageDone $s.a)) {
        Write-Output ("--- [{0}] {1}  SKIP (artifact present: {2})" -f $ts.ToString('HH:mm:ss'), $s.n, $s.a)
        $results += [pscustomobject]@{ stage = $s.n; exit = 'skipped'; minutes = 0 }
        continue
    }
    Write-Output ""
    Write-Output ("--- [{0}] {1}  args: {2}" -f $ts.ToString('HH:mm:ss'), $s.n, ($s.c -join ' '))
    $c = $s.c
    if ($c.Count -eq 1) {
        & python -X utf8 $c[0] *>&1 | Tee-Object -FilePath $log
    } else {
        & python -X utf8 $c[0] $c[1..($c.Count - 1)] *>&1 | Tee-Object -FilePath $log
    }
    $code = $LASTEXITCODE
    $dur = (Get-Date) - $ts
    Write-Output ("--- [{0}] exit={1}  elapsed={2:hh\:mm\:ss}" -f $s.n, $code, $dur)
    $results += [pscustomobject]@{ stage = $s.n; exit = $code; minutes = [math]::Round($dur.TotalMinutes, 1) }
}

$total = (Get-Date) - $t0
Write-Output ""
Write-Output "=== stage summary ==="
$results | Format-Table -AutoSize | Out-String | Write-Output
Write-Output ("total elapsed {0:hh\:mm\:ss}" -f $total)
$results | ConvertTo-Json | Set-Content (Join-Path $LOGD 'rerun_stage_summary.json')
Write-Output "=== done ==="
