param(
    [ValidateSet('fuzzy_mip','fuzzy_silk','shot_silk')][string]$Material = 'fuzzy_mip',
    [string]$Model = '',
    [string]$Scene = 'Data/NeuralMaterial/NeuralMaterial.pyscene',
    [string]$Environment = 'build/neural-assets/studio.hdr',
    [string]$BuildDirectory = 'build/windows-ninja-msvc-ci',
    [ValidateRange(16,16384)][int]$Width = 1920,
    [ValidateRange(16,16384)][int]$Height = 1080,
    [ValidateRange(0,100000)][int]$Frames = 0,
    [string]$CaptureDirectory = '',
    [string[]]$CaptureOutputs = @(),
    [string]$OptionsJson = '',
    [switch]$Headless,
    [switch]$VSync,
    [switch]$Fullscreen,
    [switch]$Animate,
    [switch]$Build,
    [string]$CudaArchitectures = 'native'
)
$ErrorActionPreference = 'Stop'
if ($Build) {
    & (Join-Path $PSScriptRoot 'build_neural.ps1') -BuildDirectory $BuildDirectory -CudaArchitectures $CudaArchitectures
}
Push-Location $PSScriptRoot
$taskOldConfig = $env:MPMAT_CONFIG
$taskOldEnvironment = $env:MPMAT_ENVIRONMENT
$taskOldAnimate = $env:MPMAT_ANIMATE
try {
    if (-not $Model) { $Model = "build/neural-assets/$Material.mpmat" }
    $taskModel = (Resolve-Path -LiteralPath $Model).Path
    $taskScene = (Resolve-Path -LiteralPath $Scene).Path
    $taskExe = Join-Path $BuildDirectory 'neural-runtime/Release/Mogwai.exe'
    if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Build the neural Falcor plugin first with build_neural.ps1.' }
    $taskOptions = if ($OptionsJson) { Get-Content -LiteralPath $OptionsJson -Raw | ConvertFrom-Json } else { @{} }
    $taskConfig = @{model=$taskModel; scene=$taskScene; options=$taskOptions; frames=$Frames}
    if ($CaptureOutputs.Count) { $taskConfig.captureOutputs = $CaptureOutputs }
    if ($CaptureDirectory) { $taskConfig.captureDirectory = [IO.Path]::GetFullPath($CaptureDirectory) }
    $taskRunDirectory = Join-Path $PSScriptRoot 'build/neural-runs'
    New-Item -ItemType Directory -Path $taskRunDirectory -Force | Out-Null
    $taskConfigPath = Join-Path $taskRunDirectory ("run_" + [guid]::NewGuid().ToString('N') + '.json')
    $taskConfig | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskConfigPath -Encoding UTF8
    $env:MPMAT_CONFIG = $taskConfigPath
    $env:MPMAT_ENVIRONMENT = if ($Environment) { (Resolve-Path -LiteralPath $Environment).Path } else { '' }
    $env:MPMAT_ANIMATE = if ($Animate) { '1' } else { '0' }
    $taskArguments = @('--device-type','d3d12','--script','scripts/NeuralMaterial.py','--width',"$Width",'--height',"$Height")
    $taskArguments += if ($VSync) { '--vsync' } else { '--no-vsync' }
    if ($Fullscreen) { $taskArguments += '--fullscreen' }
    if ($Headless) {
        if ($Frames -lt 1) { throw 'Headless mode requires -Frames greater than zero.' }
        $taskArguments += '--headless'
    }
    & $taskExe @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Falcor exited with code $LASTEXITCODE" }
} finally {
    $env:MPMAT_CONFIG = $taskOldConfig
    $env:MPMAT_ENVIRONMENT = $taskOldEnvironment
    $env:MPMAT_ANIMATE = $taskOldAnimate
    Pop-Location
}
