param([string]$BuildDirectory = 'build/neural-source-relocation')
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory = Join-Path $taskRoot $BuildDirectory }
$taskTestRoot = [IO.Path]::GetFullPath($BuildDirectory)
$taskSource = Join-Path $taskTestRoot 'source with spaces'
$taskBuild = Join-Path $taskTestRoot 'build with spaces'
$taskCmake = Join-Path $taskRoot 'tools/.packman/cmake/bin/cmake.exe'
$taskNinja = Join-Path $taskRoot 'tools/.packman/ninja/ninja.exe'
if (-not (Test-Path -LiteralPath $taskCmake)) { throw 'Run build_neural.ps1 to prepare Falcor tools first.' }
New-Item -ItemType Directory -Path $taskSource -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $taskRoot 'Source/NeuralInference') -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $taskSource -Force }

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $taskVsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $taskVsRoot = & $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $taskVsRoot) { throw 'Visual Studio C++ tools were not found.' }
    $taskEnvScript = Join-Path $taskTestRoot 'compiler-environment.cmd'
    @('@echo off', ('call "' + (Join-Path $taskVsRoot 'Common7/Tools/VsDevCmd.bat') + '" -no_logo -arch=x64 -host_arch=x64 >nul'),
        'if errorlevel 1 exit /b %errorlevel%', 'set') | Set-Content -LiteralPath $taskEnvScript -Encoding ASCII
    $taskEnvironment = & $env:ComSpec /d /c $taskEnvScript
    if ($LASTEXITCODE -ne 0) { throw 'Compiler environment setup failed.' }
    foreach ($taskEntry in $taskEnvironment) {
        if ($taskEntry -match '^([^=]+)=(.*)$' -and $Matches[1] -notin @('HOME', 'CODEX_HOME')) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

# Verify that the self-contained snapshot has exactly the recorded kernel data.
$taskManifest = Get-Content -LiteralPath (Join-Path $taskSource 'UPSTREAM.json') -Raw | ConvertFrom-Json
foreach ($taskEntry in $taskManifest.files.PSObject.Properties) {
    $taskHash = (Get-FileHash -LiteralPath (Join-Path $taskSource $taskEntry.Name) -Algorithm SHA256).Hash
    if ($taskHash -ine $taskEntry.Value) { throw "Source snapshot hash mismatch: $($taskEntry.Name)" }
}

& $taskCmake -S $taskSource -B $taskBuild -G Ninja "-DCMAKE_MAKE_PROGRAM:FILEPATH=$taskNinja" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'Relocated inference configuration failed.' }
& $taskCmake --build $taskBuild --target mpmat_inference --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Relocated inference build failed.' }
$taskDll = Join-Path $taskBuild 'mpmat_inference.dll'
if (-not (Test-Path -LiteralPath $taskDll)) { throw 'Relocated build produced no inference library.' }

# Changing a shared kernel header must rebuild the embedded cooperative PTX,
# not just the ordinary CUDA translation unit. Only the copied test file changes.
Add-Content -LiteralPath (Join-Path $taskSource 'lobe_math.h') -Value "`n// Incremental dependency smoke test."
$taskRebuild = & $taskCmake --build $taskBuild --target mpmat_inference --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Incremental relocated inference rebuild failed.' }
if (-not ($taskRebuild -match 'Compiling and embedding neural cooperative-vector device code')) {
    throw 'Shared kernel header failed to invalidate embedded PTX.'
}
$taskRebuild | Write-Output
# Reconstruct the embedded string and compare with the actual compiler output.
$taskHeader = [IO.File]::ReadAllText((Join-Path $taskBuild 'coop_ptx.h'))
$taskChunks = [regex]::Matches($taskHeader, 'text\.append\(R"MPMATPTX\((.*?)\)MPMATPTX"\);', [Text.RegularExpressions.RegexOptions]::Singleline)
$taskReconstructed = [Text.StringBuilder]::new()
foreach ($taskChunk in $taskChunks) { [void]$taskReconstructed.Append($taskChunk.Groups[1].Value) }
if ($taskReconstructed.ToString() -cne [IO.File]::ReadAllText((Join-Path $taskBuild 'coop_programs.ptx')).Replace("`r`n", "`n")) {
    throw 'Embedded cooperative PTX differs from compiler output.'
}
@{ passed = $true; sourceFilesVerified = @($taskManifest.files.PSObject.Properties).Count;
    headerInvalidatesEmbeddedPtx = $true; embeddedPtxExact = $true; library = $taskDll } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskTestRoot 'results.json') -Encoding UTF8
Write-Host 'Relocated source-only build, source hashes and shared-header dependencies passed.'
