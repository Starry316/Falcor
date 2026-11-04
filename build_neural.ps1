param(
    [Parameter(Mandatory=$true)][string]$SdkDirectory,
    [string]$BuildDirectory = 'build/windows-ninja-msvc-ci',
    [ValidateSet('Release','Debug')][string]$Configuration = 'Release',
    [switch]$Clean
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskSdk = (Resolve-Path -LiteralPath $SdkDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $taskSdk 'include/inference_bridge.h'))) {
    throw 'SdkDirectory must be an exported MPMat inference SDK.'
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $taskVsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $taskVsRoot = & $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $taskVsRoot) { throw 'Visual Studio C++ tools were not found.' }
    $taskVsDev = Join-Path $taskVsRoot 'Common7/Tools/VsDevCmd.bat'
    $taskEnvDirectory = Join-Path $taskRoot 'build'
    New-Item -ItemType Directory -Path $taskEnvDirectory -Force | Out-Null
    $taskEnvScript = Join-Path $taskEnvDirectory 'neural-vs-environment.cmd'
    @('@echo off', ('call "' + $taskVsDev + '" -no_logo -arch=x64 -host_arch=x64 >nul'),
        'if errorlevel 1 exit /b %errorlevel%', 'set') | Set-Content -LiteralPath $taskEnvScript -Encoding ASCII
    $taskEnvironment = & $env:ComSpec /d /c $taskEnvScript
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment setup failed.' }
    foreach ($taskEntry in $taskEnvironment) {
        if ($taskEntry -match '^([^=]+)=(.*)$' -and $Matches[1] -notin @('HOME','CODEX_HOME')) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}
$taskCmake = Join-Path $taskRoot 'tools/.packman/cmake/bin/cmake.exe'
$taskNinja = Join-Path $taskRoot 'tools/.packman/ninja/ninja.exe'
$env:VSLANG = '1033'

# CMake 3.24 can mis-decode localized MSVC /showIncludes output. Ninja matches
# raw bytes, so preserve the compiler's actual output encoding unchanged.
# Detect the actual prefix without changing the workstation locale.
$taskProbeDirectory = Join-Path ([IO.Path]::GetFullPath((Join-Path $taskRoot $BuildDirectory))) 'showincludes-probe'
New-Item -ItemType Directory -Path $taskProbeDirectory -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $taskProbeDirectory 'falcor-showincludes-probe.h'), '// include dependency probe', [Text.Encoding]::ASCII)
[IO.File]::WriteAllText((Join-Path $taskProbeDirectory 'probe.cpp'), "#include `"falcor-showincludes-probe.h`"`nint falcorIncludeProbe;", [Text.Encoding]::ASCII)
$taskCompiler = (Get-Command cl.exe).Source
$taskProbeProcess = [Diagnostics.Process]::new()
$taskProbeProcess.StartInfo = [Diagnostics.ProcessStartInfo]::new()
$taskProbeProcess.StartInfo.FileName = $taskCompiler
# Probe actual compilation: this compiler emits GBK for /EP but UTF-8 for /c.
$taskProbeProcess.StartInfo.Arguments = '/nologo /showIncludes /c /Fodetect-prefix.obj probe.cpp'
$taskProbeProcess.StartInfo.WorkingDirectory = $taskProbeDirectory
$taskProbeProcess.StartInfo.UseShellExecute = $false
$taskProbeProcess.StartInfo.CreateNoWindow = $false
$taskProbeProcess.StartInfo.RedirectStandardOutput = $true
$taskProbeProcess.StartInfo.RedirectStandardError = $true
$taskProbeOut = [IO.MemoryStream]::new()
$taskProbeError = [IO.MemoryStream]::new()
$taskProbeProcess.Start() | Out-Null
$taskReadOut = $taskProbeProcess.StandardOutput.BaseStream.CopyToAsync($taskProbeOut)
$taskReadError = $taskProbeProcess.StandardError.BaseStream.CopyToAsync($taskProbeError)
$taskProbeProcess.WaitForExit()
$taskReadOut.GetAwaiter().GetResult() | Out-Null
$taskReadError.GetAwaiter().GetResult() | Out-Null
if ($taskProbeProcess.ExitCode -ne 0) { throw 'MSVC include-prefix probe failed.' }
# Latin-1 is used solely as a reversible byte mapping, never as a text decoder.
$taskBytesEncoding = [Text.Encoding]::GetEncoding(28591)
$taskProbeText = $taskBytesEncoding.GetString($taskProbeOut.ToArray()) + $taskBytesEncoding.GetString($taskProbeError.ToArray())
$taskPrefixMatch = [regex]::Match($taskProbeText, '(?m)^([^\r\n]*?)(?:[A-Za-z]:[\\/])[^\r\n]*falcor-showincludes-probe\.h\r?$')
if (-not $taskPrefixMatch.Success) { throw 'Cannot identify the actual MSVC /showIncludes prefix.' }
$taskIncludePrefix = $taskPrefixMatch.Groups[1].Value
$taskProbeProcess.Dispose(); $taskProbeOut.Dispose(); $taskProbeError.Dispose()

function Repair-MsvcIncludePrefix {
    param([string]$Directory)
    $taskCompilerMetadata = Get-ChildItem -LiteralPath (Join-Path $Directory 'CMakeFiles') -Recurse -File -Filter 'CMake*Compiler.cmake' -ErrorAction SilentlyContinue
    foreach ($taskMetadata in $taskCompilerMetadata) {
        $taskText = $taskBytesEncoding.GetString([IO.File]::ReadAllBytes($taskMetadata.FullName))
        $taskRepaired = [regex]::Replace($taskText, '(?m)^(set\(CMAKE_\w+_CL_SHOWINCLUDES_PREFIX ")[^"\r\n]*("\))', [Text.RegularExpressions.MatchEvaluator]{
            param($match)
            $match.Groups[1].Value + $taskIncludePrefix + $match.Groups[2].Value
        })
        if ($taskText -cne $taskRepaired) { [IO.File]::WriteAllBytes($taskMetadata.FullName, $taskBytesEncoding.GetBytes($taskRepaired)) }
    }
    $taskRulesPath = Join-Path $Directory 'CMakeFiles/rules.ninja'
    if (Test-Path -LiteralPath $taskRulesPath) {
        $taskText = $taskBytesEncoding.GetString([IO.File]::ReadAllBytes($taskRulesPath))
        $taskRepaired = [regex]::Replace($taskText, '(?m)^msvc_deps_prefix = [^\r\n]*', [Text.RegularExpressions.MatchEvaluator]{
            param($match)
            'msvc_deps_prefix = ' + $taskIncludePrefix
        })
        if ($taskText -cne $taskRepaired) { [IO.File]::WriteAllBytes($taskRulesPath, $taskBytesEncoding.GetBytes($taskRepaired)) }
    }
}

# Verify the detected bytes with Ninja itself before relying on incremental builds.
$taskProbeNinja = "msvc_deps_prefix = $taskIncludePrefix`nrule check`n  command = `"$($taskCompiler.Replace('$', '$$'))`" /nologo /showIncludes /c /Foprefix-probe.obj probe.cpp`n  deps = msvc`nbuild prefix-probe.obj: check probe.cpp`n"
[IO.File]::WriteAllBytes((Join-Path $taskProbeDirectory 'build.ninja'), $taskBytesEncoding.GetBytes($taskProbeNinja))
& $taskNinja -C $taskProbeDirectory prefix-probe.obj
if ($LASTEXITCODE -ne 0) { throw 'Ninja include-prefix probe failed.' }
$taskProbeDeps = & $taskNinja -C $taskProbeDirectory -t deps prefix-probe.obj
if ($LASTEXITCODE -ne 0 -or -not ($taskProbeDeps -match 'falcor-showincludes-probe\.h')) {
    throw 'Ninja failed to record the probe header dependency; refusing an unsafe incremental build.'
}
Push-Location $taskRoot
try {
    Write-Host "Neural inference SDK: $taskSdk"
    Repair-MsvcIncludePrefix $BuildDirectory
    & $taskCmake --preset windows-ninja-msvc-ci -B $BuildDirectory "-DMPMAT_SDK_DIR:PATH=$taskSdk"
    if ($LASTEXITCODE -ne 0) { throw 'Falcor CMake configuration failed.' }
    Repair-MsvcIncludePrefix $BuildDirectory
    $taskBuildArguments = @('--build', $BuildDirectory, '--config', $Configuration)
    if ($Clean) { $taskBuildArguments += '--clean-first' }
    $taskBuildArguments += @('--target', 'Falcor', 'Mogwai', 'GBuffer', 'ToneMapper', 'AccumulatePass', 'NeuralMaterialPass', 'AssimpImporter', 'PythonImporter')
    & $taskCmake @taskBuildArguments
    if ($LASTEXITCODE -ne 0) { throw 'Falcor neural renderer build failed.' }
    # A separate runtime prevents unrelated plugins from an older build loading
    # against the newly built Falcor.dll. The original plugin manifest is untouched.
    $taskSourceRuntime = Join-Path $BuildDirectory "bin/$Configuration"
    $taskNeuralRuntime = Join-Path $BuildDirectory "neural-runtime/$Configuration"
    New-Item -ItemType Directory -Path $taskNeuralRuntime -Force | Out-Null
    Get-ChildItem -LiteralPath $taskSourceRuntime -File | Where-Object {
        $_.Extension -in @('.dll','.exe','.manifest','.json')
    } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $taskNeuralRuntime -Force }
    foreach ($taskSubdir in @('D3D12','python','pythondist','shaders','data','scripts')) {
        $taskSource = Join-Path $taskSourceRuntime $taskSubdir
        if (Test-Path -LiteralPath $taskSource) {
            & robocopy $taskSource (Join-Path $taskNeuralRuntime $taskSubdir) /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
            if ($LASTEXITCODE -ge 8) { throw "Failed to copy runtime directory $taskSubdir" }
        }
    }
    $taskPlugins = @('AssimpImporter','PythonImporter','GBuffer','ToneMapper','AccumulatePass','NeuralMaterialPass')
    $taskPluginDirectory = Join-Path $taskNeuralRuntime 'plugins'
    New-Item -ItemType Directory -Path $taskPluginDirectory -Force | Out-Null
    foreach ($taskPlugin in $taskPlugins) {
        Copy-Item -LiteralPath (Join-Path $taskSourceRuntime "plugins/$taskPlugin.dll") -Destination $taskPluginDirectory -Force
    }
    $taskPlugins | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskPluginDirectory 'plugins.json') -Encoding ASCII
    Write-Host "Neural runtime: $taskNeuralRuntime"
    # Robocopy uses successful nonzero codes for files copied/skipped.
    $global:LASTEXITCODE = 0
} finally { Pop-Location }
