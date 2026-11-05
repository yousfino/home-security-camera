# Local Windows launcher. Models/data stay outside Git. No networking or elevation.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [Parameter(Mandatory=$true)][string]$PersonModel,
    [Parameter(Mandatory=$true)][string]$FaceModelRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$PersonSha256,
    [ValidateRange(1,100)][int]$MaxPeople = 10,
    [ValidateRange(1,86400)][int]$MaxRuntime = 3600
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) { throw 'Create the Python 3.11 project virtual environment first.' }
if (-not (Test-Path $PersonModel)) { throw 'Provision a local licensed YOLO ONNX model first.' }
if (-not (Test-Path $FaceModelRoot)) { throw 'Provision a local licensed face model pack first.' }
$AppArgs = @('-m','security_ai','--data-dir',$DataDir,'monitor',
    '--person-model',$PersonModel,'--person-sha256',$PersonSha256,
    '--face-model-root',$FaceModelRoot,'--headless','--metrics',
    '--max-people',"$MaxPeople",'--width','1280','--height','720',
    '--embeddings-per-frame','2','--max-frame-age','2',
    '--recognition-interval','0.5','--max-runtime',"$MaxRuntime")
Push-Location $ProjectRoot
try {
    & $PythonExe @AppArgs
    $AppExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $AppExitCode
