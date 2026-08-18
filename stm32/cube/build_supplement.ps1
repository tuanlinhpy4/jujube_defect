param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [switch]$RamOnly
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bundle = Join-Path $env:LOCALAPPDATA "stm32cube\bundles"
$CMake = Join-Path $Bundle "cmake\4.3.1+st.1\bin\cmake.exe"
$Ninja = Join-Path $Bundle "ninja\1.13.2+st.1\bin\ninja.exe"
$GccBin = Join-Path $Bundle "gnu-tools-for-stm32\14.3.1+st.2\bin"
$BuildDir = Join-Path $Here "build_supplement_xip"
$Mode = if ($RamOnly) { "ram_only" } else { "benchmark" }
$ArtifactDir = Join-Path $Here "..\host\out\supplement\firmware\$Mode"

foreach ($Path in @($CMake, $Ninja, (Join-Path $GccBin "arm-none-eabi-gcc.exe"))) {
    if (-not (Test-Path $Path)) { throw "Missing STM32 tool: $Path" }
}

$env:PATH = "$GccBin;$(Split-Path $CMake -Parent);$(Split-Path $Ninja -Parent);$env:PATH"
$RamFlag = if ($RamOnly) { "ON" } else { "OFF" }

& $CMake -G Ninja `
    -DCMAKE_MAKE_PROGRAM="$Ninja" `
    "-DCMAKE_BUILD_TYPE=$Config" `
    -DXIP_BUILD=ON `
    "-DJUJUBE_RAM_ONLY=$RamFlag" `
    -S "$Here" `
    -B "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Ninja -C "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null
foreach ($Name in @("jujube_h750.elf", "jujube_h750_flash.hex", "tflm_app.bin", "jujube_h750.map")) {
    $Source = Join-Path $BuildDir $Name
    if (Test-Path $Source) { Copy-Item -Force $Source $ArtifactDir }
}
Write-Host "=> XIP $Mode artifacts: $ArtifactDir"
