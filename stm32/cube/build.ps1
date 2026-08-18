param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bundle = Join-Path $env:LOCALAPPDATA "stm32cube\bundles"
$CMake = Join-Path $Bundle "cmake\4.3.1+st.1\bin\cmake.exe"
$Ninja = Join-Path $Bundle "ninja\1.13.2+st.1\bin\ninja.exe"
$GccBin = Join-Path $Bundle "gnu-tools-for-stm32\14.3.1+st.2\bin"
$BuildDir = Join-Path $Here "build"

foreach ($Path in @($CMake, $Ninja, (Join-Path $GccBin "arm-none-eabi-gcc.exe"))) {
    if (-not (Test-Path $Path)) {
        throw "Missing STM32 tool: $Path"
    }
}

$env:PATH = "$GccBin;$(Split-Path $CMake -Parent);$(Split-Path $Ninja -Parent);$env:PATH"

& $CMake -G Ninja `
    -DCMAKE_MAKE_PROGRAM="$Ninja" `
    "-DCMAKE_BUILD_TYPE=$Config" `
    -S "$Here" `
    -B "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Ninja -C "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=> $BuildDir\jujube_h750.{elf,bin,hex}"
