param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [string]$Sd = ""
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
    -DXIP_BUILD=ON `
    -DMNASNET_ONLY=OFF `
    -DJUJUBE_RAM_ONLY=ON `
    -S "$Here" `
    -B "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Ninja -C "$BuildDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=> $BuildDir\jujube_h750_flash.hex"
Write-Host "=> $BuildDir\tflm_app.bin"

if ($Sd -ne "") {
    if (-not (Test-Path $Sd)) {
        throw "SD path not found: $Sd"
    }
    Copy-Item -LiteralPath (Join-Path $BuildDir "tflm_app.bin") -Destination (Join-Path $Sd "tflm_app.bin") -Force
    New-Item -ItemType File -Path (Join-Path $Sd "REPROGRAM") -Force | Out-Null
    Write-Host "=> copied tflm_app.bin to $Sd and set REPROGRAM flag"
}
