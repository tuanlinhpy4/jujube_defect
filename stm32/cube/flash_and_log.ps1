param(
    [int]$WaitSeconds = 4,
    [string]$Elf = "",
    [string]$FlashImage = ""
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Elf -eq "") {
    $Elf = Join-Path $Here "build\jujube_h750.elf"
}
if ($FlashImage -eq "") {
    $SiblingHex = Join-Path (Split-Path -Parent $Elf) "jujube_h750_flash.hex"
    if (Test-Path $SiblingHex) {
        $FlashImage = $SiblingHex
    } else {
        $FlashImage = $Elf
    }
}

$Bundle = Join-Path $env:LOCALAPPDATA "stm32cube\bundles"
$Programmer = Join-Path $Bundle "programmer\2.22.0+st.1\bin\STM32_Programmer_CLI.exe"
$Nm = Join-Path $Bundle "gnu-tools-for-stm32\14.3.1+st.2\bin\arm-none-eabi-nm.exe"

foreach ($Path in @($Programmer, $Nm, $Elf, $FlashImage)) {
    if (-not (Test-Path $Path)) {
        throw "Missing file/tool: $Path"
    }
}

Write-Host "=== flash ==="
$flash = & $Programmer -c port=SWD mode=UR -w "$FlashImage" -rst 2>&1
$flash | Where-Object { $_ -match "verified|error|reset" }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$nmOut = & $Nm "$Elf"
$line = $nmOut | Where-Object { $_ -match "\s+g_logbuf$" } | Select-Object -First 1
if (-not $line) {
    throw "Cannot find g_logbuf symbol in $Elf"
}
$addr = "0x" + (($line -split "\s+")[0])

Write-Host "=== run ${WaitSeconds}s then read log @ $addr ==="
Start-Sleep -Seconds $WaitSeconds

$tmp = Join-Path $env:TEMP "jujube_log.bin"
& $Programmer -c port=SWD mode=HOTPLUG -r "$addr" 0x8000 "$tmp" *> $null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$bytes = [System.IO.File]::ReadAllBytes($tmp)
$clean = @($bytes | Where-Object { $_ -ne 0 })
$text = [System.Text.Encoding]::ASCII.GetString([byte[]]$clean)

Write-Host "----------------------------------------"
Write-Host $text
Write-Host "----------------------------------------"
