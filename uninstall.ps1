# dlm uninstaller for Windows — remove the binary install.ps1 dropped and clean PATH.
#
#   irm https://raw.githubusercontent.com/vedantnimbarte/dlm/main/uninstall.ps1 | iex
#
# Env:
#   DLM_INSTALL_DIR   install location to clean (default: %LOCALAPPDATA%\Programs\dlm)

$ErrorActionPreference = 'Stop'

$Bin = 'dlm.exe'
$InstallDir = if ($env:DLM_INSTALL_DIR) { $env:DLM_INSTALL_DIR }
              else { Join-Path $env:LOCALAPPDATA 'Programs\dlm' }
$target = Join-Path $InstallDir $Bin

if (Test-Path $target) {
    Remove-Item $target -Force
    Write-Host "Removed $target"
} else {
    $found = Get-Command dlm -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "No dlm.exe in $InstallDir, but found one at $($found.Source) — remove it manually."
    } else {
        Write-Host "dlm is not installed (nothing at $target)."
    }
}

# Remove our install dir from the user PATH (exact-segment match, so a sibling
# like ...\dlm2 is left alone).
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$segments = @($userPath -split ';' | Where-Object { $_ -ne '' })
if ($segments -contains $InstallDir) {
    $newPath = ($segments | Where-Object { $_ -ne $InstallDir }) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Host "Removed $InstallDir from your user PATH (open a new terminal to pick it up)."
}

# Clean up the install dir if it's now empty.
if ((Test-Path $InstallDir) -and -not (Get-ChildItem $InstallDir -Force)) {
    Remove-Item $InstallDir -Force
}
