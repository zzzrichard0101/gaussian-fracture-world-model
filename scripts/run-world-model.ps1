param(
    [Parameter(Position=0)]
    [ValidateSet('bootstrap','run','rank','evaluate','action-adapter','belief-state','python','test')]
    [string]$Command = 'run',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $PSScriptRoot 'world-model.sh'
$taskLinuxScript = (& wsl.exe -d Ubuntu -- wslpath -u $taskScript).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve world-model script inside WSL.' }
& wsl.exe -d Ubuntu -- bash $taskLinuxScript $Command @Arguments
exit $LASTEXITCODE
