param(
    [Parameter(Position=0)]
    [ValidateSet('bootstrap','doctor','smoke','fetch-asset','asset-check','first-impact','impact-report','analyze-separation','replay-transfer','stateful-probe','resume-impact','checkpoint-probe','history-pilot','fracture-probe','cohesive-probe','cohesive-inverse','history-refinement','direct-benchmark','joint-damage-material','response-diversity','active-probe','compliant-pilot','synthetic-filter-benchmark','test','python','upstream')]
    [string]$Command = 'doctor',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $PSScriptRoot 'wsl.sh'
$taskLinuxScript = (& wsl.exe -d Ubuntu -- wslpath -u $taskScript).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the project path inside WSL.' }
& wsl.exe -d Ubuntu -- bash $taskLinuxScript $Command @Arguments
exit $LASTEXITCODE
