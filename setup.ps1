# Portable setup: downloads Python + dependencies INTO THIS FOLDER. Nothing is installed system-wide.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location $PSScriptRoot

$Py     = Join-Path $PSScriptRoot 'python\python.exe'

# All download addresses live in sources.json (editable): if python.org / pip / a package index ever changes,
# edit that file, no code changes needed. Lists are tried in order; built-in defaults are used if the file is bad.
$Src = @{
    python_embed_urls = @('https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip')
    get_pip_urls      = @('https://bootstrap.pypa.io/get-pip.py')
    pip_index_urls    = @('')
}
try {
    $j = Get-Content (Join-Path $PSScriptRoot 'sources.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($k in @($Src.Keys)) { if ($j.$k -and @($j.$k).Count -gt 0) { $Src[$k] = @($j.$k) } }
} catch { Write-Host 'sources.json not readable - using built-in addresses.' }

function Get-FirstWorking([string[]]$Urls, [string]$Out) {
    foreach ($u in $Urls) {
        try { Invoke-WebRequest $u -OutFile $Out -UseBasicParsing; return }
        catch { Write-Host "  could not download $u : $($_.Exception.Message)" }
    }
    throw "None of the download addresses worked (see sources.json): $($Urls -join ', ')"
}

if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }

if (-not (Test-Path $Py)) {
    Write-Host '[1/5] Downloading portable Python (~10 MB) ...'
    $PyZip = 'python-embed.zip'
    Get-FirstWorking $Src.python_embed_urls $PyZip
    Expand-Archive $PyZip -DestinationPath 'python' -Force
    Remove-Item $PyZip
    # enable site-packages (required for pip)
    $pth = Get-ChildItem 'python' -Filter 'python*._pth' | Select-Object -First 1
    (Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' | Set-Content $pth.FullName
} else { Write-Host '[1/5] Python already present.' }

& $Py -c "import pip" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '[2/5] Installing pip (into the python folder) ...'
    Get-FirstWorking $Src.get_pip_urls 'get-pip.py'
    & $Py 'get-pip.py' --no-warn-script-location --no-cache-dir
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install pip.' }
    Remove-Item 'get-pip.py'
} else { Write-Host '[2/5] pip already present.' }

Write-Host '[3/5] Installing libraries (speech engine + Qt window) ...'
$ok = $false
foreach ($idx in $Src.pip_index_urls) {
    $extra = @(); if ($idx) { $extra = @('--index-url', $idx) }
    & $Py -m pip install -r requirements.txt --no-warn-script-location --no-cache-dir @extra
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Write-Host "  package index '$idx' failed, trying the next one ..."
}
if (-not $ok) { throw 'Failed to install libraries.' }

# The Qt package ships many parts this program never uses (QML, PDF, designer tools, translations ...).
# Removing them saves a lot of disk space. If the trimmed Qt fails to import, it is reinstalled untrimmed.
Write-Host '[4/5] Trimming unused Qt parts ...'
$qt = Join-Path $PSScriptRoot 'python\Lib\site-packages\PySide6'
if (Test-Path $qt) {
    foreach ($d in 'qml','translations','include','typesystems','glue','scripts','examples','metatypes') {
        Remove-Item (Join-Path $qt $d) -Recurse -Force -ErrorAction SilentlyContinue
    }
    foreach ($d in 'qmltooling','sqldrivers','designer','scenegraph') {
        Remove-Item (Join-Path $qt "plugins\$d") -Recurse -Force -ErrorAction SilentlyContinue
    }
    foreach ($pat in '*.exe','*.pyi','Qt6Quick*.dll','Qt6Qml*.dll','Qt6Pdf*.dll','Qt6Designer*.dll','Qt6Help*.dll',
                     'Qt6Sql*.dll','Qt6Test*.dll','Qt6UiTools*.dll','opengl32sw.dll',
                     'QtQ*.pyd','QtPdf*.pyd','QtDesigner.pyd','QtHelp.pyd','QtSql.pyd','QtTest.pyd','QtUiTools.pyd') {
        Get-ChildItem $qt -Filter $pat -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}
& $Py -c "from PySide6 import QtCore, QtGui, QtWidgets, QtNetwork" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Trimmed Qt did not start - reinstalling it without trimming ...'
    & $Py -m pip install --force-reinstall --no-deps --no-cache-dir --no-warn-script-location PySide6-Essentials shiboken6
    if ($LASTEXITCODE -ne 0) { throw 'Failed to reinstall Qt.' }
}

Write-Host '[5/5] Verifying ...'
& $Py -c "import faster_whisper, sounddevice, keyboard, pyperclip, numpy; from PySide6 import QtWidgets, QtNetwork; print('OK')"
if ($LASTEXITCODE -ne 0) { throw 'Import check failed.' }
Write-Host 'Done.'
