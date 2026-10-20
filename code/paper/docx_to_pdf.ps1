param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf
)
# Opens the manuscript in Word, updates its fields and saves it as PDF.
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($InputDocx, $false, $true)
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    $doc.SaveAs([ref]$OutputPdf, [ref]17)
    $doc.Close([ref]0)
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
