# Spotify MLOps Demo Helper

Write-Host "--- Spotify MLOps Demo Setup ---" -ForegroundColor Green
Write-Host "1. Ensure you have 'ngrok http 8000' running in another window."
Write-Host "2. Copy the https URL from that window (e.g. https://xyz.ngrok-free.app)"
Write-Host ""

$url = Read-Host "Paste Backend Ngrok URL here"

if ([string]::IsNullOrWhiteSpace($url)) {
    Write-Host "URL cannot be empty." -ForegroundColor Red
    exit 1
}

# Remove trailing slash if present
if ($url.EndsWith("/")) {
    $url = $url.Substring(0, $url.Length - 1)
}

Write-Host "Setting VITE_API_URL to $url" -ForegroundColor Yellow
$env:VITE_API_URL = $url

Write-Host "Starting Frontend..." -ForegroundColor Green
cd frontend
# Using --host to ensure it listens on all interfaces (needed for some setups)
npm run dev -- --host
