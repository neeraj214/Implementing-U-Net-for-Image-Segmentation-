$ErrorActionPreference = "Stop"

# Activate virtual environment
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Warning "Virtual environment activation script not found at .\venv\Scripts\Activate.ps1"
}

Write-Host ">>> Verifying GPU..." -ForegroundColor Cyan
python setup\check_gpu.py

Write-Host "`n>>> Phase 1: Running single image demo..." -ForegroundColor Cyan
python src\single_image_demo.py

Write-Host "`n>>> Phase 2: Loading dataset..." -ForegroundColor Cyan
python src\load_dataset.py

Write-Host "`n>>> Phase 3: Checking U-Net model summary..." -ForegroundColor Cyan
python src\unet_model.py

Write-Host "`n>>> Phase 4: Starting Training (This may take 30-60 minutes)..." -ForegroundColor Magenta
python src\train.py

Write-Host "`n>>> Phase 5: Evaluating model..." -ForegroundColor Cyan
python src\evaluate.py

Write-Host "`n>>> Phase 6: Plotting training curves..." -ForegroundColor Cyan
python src\plot_curves.py

Write-Host "`n>>> Phase 7: Generating visualizations..." -ForegroundColor Cyan
python src\visualize.py

Write-Host "`n>>> All phases completed successfully!" -ForegroundColor Green
