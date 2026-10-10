# KisanDekhe - Crop Stress Early Warning

Ten-day early warning for groundnut and other crop stress in Tamil Nadu's Kongu belt and Salem, built from free satellite and weather data.

## How it works
1. **Satellite:** MODIS NDVI (250 m) measures crop greenness, compared against a normal curve for each date.
2. **Weather:** CHIRPS rainfall and ERA5-Land temperature.
3. **Model:** XGBoost forecasts whether stress will occur in the next 10 days, using only data available up to that day.
4. **Advice:** Risk level, the main reasons, and timed groundnut tasks from TNAU Agritech guidance.

## Results (held-out test period)
Recall on days without visible stress: see `metrics.json`. A simple "stressed today" rule catches none of these early warnings.

## Limitations
- Stress labels are rule-based proxies, not field-verified.
- Satellite NDVI is 250 m, so results describe a neighbourhood, not a single field.
- Sowing dates and farm details are assumptions until verified with a farmer.

## Run it
pip install -r requirements.txt
python batch_pull.py    # needs a Google Earth Engine account
python add_farms.py
python train_all.py
python export_dashboard.py
Open dashboard/index.html (or run python -m http.server in dashboard/).

## Data sources
CHIRPS, ERA5-Land, MODIS MOD13Q1 via Google Earth Engine; TNAU Agritech Portal for groundnut guidance.