# Demand forecast model (AI-08)

The owner dashboard's "Dự báo nhu cầu 7 ngày tới" uses **Chronos-Bolt tiny** from Amazon (Apache-2.0, about 9M parameters, revision `a0e552de83495b5c28c14c71c374f3e33280b340`). It is a time-series transformer trained in advance. It forecasts zero-shot, with no training on GoatSports data.

## How it runs

- **Build:** the Docker stage `demand-forecast-model` installs CPU PyTorch and `chronos-forecasting`, then runs `scripts/export_chronos_bolt.py`.
  - The script exports only the network (patch embedding → T5 encoder → decoder → quantile head) to `chronos-bolt-tiny.onnx`, plus a `.json` config.
  - It checks `app/infrastructure/forecast/chronos_onnx.py` against the original PyTorch pipeline, for 64-hour and 168-hour horizons. If they differ by more than 1e-4, the build fails.
  - The runtime image copies only these two files. It has no PyTorch.
- **Runtime:** `ChronosBoltForecaster` uses onnxruntime and numpy. It covers the steps that cannot be exported:
  - instance normalisation with NaN-aware mean and std;
  - 16-hour patching with an observed-mask;
  - inverse scaling;
  - the long-horizon rollout. Each of the 9 quantile paths is appended to the context and forecast again, then the 81 samples are reduced back to 9 quantiles. This is exactly what `ChronosBoltPipeline` does.
- **Data:** venue-service `GET /internal/owner-demand/{venueId}?ownerId=` returns hourly occupancy for the last 2048 hours (court-hours held divided by active courts). It also checks that the caller owns the venue.
- **Output:** p10 / p50 / p90 for each of the next 168 hours. Hours after the first 64 are flagged `extended` (rolled out, less certain). Crossing quantiles are re-sorted per hour.
- **Honesty checks:**
  - A backtest forecasts the last week from earlier data only. It reports the MAE against the "same as last week" baseline and the share of hours inside p10–p90.
  - With less than one day of history the API returns 422.

## Local development

```bash
uv venv .forecast-export --python 3.12
uv pip install --python .forecast-export --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
uv pip install --python .forecast-export -r requirements-forecast-export.txt
.forecast-export/Scripts/python scripts/export_chronos_bolt.py models
```

`models/` is git-ignored. Without the files, `GET /api/v1/ai/demand-forecast/venues/{id}` returns "capability unavailable" and every other feature keeps working.

To change the model or revision, edit `REVISION` in the export script, review the license, and rebuild. The export fails if the ONNX output no longer matches PyTorch.
