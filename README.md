# KOFA Kore2 Battery SOH — Deployment Layer

CRISP-DM Phase 6: Deployment  
**Project:** Kore2 fleet early-warning system  
**Author:** Justin Muchina Komo

---

## Project Structure

```
kofa_soh_deployment/
├── app.py                         # Streamlit dashboard (run this)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── └── Kore2_battery_performance.csv
├── models/                        # Auto-created by train_model.py
│   ├── rf_classifier.pkl
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── summary.csv
│   └── label_map.csv
└── scripts/
    ├── train_model.py             # One-time training script
    └── screen_fleet.py            # Reusable pipeline function
```

---

## Quickstart (Local)

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Train & save models
python3 scripts/train_model.py

# 3 — Launch dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io → New app
3. Point to your repo, set main file = `app.py`
4. Done — public URL generated automatically

---

## screen_fleet() — use in any scheduled job

```python
import pandas as pd
from scripts.screen_fleet import screen_fleet

df = pd.read_csv("new_telemetry.csv", parse_dates=["timestamp"])
result = screen_fleet(df)
print(result[result.flagged])   # pull these batteries
```

## score_reading() — live single-reading API

```python
from scripts.screen_fleet import score_reading

result = score_reading(
    speed_kph=38, voltage_V=45.2,
    current_A=30, soc_percent=22, ambient_temp_C=27
)
# {'prediction': 'Faulty', 'probability': 0.99}
```
