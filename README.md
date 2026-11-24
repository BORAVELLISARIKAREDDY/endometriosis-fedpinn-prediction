# Endometriosis Risk Prediction — Research Prototype

An interactive Streamlit dashboard for a research demonstration using `advanced_endometriosis_dataset.csv`.

## Repository files

- `app.py` — Streamlit interface and 9-feature neural-network baseline.
- `requirements.txt` — pinned Python dependencies.
- `advanced_endometriosis_dataset.csv` — required dataset file (must be uploaded to the repository root).

## Important implementation note

The supplied legacy H5 model declares an input shape of **5 features**, while the supplied dataset and intended prediction feature list contain **9 features**. The old Flask API also uses mismatched field names. The original training script's custom physics-loss call is inconsistent. For safety and reproducibility, this app does not silently use that incompatible artifact. On first launch it trains a matching 9-feature neural-network baseline from the included CSV and caches the model in the running app.

This baseline is **not a verified FedPINN implementation**: the supplied custom loss does not implement a demonstrated physics constraint. Do not claim the deployed baseline is a clinically validated digital twin or diagnostic system.

## Deploy on Streamlit Community Cloud

1. Ensure `advanced_endometriosis_dataset.csv` is present in the repository root.
2. Visit https://share.streamlit.io and sign in with GitHub.
3. Choose **Create app**, select this repository, branch `main`, and entrypoint `app.py`.
4. In Advanced settings, select Python **3.11**.
5. Deploy. First startup trains the baseline and may take longer than later launches.

## Data and medical disclaimer

The UI exposes ultrasound/MRI numeric codes exactly as stored because no code legend was supplied. This is an educational/research prototype only, not medical advice, diagnosis, screening, or a substitute for a clinician. Before publishing the CSV, verify that it contains no identifiable or confidential patient information. The app must not be used to make medical decisions.
