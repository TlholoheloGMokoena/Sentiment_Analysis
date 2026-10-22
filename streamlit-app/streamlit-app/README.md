# Customer Sentiment Analysis & Insights Dashboard (Streamlit)

Upload customer feedback (CSV or typed text) and get sentiment results, charts, written insights and a report.

## Files
- `app.py` – Streamlit interface (navigation, filters, charts, exports)
- `sentiment.py` – preprocessing, sentiment model, data-quality checks, analytics, insights
- `data/` – sample datasets (`sample-reviews-250.csv` loads on start) and three labelled test sets
- `requirements.txt`, `.streamlit/config.toml` – dependencies and dark theme

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repository and upload everything in this folder (keep `app.py` at the top level).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. The first build installs `requirements.txt` and takes a minute or two.

## Notes
- Each new upload **replaces** the current dataset; nothing is stored on disk.
- Metrics (accuracy, precision, recall, F1) appear only when the file has a labelled `sentiment` or `label` column.
- To swap the model, replace `predict()` in `sentiment.py` (it must return score, label and confidence).
