# AirSense — Air Quality Data Explorer

AirSense is a Streamlit project for exploring historical air-quality data from India. It combines an AQI dashboard, interactive charts, a CSV data explorer, and reusable Python preprocessing code.

## Project files

- `app.py` — Streamlit app and visual analysis
- `aqi_pipeline.py` — reusable data collection, cleaning, transformation, integration, and preprocessing functions
- `PROJECT_OVERVIEW.md` — project objective, dataset files, and workflow
- `requirements.txt` — Python dependencies

## Run locally

Install Python 3.10 or newer, open PowerShell in the project folder, and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

If `data/air_quality_data.zip` is present, the app reads it. Otherwise, it downloads the public Kaggle archive automatically. There is no upload control. The **Dataset files** page previews each source CSV and lets you download it; **Overview** and **Analytics** show charts.

## Kaggle source and data workflow

Source: [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india). The app uses `city_day.csv` for the daily AQI series, integrates city/hour and station/day/hour readings by city and date, and maps station readings with `stations.csv`. `aqi_pipeline.py` demonstrates cleaning, date feature creation, missing-value imputation, scaling, and categorical encoding.

## Upload the project through GitHub’s website

The Kaggle ZIP is about 73 MB, so it is excluded from the GitHub repository; the deployed app downloads it from Kaggle when it starts. In GitHub, create an empty repository, choose **Add file → Upload files**, and upload these small project files: `app.py`, `aqi_pipeline.py`, `PROJECT_OVERVIEW.md`, `README.md`, and `requirements.txt`. Commit the files, then deploy `app.py` from [Streamlit Community Cloud](https://share.streamlit.io/).

The source data is historical, not live. The forecast page describes past trends and does not predict future AQI. Sensor and alert controls are demonstrations.
