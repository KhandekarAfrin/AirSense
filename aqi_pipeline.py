"""Reusable AQI data collection, cleaning, integration and preprocessing steps."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_HINTS = {
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene", "AQI", "Latitude", "Longitude",
}


def read_csv_upload(uploaded_file) -> pd.DataFrame:
    """Read a CSV from Streamlit's UploadedFile or a file-like object."""
    return pd.read_csv(uploaded_file)


def fetch_csv_url(url: str, timeout: int = 30) -> pd.DataFrame:
    """Collect CSV data from a public URL (for example, a raw data endpoint)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and standardize common column spellings."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    aliases = {
        "city_name": "City", "date": "Date", "aqi": "AQI", "aqi_bucket": "AQI_Bucket",
        "pm2_5": "PM2.5", "pm2.5": "PM2.5", "pm10": "PM10", "no2": "NO2",
        "so2": "SO2", "o3": "O3", "nox": "NOx", "nh3": "NH3",
    }
    rename = {c: aliases[c.strip().lower()] for c in out.columns if c.strip().lower() in aliases}
    return out.rename(columns=rename)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean duplicates, types and invalid numeric values; preserve missing values for imputation."""
    out = normalize_column_names(df)
    report = {"input_rows": len(out), "duplicate_rows_removed": int(out.duplicated().sum())}
    out = out.drop_duplicates().copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False)
    for col in out.columns:
        if col in NUMERIC_HINTS or col in {"Year", "Month", "Day"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]), col] = np.nan
    # AQI and pollutant concentrations cannot be negative.
    nonnegative = [c for c in out.columns if c in NUMERIC_HINTS and c not in {"Latitude", "Longitude"}]
    for col in nonnegative:
        out.loc[out[col] < 0, col] = np.nan
    report["rows_after_deduplication"] = len(out)
    report["missing_cells_after_cleaning"] = int(out.isna().sum().sum())
    return out, report


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features and AQI categories where source fields allow."""
    out = df.copy()
    if "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        out["Year"] = out["Date"].dt.year
        out["Month"] = out["Date"].dt.month
        out["Day"] = out["Date"].dt.day
        out["DayOfWeek"] = out["Date"].dt.day_name()
        out["Season"] = out["Month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"})
    if "AQI" in out.columns and "AQI_Bucket" not in out.columns:
        out["AQI_Bucket"] = pd.cut(
            out["AQI"], bins=[-np.inf, 50, 100, 200, 300, 400, np.inf],
            labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        ).astype("string")
    return out


def integrate_data(base: pd.DataFrame, extra: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    """Integrate an optional companion CSV by shared City/Date keys, or append matching schemas."""
    if extra is None or extra.empty:
        return base.copy(), "No companion dataset supplied."
    left, right = normalize_column_names(base), normalize_column_names(extra)
    keys = [k for k in ("City", "Date") if k in left.columns and k in right.columns]
    if "Date" in keys:
        left["Date"] = pd.to_datetime(left["Date"], errors="coerce")
        right["Date"] = pd.to_datetime(right["Date"], errors="coerce")
    if keys:
        merged = left.merge(right, on=keys, how="left", suffixes=("", "_extra"), validate="many_to_many")
        return merged, f"Left-joined on {', '.join(keys)}; unmatched rows remain missing."
    if set(left.columns) == set(right.columns):
        return pd.concat([left, right], ignore_index=True), "Appended rows because both files share the same schema."
    return left, "Could not find shared City/Date keys or matching schemas; kept primary dataset."


def build_preprocessor(df: pd.DataFrame, target: Optional[str] = "AQI") -> tuple[Pipeline, list[str], list[str], list[str]]:
    """Build a train-ready sklearn transformer; excludes target/date and all-empty columns."""
    excluded = {target, "Date", "AQI_Bucket"}
    features = [c for c in df.columns if c not in excluded and not df[c].isna().all()]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop", verbose_feature_names_out=False)
    return preprocessor, numeric, categorical, features


def preprocess_features(df: pd.DataFrame, target: Optional[str] = "AQI") -> tuple[pd.DataFrame, Pipeline, list[str], list[str]]:
    """Fit preprocessing on available feature rows and return an encoded/scaled frame."""
    prep, numeric, categorical, features = build_preprocessor(df, target)
    if not features:
        return pd.DataFrame(index=df.index), prep, numeric, categorical
    matrix = prep.fit_transform(df[features])
    names = prep.get_feature_names_out()
    return pd.DataFrame(matrix, columns=names, index=df.index), prep, numeric, categorical


def make_demo_data() -> pd.DataFrame:
    """Small deterministic sample for exploring the app before downloading Kaggle data."""
    rng = np.random.default_rng(42)
    cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad"]
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    rows = []
    for city_i, city in enumerate(cities):
        for i, date in enumerate(dates):
            season = 1.25 if date.month in (10, 11, 12, 1) else 0.75
            pm25 = max(4, rng.normal(62 + city_i * 10, 20) * season)
            pm10 = max(8, pm25 * rng.uniform(1.2, 2.1))
            aqi = max(12, pm25 * rng.uniform(1.5, 2.5))
            rows.append({"City": city, "Date": date, "PM2.5": round(pm25, 1), "PM10": round(pm10, 1), "NO2": round(max(2, rng.normal(24 + city_i * 2, 8)), 1), "CO": round(max(.1, rng.normal(.8 + city_i * .1, .3)), 2), "AQI": round(aqi, 1)})
    frame = pd.DataFrame(rows)
    # Deliberately add gaps and a couple duplicates so the cleaning workflow is visible.
    frame.loc[[7, 88, 311, 720], "PM2.5"] = np.nan
    frame = pd.concat([frame, frame.iloc[[12, 202]]], ignore_index=True)
    return frame
