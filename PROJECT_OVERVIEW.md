# AirSense project overview

## Objective

Build a clear, attractive dashboard that demonstrates practical air-quality data collection, cleaning, transformation, integration, exploratory analysis, and preprocessing. The app uses historical city and station measurements from the Kaggle Air Quality Data in India archive.

## Dataset

Kaggle source: [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The bundled archive contains:

| CSV | Data level | How AirSense uses it |
|---|---|---|
| `city_day.csv` | Daily city readings | Main city/date AQI series and dashboard source |
| `city_hour.csv` | Hourly city readings | Aggregated to daily city/date pollutant summaries |
| `station_day.csv` | Daily station readings | Joined to station metadata, then summarized by city/date |
| `station_hour.csv` | Hourly station readings | Summarized by city/date for integration and shown in the file explorer |
| `stations.csv` | Station metadata | Maps station identifiers to city names |

## Project workflow

1. **Collection:** the app reads the bundled Kaggle ZIP and inventories all CSV tables. `scripts/aqi_pipeline.py` also includes functions for CSV upload or URL-based CSV collection.
2. **Cleaning:** remove duplicate records; parse dates; coerce pollutant measurements and AQI to numeric values; mark invalid negative or infinite measurements as missing.
3. **Transformation:** derive year, month, day, weekday, and season fields; create AQI category labels when the source does not provide them.
4. **Integration:** use `stations.csv` to connect station readings to city names; aggregate hourly and station readings by city/date; left-join the companion measurements to daily city readings.
5. **Exploratory data analysis:** visualize AQI history, distributions, city comparisons, pollutant readings, and source-table completeness.
6. **Preprocessing:** the reusable pipeline imputes numeric/categorical gaps, standardizes numeric features, and one-hot encodes categorical features. It excludes AQI-derived bucket labels from model features to reduce target leakage.

## App sections

- **Overview:** selected city’s latest AQI, pollutant levels, and historical chart.
- **Dataset files:** source-table previews, shapes, schemas, missing values, and CSV downloads.
- **Forecast:** a historical rolling trend. It is explicitly not a future prediction.
- **Map view:** city AQI markers on an approximate India outline, with hover details.
- **Analytics:** historical charts, category distribution, file inventory, and integrated-data export.
- **Sensors / alerts / learn / about:** interface demonstrations and project information.

## Important limitations

The Kaggle data is historical and does not update in real time. Sensor statuses and alert controls are illustrative UI elements. Approximate city coordinates are used for map visualization. If deployed to a hosted service, its memory limits may be relevant because `station_hour.csv` is the largest archive table.
