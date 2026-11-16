from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(page_title="AirSense | Breathe smarter", page_icon="🌬️", layout="wide", initial_sidebar_state="expanded")

KAGGLE_ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/rohanrao/air-quality-data-in-india"

PALETTE = {"mint": "#66d8b6", "purple": "#ad91f5", "yellow": "#f1c968", "orange": "#ed995f", "red": "#ed777b", "blue": "#72aaff"}
POLLUTANTS = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
POLLUTANT_UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "O3": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "CO": "mg/m³"}
POLLUTANT_COLORS = dict(zip(POLLUTANTS, [PALETTE["purple"], PALETTE["mint"], PALETTE["yellow"], PALETTE["orange"], PALETTE["blue"], "#d994d7"]))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#091211;--surface:#13211f;--surface2:#172925;--border:rgba(190,224,210,.11);--text:#ecf5f0;--muted:#8ba39a;--mint:#66d8b6}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 83% -13%,rgba(48,125,102,.16),transparent 36%),#091211;color:#ecf5f0}
[data-testid="stHeader"]{background:rgba(9,18,17,.72)}
[data-testid="stSidebar"]{background:#0c1716;border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:#ecf5f0}
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1500px}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.04em}
h1{font-weight:800!important;font-size:2rem!important}h2{font-weight:700!important;font-size:1.2rem!important}h3{font-weight:650!important;font-size:1rem!important}
.brand{display:flex;align-items:center;gap:10px;margin:4px 0 28px}.brand-icon{height:38px;width:38px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(145deg,#7ce3bd,#35ae8c);color:#09251d;font-size:21px;box-shadow:0 5px 20px #41c59c3b}.brand-name{font:800 20px Manrope,sans-serif;color:#ecf5f0;line-height:1}.brand-name span{color:#66d8b6}.brand-tag{font-size:8px;color:#8ba39a;letter-spacing:1.5px;margin-top:5px}
.eyebrow{color:#8ba39a;font-size:9px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:5px}.muted{color:#8ba39a;font-size:11px}
.card{background:linear-gradient(145deg,rgba(23,41,37,.94),rgba(16,30,27,.94));border:1px solid var(--border);border-radius:16px;padding:18px 20px;box-shadow:0 10px 32px #0002;margin-bottom:12px}.hero-card{background:radial-gradient(ellipse at 28% 75%,rgba(41,129,103,.14),transparent 48%),linear-gradient(145deg,rgba(23,41,37,.96),rgba(16,30,27,.96));border:1px solid var(--border);border-radius:18px;padding:20px 22px;box-shadow:0 12px 38px #0002}[data-testid="stVerticalBlockBorderWrapper"]{background:radial-gradient(ellipse at 28% 75%,rgba(41,129,103,.14),transparent 48%),linear-gradient(145deg,rgba(23,41,37,.96),rgba(16,30,27,.96));border:1px solid var(--border);border-radius:18px;padding:16px 18px;box-shadow:0 12px 38px #0002}
.metric{background:rgba(23,41,37,.82);border:1px solid var(--border);border-radius:14px;padding:14px 15px;min-height:100px}.metric-label{font-size:9px;color:#8ba39a;margin-bottom:7px}.metric-value{font:700 22px Manrope,sans-serif;letter-spacing:-.04em}.metric-unit{font:500 10px 'DM Sans',sans-serif;color:#8ba39a;margin-left:4px}.metric-note{font-size:8px;color:#71867d;margin-top:4px}
.status-pill{display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(102,216,182,.2);background:rgba(102,216,182,.08);color:#66d8b6;border-radius:20px;padding:6px 10px;font-size:9px}.dot{width:6px;height:6px;border-radius:50%;background:#66d8b6;box-shadow:0 0 8px #66d8b6}.data-note{border:1px solid rgba(102,216,182,.17);background:rgba(102,216,182,.07);border-radius:12px;padding:11px 14px;color:#cde7dc;font-size:10px;margin:10px 0 14px}.stButton>button{border-radius:9px;border:1px solid var(--border);background:#172925;color:#ecf5f0;font-size:10px;min-height:35px}.stButton>button:hover{border-color:#66d8b6;color:#66d8b6}.stDownloadButton>button{border-radius:9px;background:#55cda8;color:#07251c;border:0;font-weight:700;font-size:10px}.stDownloadButton>button:hover{background:#72dfba;color:#07251c}.stSelectbox label,.stSlider label,.stTextInput label,.stTextArea label{font-size:10px;color:#9db0a8}
[data-testid="stMetric"]{background:#13211f;border:1px solid var(--border);padding:13px 14px;border-radius:13px}.stMetric label{font-size:9px!important;color:#8ba39a!important}.stMetric [data-testid="stMetricValue"]{font:700 22px Manrope,sans-serif;color:#ecf5f0}
div[data-testid="stExpander"]{background:#13211f;border:1px solid var(--border);border-radius:12px}div[data-testid="stExpander"] summary{font-size:10px}
.stRadio [role=radiogroup]{gap:5px}.stRadio [role=radio]{border-radius:9px;padding:8px 10px}.stRadio [role=radio] p{font-size:11px}.stFileUploader{background:rgba(23,41,37,.45);border:1px dashed rgba(102,216,182,.3);border-radius:12px;padding:7px}
footer{visibility:hidden}.stCaption{color:#71867d!important}
@media(max-width:800px){h1{font-size:1.6rem!important}.hero-card{min-height:unset}.card{padding:15px}}
</style>
""", unsafe_allow_html=True)


def aqi_category(value):
    if pd.isna(value): return "No reading", "#8ba39a"
    if value <= 50: return "Good", "#52d4a1"
    if value <= 100: return "Moderate", "#e7c45f"
    if value <= 150: return "Sensitive groups", "#ec9b58"
    if value <= 200: return "Unhealthy", "#ef777b"
    if value <= 300: return "Very unhealthy", "#bb83e5"
    return "Hazardous", "#df5d68"


def normalize_table(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Date" in frame:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    for col in POLLUTANTS + ["NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.loc[~np.isfinite(frame[col]), col] = np.nan
            frame.loc[frame[col] < 0, col] = np.nan
    return frame.drop_duplicates()


def load_kaggle_zip(upload):
    """Read every CSV from the Kaggle archive and integrate all AQI tables at city/date level."""
    tables = {}
    with zipfile.ZipFile(upload) as archive:
        paths = [p for p in archive.namelist() if p.lower().endswith(".csv") and not p.endswith("/")]
        for path in paths:
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            raw = pd.read_csv(io.BytesIO(archive.read(path)), low_memory=False)
            tables[name] = normalize_table(raw)
    if not tables:
        raise ValueError("No CSV files were found in that ZIP archive.")

    def find(prefix):
        return next((df.copy() for name, df in tables.items() if name.startswith(prefix)), None)

    daily = find("city_day")
    if daily is None:
        raise ValueError("The archive needs city_day.csv to show daily AQI readings.")
    if "City" not in daily or "Date" not in daily:
        raise ValueError("city_day.csv must include City and Date columns.")
    daily["City"] = daily["City"].astype("string").str.strip().str.title()
    daily = daily.dropna(subset=["Date"])
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    daily["Day"] = daily["Date"].dt.day
    daily["DayOfWeek"] = daily["Date"].dt.day_name()
    daily["Season"] = daily["Month"].map({12:"Winter",1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Post-monsoon"})
    if "AQI_Bucket" not in daily and "AQI" in daily:
        daily["AQI_Bucket"] = pd.cut(daily["AQI"],[-np.inf,50,100,200,300,400,np.inf],labels=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"])

    stations = find("stations")
    station_to_city = {}
    if stations is not None and {"StationId", "City"}.issubset(stations.columns):
        station_to_city = dict(zip(stations["StationId"].astype(str), stations["City"].astype(str).str.title()))

    companions = []
    numeric = [c for c in POLLUTANTS + ["NO", "NOx", "NH3", "Benzene", "Toluene", "Xylene"]]
    for table_name in ("city_hour", "station_day", "station_hour"):
        table = find(table_name)
        if table is None or "Date" not in table:
            continue
        if "City" not in table and "StationId" in table:
            table["City"] = table["StationId"].astype(str).map(station_to_city)
        if "City" not in table:
            continue
        table["City"] = table["City"].astype("string").str.strip().str.title()
        table = table.dropna(subset=["Date", "City"])
        metrics = [c for c in numeric if c in table and c != "AQI_Bucket"]
        if not metrics:
            continue
        # Hourly readings and station measurements are averaged to daily city-level values.
        agg = table.groupby(["City", "Date"], as_index=False)[metrics].mean()
        agg = agg.rename(columns={c: f"{table_name}_{c}" for c in metrics})
        companions.append(agg)

    integrated = daily.copy()
    for extra in companions:
        integrated = integrated.merge(extra, on=["City", "Date"], how="left")
    file_info = pd.DataFrame([{"CSV file": f"{name}.csv", "Rows": len(df), "Columns": len(df.columns), "Integrated": "Yes" if name == "stations" or name in ("city_day", "city_hour", "station_day", "station_hour") else "Read"} for name, df in tables.items()])
    cities = sorted(integrated["City"].dropna().unique().tolist())
    return integrated, file_info, cities, tables


def make_preview():
    rng = np.random.default_rng(19)
    dates = pd.date_range("2024-10-01", periods=100)
    rows = []
    for city, bias in [("Brooklyn", 0), ("Manhattan", 12), ("Queens", -7)]:
        for i, date in enumerate(dates):
            pm25 = max(5, 28 + bias + 12*np.sin(i/11) + rng.normal(0, 5))
            rows.append({"City":city,"Date":date,"AQI":round(pm25*1.65),"PM2.5":round(pm25,1),"PM10":round(pm25*1.7,1),"O3":round(20+rng.normal(0,4),1),"NO2":round(16+rng.normal(0,3),1),"SO2":round(5+rng.normal(0,1),1),"CO":round(.5+rng.normal(0,.1),2)})
    return pd.DataFrame(rows), pd.DataFrame(), ["Brooklyn","Manhattan","Queens"]


def gauge(value):
    label, color = aqi_category(value)
    # Use an inline SVG instead of Plotly's Indicator trace. This avoids blank
    # gauge panels in some Streamlit/Plotly browser combinations.
    number = float(value) if pd.notna(value) else 0.0
    radius = 77
    circumference = 2 * np.pi * radius
    progress = min(max(number / 300, 0), 1)
    dash_offset = circumference * (1 - progress)
    markup = f'''<div style="height:225px;display:flex;align-items:center;justify-content:center">
      <div style="width:210px;height:210px;position:relative;--aqi-color:{color}">
        <svg viewBox="0 0 210 210" width="210" height="210" aria-label="AQI {number:.0f}, {label}" role="img" style="transform:rotate(-90deg);overflow:visible">
          <circle cx="105" cy="105" r="{radius}" fill="none" stroke="rgba(190,224,210,.15)" stroke-width="11"/>
          <circle cx="105" cy="105" r="{radius}" fill="none" stroke="{color}" stroke-width="11" stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{dash_offset:.1f}"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#ecf5f0">
          <strong style="font:800 43px Manrope, sans-serif;line-height:1">{number:.0f}</strong>
          <span style="font:500 10px 'DM Sans',sans-serif;color:#8ba39a;letter-spacing:1.5px;margin-top:5px">AQI</span>
          <b style="font:600 11px 'DM Sans',sans-serif;color:{color};margin-top:8px">{label}</b>
        </div>
      </div>
    </div>'''
    return markup, label, color


def line_chart(data, x, y, title=None, color=None):
    fig = px.line(data, x=x, y=y, color_discrete_sequence=[color or PALETTE["mint"]], title=title)
    fig.update_traces(line=dict(width=2.5), fill="tozeroy", fillcolor="rgba(102,216,182,.08)", hovertemplate="%{x}<br>%{y:.1f}<extra></extra>")
    fig.update_layout(height=290, margin=dict(l=5,r=5,t=20,b=5), paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8ba39a",font=dict(family="DM Sans",size=10),xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(gridcolor="rgba(190,224,210,.09)",zeroline=False),showlegend=False)
    return fig


def section_header(kicker, title, subtitle=""):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1 style="margin:0 0 3px">{title}</h1><div class="muted" style="margin-bottom:18px">{subtitle}</div>', unsafe_allow_html=True)


def main():
    if "air_data" not in st.session_state:
        archive_path = Path(__file__).parent / "data" / "air_quality_data.zip"
        try:
            with st.spinner("Loading the Kaggle CSV files…"):
                if archive_path.exists():
                    archive_bytes = archive_path.read_bytes()
                else:
                    response = requests.get(KAGGLE_ARCHIVE_URL, timeout=(20, 240))
                    response.raise_for_status()
                    archive_bytes = response.content
                integrated, info, cities, tables = load_kaggle_zip(io.BytesIO(archive_bytes))
            st.session_state.air_data = integrated
            st.session_state.file_info = info
            st.session_state.cities = cities
            st.session_state.raw_tables = tables
            st.session_state.is_kaggle = True
        except Exception as exc:
            st.error(f"Could not load the Kaggle dataset: {exc}")
            st.info("Check the app's internet connection, then refresh the page.")
            st.stop()
    if "theme" not in st.session_state: st.session_state.theme = "Dark"

    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-icon">〰</div><div><div class="brand-name">Air<span>Sense</span></div><div class="brand-tag">BREATHE BETTER</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
        city_options = st.session_state.cities or ["All cities"]
        if st.session_state.get("selected_city") not in city_options:
            st.session_state.selected_city = city_options[0]
        city = st.selectbox("Location", city_options, key="selected_city")
        page = st.radio("NAVIGATION", ["Overview", "Dataset files", "Forecast", "Map view", "Analytics", "My sensors", "Alerts", "Learn", "About"], label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.is_kaggle:
            st.markdown('<div class="status-pill"><span class="dot"></span> Kaggle data connected</div>', unsafe_allow_html=True)
            st.caption(f"{len(st.session_state.raw_tables)} CSV files · Kaggle historical archive")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("AirSense · Breathe smarter, live cleaner.")

    data = st.session_state.air_data.copy()
    if "City" in data and city:
        data = data[data.City.astype(str).str.casefold() == str(city).casefold()]
    data = data.sort_values("Date") if "Date" in data else data
    has_kaggle = st.session_state.is_kaggle
    source_label = "Kaggle historical dataset"
    st.markdown(f'<div class="data-note">✓ &nbsp; Using <b>Air Quality Data in India</b> · {len(st.session_state.air_data):,} city-day records · {len(st.session_state.raw_tables)} CSV files loaded and integrated</div>', unsafe_allow_html=True)

    if page == "Dataset files":
        section_header("KAGGLE ARCHIVE", "Explore the dataset files.", "Browse the CSV tables bundled with AirSense. Select a file to inspect its rows and data quality.")
        raw_tables = st.session_state.raw_tables
        file_names = list(raw_tables.keys())
        chosen_file = st.selectbox("CSV file", file_names, format_func=lambda name: f"{name}.csv")
        frame = raw_tables[chosen_file]
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(frame):,}")
        m2.metric("Columns", f"{len(frame.columns):,}")
        m3.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
        st.markdown(f"#### {chosen_file}.csv")
        st.caption("Showing the first rows of the bundled source table. Download the complete CSV below.")
        preview_rows = st.select_slider("Preview rows", options=[25, 50, 100, 250, 500], value=100)
        st.dataframe(frame.head(preview_rows), use_container_width=True, height=420, hide_index=True)
        missing = frame.isna().sum().rename("Missing values").to_frame()
        missing = missing[missing["Missing values"] > 0].sort_values("Missing values", ascending=False)
        with st.expander("Column and missing-value summary"):
            schema = pd.DataFrame({"Column": frame.columns, "Data type": [str(t) for t in frame.dtypes], "Missing values": [int(frame[c].isna().sum()) for c in frame.columns]})
            st.dataframe(schema, use_container_width=True, hide_index=True)
        st.download_button(f"Download {chosen_file}.csv", frame.to_csv(index=False).encode("utf-8"), f"{chosen_file}.csv", "text/csv")
        st.markdown("#### Archive contents")
        st.dataframe(st.session_state.file_info, use_container_width=True, hide_index=True)

    # Overview
    elif page == "Overview":
        latest = data.dropna(subset=["AQI"]).iloc[-1] if "AQI" in data and data.AQI.notna().any() else None
        current_aqi = float(latest.AQI) if latest is not None else np.nan
        label, color = aqi_category(current_aqi)
        st.markdown(f'<div class="eyebrow">● &nbsp; YOUR PERSONAL AIRSPACE</div><h1>Your air, at a glance.</h1><p class="muted">A little more insight. A lot more peace of mind.</p>', unsafe_allow_html=True)
        if latest is not None:
            left, right = st.columns([1.12, .88], gap="medium")
            with left:
                with st.container(border=True):
                    st.markdown(f'<div class="eyebrow">YOUR AIR QUALITY · {city.upper()}</div>', unsafe_allow_html=True)
                    gauge_col, detail_col = st.columns([.9,1.1], vertical_alignment="center")
                    with gauge_col:
                        gauge_markup, _, _ = gauge(current_aqi)
                        st.markdown(gauge_markup, unsafe_allow_html=True)
                    with detail_col:
                        st.markdown(f'<div style="font:700 20px Manrope;color:{color}">{label}</div>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size:12px;line-height:1.6">{health_advice(current_aqi)}</p>', unsafe_allow_html=True)
                        st.caption(f"Latest source observation · {latest.Date:%d %b %Y}" if pd.notna(latest.Date) else source_label)
                        st.markdown('<span class="status-pill"><span class="dot"></span> Dataset connected</span>' if has_kaggle else '<span class="status-pill"><span class="dot"></span> Preview mode</span>', unsafe_allow_html=True)
            with right:
                with st.container(border=True):
                    st.markdown('<div class="eyebrow">YOUR ENVIRONMENT</div><h3>Pollutants in your air</h3>', unsafe_allow_html=True)
                    cols = st.columns(2)
                    for i, pollutant in enumerate(POLLUTANTS):
                        value = latest.get(pollutant, np.nan)
                        with cols[i % 2]:
                            st.markdown(metric_html(pollutant, value, POLLUTANT_UNITS[pollutant], "Kaggle daily reading" if has_kaggle else "Preview reading"), unsafe_allow_html=True)
            st.markdown('<div class="eyebrow" style="margin-top:16px">AIR QUALITY HISTORY</div><h2>Recent readings</h2>', unsafe_allow_html=True)
            pollutant_choice = st.selectbox("Chart pollutant", ["AQI"] + [p for p in POLLUTANTS if p in data], label_visibility="collapsed")
            chart_data = data.dropna(subset=[pollutant_choice]).tail(90)
            st.plotly_chart(line_chart(chart_data, "Date", pollutant_choice, color=PALETTE["mint"] if pollutant_choice == "AQI" else POLLUTANT_COLORS.get(pollutant_choice)), use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="eyebrow">ENVIRONMENT SNAPSHOT</div>', unsafe_allow_html=True)
            display_cols = [c for c in ["Date","City","AQI","AQI_Bucket"] + POLLUTANTS if c in data]
            st.dataframe(data[display_cols].tail(12).sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("No AQI readings are available for this city. Choose another city or load the Kaggle ZIP.")

    elif page == "Forecast":
        section_header("HISTORICAL TREND INSIGHTS", "A clearer view ahead.", "Explore recent historical patterns from your selected city. This view summarizes observed Kaggle data; it is not a predictive model.")
        if "AQI" in data and data.AQI.notna().any():
            trend = data.dropna(subset=["AQI"]).tail(30).copy()
            trend["7-day average"] = trend.AQI.rolling(7, min_periods=1).mean()
            a,b,c = st.columns(3)
            a.metric("Latest AQI", f"{trend.AQI.iloc[-1]:.0f}", aqi_category(trend.AQI.iloc[-1])[0])
            c.metric("30-day average", f"{trend.AQI.mean():.0f}")
            difference = trend.AQI.iloc[-1] - trend.AQI.iloc[0]
            b.metric("Change across view", f"{difference:+.0f}", "Historical readings")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend.Date,y=trend.AQI,mode="lines",name="Daily AQI",line={"color":PALETTE["mint"],"width":2},fill="tozeroy",fillcolor="rgba(102,216,182,.08)"))
            fig.add_trace(go.Scatter(x=trend.Date,y=trend["7-day average"],mode="lines",name="7-day rolling average",line={"color":PALETTE["purple"],"dash":"dash","width":2}))
            fig.update_layout(height=390,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8ba39a",yaxis_gridcolor="rgba(190,224,210,.09)",xaxis_showgrid=False,legend_orientation="h")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            st.info("Historical data does not include future predictions. The rolling average is shown as a trend guide, not a forecast.")

    elif page == "Map view":
        section_header("CITY COMPARISON", "Air quality, around you.", "Compare latest available city readings from the Kaggle dataset.")
        full = st.session_state.air_data
        if {"City","AQI"}.issubset(full.columns):
            latest = full.dropna(subset=["AQI"]).sort_values("Date").groupby("City",as_index=False).tail(1)
            coords = {"Delhi":(28.61,77.21),"Mumbai":(19.08,72.88),"Kolkata":(22.57,88.36),"Bengaluru":(12.97,77.59),"Chennai":(13.08,80.27),"Hyderabad":(17.39,78.49),"Pune":(18.52,73.86),"Ahmedabad":(23.02,72.57),"Jaipur":(26.91,75.79),"Lucknow":(26.85,80.95),"Patna":(25.61,85.14),"Guwahati":(26.14,91.74),"Amritsar":(31.63,74.87),"Chandigarh":(30.73,76.78),"Bhopal":(23.26,77.41),"Kochi":(9.93,76.27),"Visakhapatnam":(17.69,83.22),"Thiruvananthapuram":(8.52,76.94),"Coimbatore":(11.02,76.96),"Amaravati":(16.54,80.52),"Aizawl":(23.73,92.72),"Shillong":(25.58,91.89),"Jorapokhar":(23.72,86.41),"Talcher":(20.95,85.23),"Brajrajnagar":(21.82,83.92),"Ernakulam":(9.98,76.28),"Gurugram":(28.46,77.03)}
            latest["lat"] = latest.City.map(lambda x: coords.get(x,(20.59,78.96))[0]); latest["lon"] = latest.City.map(lambda x: coords.get(x,(20.59,78.96))[1])
            latest["Category"] = latest.AQI.map(lambda v: aqi_category(v)[0])
            # Draw a self-contained coordinate map. Plotly's tile-backed map
            # can render as a black panel when the browser cannot fetch tiles.
            india_outline_lon = [68, 71, 75, 77, 80, 83, 86, 89, 94, 97, 94, 91, 88, 85, 82, 79, 76, 73, 71, 68, 68]
            india_outline_lat = [23, 30, 34, 35, 34, 31, 29, 27, 28, 24, 20, 17, 13, 9, 8, 10, 13, 17, 20, 22, 23]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=india_outline_lon,y=india_outline_lat,mode="lines",fill="toself",fillcolor="rgba(70,137,105,.13)",line={"color":"rgba(102,216,182,.38)","width":1.5},hoverinfo="skip",showlegend=False))
            fig.add_trace(go.Scatter(x=latest["lon"],y=latest["lat"],mode="markers+text",text=latest["City"],textposition="top center",textfont={"size":9,"color":"#d9ebe2"},customdata=np.stack([latest["AQI"],latest["Category"],latest["Date"].dt.strftime("%d %b %Y")],axis=-1),hovertemplate="<b>%{text}</b><br>AQI %{customdata[0]:.0f}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",marker={"size":13,"color":latest["AQI"],"cmin":0,"cmax":300,"colorscale":[[0,"#52d4a1"],[.17,"#e7c45f"],[.34,"#ec9b58"],[.55,"#ef777b"],[.78,"#bb83e5"],[1,"#df5d68"]],"line":{"color":"#eaf6ef","width":1},"showscale":True,"colorbar":{"title":"AQI","thickness":10,"len":.7,"tickfont":{"color":"#9bb0a5"}}},showlegend=False))
            fig.update_layout(height=560,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0c1716",margin=dict(l=8,r=35,t=8,b=8),font_color="#ecf5f0",xaxis={"range":[66,100],"visible":False,"showgrid":True,"gridcolor":"rgba(190,224,210,.07)","dtick":5},yaxis={"range":[5,38],"visible":False,"showgrid":True,"gridcolor":"rgba(190,224,210,.07)","dtick":5,"scaleanchor":"x","scaleratio":1},hoverlabel={"bgcolor":"#13211f","font_color":"#ecf5f0"})
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            st.caption("City coordinates are approximate map markers. AQI values and dates come from the uploaded dataset.")
            st.dataframe(latest[["City","Date","AQI","Category"]].sort_values("AQI",ascending=False),use_container_width=True,hide_index=True)
        else: st.info("Load the Kaggle dataset to compare city readings.")

    elif page == "Analytics":
        section_header("EXPLORE YOUR DATA", "The bigger picture.", "Historical patterns, pollutant comparisons, and downloadable project data.")
        full=st.session_state.air_data
        if "AQI" in data and data.AQI.notna().any():
            x,y,z=st.columns(3); x.metric("Average AQI",f"{data.AQI.mean():.1f}"); y.metric("Maximum AQI",f"{data.AQI.max():.0f}"); z.metric("Daily records",f"{len(data):,}")
            col1,col2=st.columns(2)
            with col1:
                if "Date" in data:
                    daily=data.groupby("Date",as_index=False).AQI.mean()
                    st.plotly_chart(line_chart(daily,"Date","AQI"),use_container_width=True,config={"displayModeBar":False})
            with col2:
                if "AQI_Bucket" in data:
                    buckets=data["AQI_Bucket"].value_counts().rename_axis("Category").reset_index(name="Days")
                    fig=px.bar(buckets,x="Category",y="Days",color="Category",color_discrete_sequence=["#52d4a1","#e7c45f","#ec9b58","#ef777b","#bb83e5","#df5d68"])
                    fig.update_layout(height=290,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8ba39a",showlegend=False,margin=dict(l=5,r=5,t=20,b=5))
                    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            st.markdown('<div class="eyebrow">DATA INTEGRATION INVENTORY</div>',unsafe_allow_html=True)
            st.dataframe(st.session_state.file_info,use_container_width=True,hide_index=True)
            csv=full.to_csv(index=False).encode("utf-8")
            st.download_button("Download integrated CSV",csv,"airsense_integrated_air_quality.csv","text/csv",use_container_width=False)
        else: st.info("Upload the Kaggle ZIP to explore historical analytics.")

    elif page == "My sensors":
        section_header("DEVICE MANAGEMENT", "Your sensor network.", "Device cards are UI demonstrations; this project does not connect to physical sensors.")
        sensors=[("Living room","AS-2048","Online",92,"2.4.1","12 Oct 2024"),("Bedroom","AS-2051","Online",76,"2.4.1","09 Oct 2024"),("Studio","AS-1982","Low battery",18,"2.3.8","28 Sep 2024")]
        cols=st.columns(3)
        for col,(name,sid,status,battery,firmware,calibration) in zip(cols,sensors):
            with col:
                st.markdown('<div class="card">',unsafe_allow_html=True)
                st.markdown(f'<div class="eyebrow">{status.upper()} · {sid}</div><h3>{name}</h3>',unsafe_allow_html=True)
                st.metric("Battery",f"{battery}%")
                st.caption(f"Firmware {firmware} · Calibrated {calibration}")
                if st.button(f"Manage {name}",key=sid): st.toast(f"Device settings for {name}")
                st.markdown('</div>',unsafe_allow_html=True)
        with st.expander("＋ Add a sensor"):
            n=st.text_input("Sensor name",placeholder="e.g. Nursery")
            if st.button("Pair sensor") and n: st.success(f"{n} is ready to pair in this demo.")

    elif page == "Alerts":
        section_header("PERSONALIZED NOTIFICATIONS", "Stay one step ahead.", "Set a project threshold and choose how you’d like alerts to appear.")
        left,right=st.columns([.9,1.1])
        with left:
            st.markdown('<div class="card"><div class="eyebrow">YOUR ALERT RULE</div><h3>Set your comfort zone.</h3>',unsafe_allow_html=True)
            threshold=st.slider("Alert me when PM2.5 is above",5,100,35)
            st.caption(f"Current threshold: {threshold} µg/m³")
            for channel in ["Email notifications","SMS notifications","Push notifications"]: st.toggle(channel,value=channel!="SMS notifications",key=channel)
            if st.button("Save alert preferences",use_container_width=True): st.success("Preferences saved for this session.")
            st.markdown('</div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="card"><div class="eyebrow">RECENT ACTIVITY</div><h3>Alert history</h3>',unsafe_allow_html=True)
            if "PM2.5" in data:
                logs=data.dropna(subset=["PM2.5"]).tail(8).sort_values("Date",ascending=False)
                for _,row in logs.iterrows():
                    if row["PM2.5"]>threshold: st.warning(f"PM2.5 above threshold · {row['PM2.5']:.1f} µg/m³ · {row['Date']:%d %b %Y}")
                    else: st.success(f"Air within your PM2.5 threshold · {row['PM2.5']:.1f} µg/m³ · {row['Date']:%d %b %Y}")
            else: st.info("Upload data containing PM2.5 measurements to view alert history.")
            st.markdown('</div>',unsafe_allow_html=True)

    elif page == "Learn":
        section_header("A LITTLE KNOWLEDGE GOES A LONG WAY", "Small steps, cleaner air.", "Friendly, science-backed notes for feeling good in your space.")
        tips=[("THE BASICS","What is AQI, anyway?","AQI translates several pollutant measurements into a single, easier-to-understand scale. Lower readings generally mean cleaner air."),("FOR YOUR HOME","A breath of fresh air, indoors","Ventilate when outdoor air is clean, use a correctly sized HEPA purifier, and reduce smoke and strong fragrances."),("WELLNESS","Masks, decoded","A well-fitting respirator can reduce exposure to fine particles when outdoor air quality is poor."),("EXTRA CARE","For sensitive groups","Children, older adults, and people with heart or lung conditions may want to limit strenuous activity when AQI rises.")]
        cols=st.columns(2)
        for i,(tag,title,body) in enumerate(tips):
            with cols[i%2]:
                st.markdown(f'<div class="card"><div class="eyebrow">{tag}</div><h3>{title}</h3><p class="muted">{body}</p></div>',unsafe_allow_html=True)
        with st.expander("Where does this dashboard data come from?"): st.write("Upload the Air Quality Data in India Kaggle ZIP. AirSense reads all CSV files locally, uses city_day as the daily AQI base, aggregates hourly/station readings to daily city values, and joins them on City and Date.")
        with st.expander("Does historical data give a forecast?"): st.write("No. Historical observations and rolling averages describe past readings, not future air quality.")

    elif page == "About":
        section_header("OUR STORY", "Breathe smarter, live cleaner.", "Better air starts with knowing.")
        left,right=st.columns([1.1,.9])
        with left:
            st.markdown('<div class="card"><div class="eyebrow">THE AIRSENSE IDEA</div><h2>Make the invisible easier to understand.</h2><p class="muted">Air is part of every moment. AirSense turns air quality data into clear, calm guidance, helping people make informed choices about the spaces where they live, work, and rest.</p></div>',unsafe_allow_html=True)
            st.markdown("#### Frequently asked")
            with st.expander("Where does AirSense get its data?"): st.write("From the Kaggle Air Quality Data in India ZIP you upload. Data is processed in this app session.")
            with st.expander("Is this connected to physical sensors?"): st.write("No. Sensor management is a product demonstration; it does not connect to a real device.")
            with st.expander("Is my ZIP uploaded to a server?"): st.write("The app reads the file in the Streamlit session. If hosted remotely, files are processed by that host. For local use, it stays on your computer.")
        with right:
            st.markdown('<div class="card"><div class="eyebrow">SAY HELLO</div><h3>We’d love to hear from you.</h3>',unsafe_allow_html=True)
            with st.form("contact"):
                st.text_input("Your name")
                st.text_input("Email address")
                st.text_area("Message")
                if st.form_submit_button("Send a message"): st.success("Thanks for reaching out. This demo does not send messages to a server.")
            st.markdown('</div>',unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:rgba(190,224,210,.10);margin-top:28px"><div style="display:flex;justify-content:space-between;color:#71867d;font-size:9px"><span>AirSense · Breathe smarter, live cleaner.</span><span>{source_label} · © 2024</span></div>',unsafe_allow_html=True)


def health_advice(value):
    if value <= 50: return "Air quality is great. It’s a perfect day to get outside."
    if value <= 100: return "Air quality is acceptable. Sensitive individuals may want to take it easy."
    if value <= 150: return "Sensitive groups should reduce prolonged outdoor exertion."
    return "Consider limiting time outdoors and keep windows closed."


def metric_html(label, value, unit, note):
    number = "—" if pd.isna(value) else f"{value:.1f}" if isinstance(value, (int,float,np.number)) else str(value)
    return f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{number}<span class="metric-unit">{unit}</span></div><div class="metric-note">{note}</div></div>'


if __name__ == "__main__":
    main()
