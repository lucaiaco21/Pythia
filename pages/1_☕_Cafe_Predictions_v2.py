# -*- coding: utf-8 -*-
"""
☕ Cafe Sales Prediction Dashboard - Multi-Scenario Forecasting (Page 4)

Upgrade: scenarios are built from uploaded invoice_layout.txt using
data_pipeline_etl/prophet_inputs_pipeline.py + fixed weather xlsx in /data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
from tempfile import NamedTemporaryFile

# NEW: import pipeline from package folder
from data_pipeline_etl.prophet_inputs_pipeline import build_prophet_prediction_inputs  # [file:2]

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Cafe Sales Prediction Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)  # [file:1]

# ==================== CUSTOM CSS (keep your existing CSS here) ====================
st.markdown("""""", unsafe_allow_html=True)  # [file:1]

# ==================== HEADER ====================
st.markdown(
    """
<div class="main-header">
  <h1>☕ Cafe Sales Prediction Dashboard</h1>
  <p>Upload invoice_layout.txt to predict the next 14 days using Prophet ML model</p>
  <p>📚 TFM UCM - Data Science Project | ☕ Powered by Prophet ML</p>
</div>
""",
    unsafe_allow_html=True
)  # [file:1]

# ==================== CONFIG ====================
MODELCONFIG = {
    "Ventas Totales": {
        "file": "2026_02_01_prophet_total_revenue_model.joblib",
        "unit": "€",
        "color_baseline": "#1f77b4",
        "color_low": "#ff7f0e",
        "color_high": "#2ca02c",
        "description": "Total revenue in Euros"
    },
    "Coffee Clásico units": {
        "file": "2026_02_01_prophet_units_classic_coffee.joblib",
        "unit": "units",
        "color_baseline": "#d62728",
        "color_low": "#9467bd",
        "color_high": "#8c564b",
        "description": "Classic coffee units sold"
    },
    "Pastries & Sweets units": {
        "file": "2026_02_01_prophet_units_pastries_and_sweets.joblib",
        "unit": "units",
        "color_baseline": "#e377c2",
        "color_low": "#7f7f7f",
        "color_high": "#bcbd22",
        "description": "Pastries and sweets units sold"
    },
    "Desayunos/Tostadas units": {
        "file": "2026_02_01_prophet_units_DESAYUNOS_TOSTADAS_PANES_model.joblib",
        "unit": "units",
        "color_baseline": "#17becf",
        "color_low": "#ff9896",
        "color_high": "#98df8a",
        "description": "Breakfast items units sold"
    }
}


SCENARIO_NAMES = ["Baseline", "Low Pessimistic", "High Optimistic"]  # [file:1]

AVG_WEATHER_XLSX = Path("data/2026-01-31 Average Weather 2022 to 2026.xlsx")  # [file:2]

# ==================== CACHING FUNCTIONS ====================
@st.cache_resource
def load_model(model_name: str):
    try:
        model_file = MODELCONFIG[model_name]["file"]
        possible_paths = [
            Path(model_file),
            Path("models") / model_file,
            Path("data") / model_file
        ]
        for p in possible_paths:
            if p.exists():
                loaded_object = joblib.load(p)
                if isinstance(loaded_object, dict):
                    for key in ["model", "prophet", "estimator", "regressor"]:
                        if key in loaded_object:
                            return loaded_object[key]
                return loaded_object
        st.error(f"Model file not found: {model_file}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e)    
        return None  # [file:1]

@st.cache_data(show_spinner=False)
def build_scenarios_from_upload(txt_bytes: bytes, avg_weather_xlsx_path: str):
    """
    Writes the uploaded txt bytes to a temp file and runs the pipeline.
    Keeps pipeline defaults (horizon_days=14, invoices_lag_days=14, visitantes defaults).
    """
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(txt_bytes)
        tmp_path = tmp.name

    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=tmp_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )  # [file:2]

    return {
        "Baseline": baseline_df,
        "Low Pessimistic": low_df,
        "High Optimistic": high_df
    }  # [file:2]

# ==================== PREDICTION + VISUALS ====================
def run_multi_scenario_prediction(model_name: str, scenarios_dict: dict):
    model = load_model(model_name)
    if model is None:
        return None

    results = {}
    all_predictions = pd.DataFrame()

    for scenario_name in SCENARIO_NAMES:
        scenario_df = scenarios_dict.get(scenario_name)
        if scenario_df is None or scenario_df.empty:
            continue

        try:
            forecast = model.predict(scenario_df)  # Prophet uses 'ds' column
            preds = forecast["yhat"].round(0).astype(int)

            dates = pd.to_datetime(scenario_df["ds"]) if "ds" in scenario_df.columns else pd.date_range(
                start=datetime.now(), periods=len(preds), freq="D"
            )

            results[scenario_name] = {
                "predictions": preds,
                "dates": dates,
                "mean": preds.mean(),
                "sum": preds.sum(),
                "min": preds.min(),
                "max": preds.max()
            }

            if all_predictions.empty:
                all_predictions["Date"] = dates
            all_predictions[scenario_name] = preds

        except Exception as e:
            st.error(f"Prediction failed for {scenario_name}: {e}")
            st.exception(e)
            #continue

    if all_predictions.empty:
        return {"dataframe": pd.DataFrame(), "details": {}, "model_name": model_name}

    all_predictions = all_predictions.set_index("Date")
    return {"dataframe": all_predictions, "details": results, "model_name": model_name}  # [file:1]

def create_comparison_chart(prediction_results: dict, model_name: str):
    df = prediction_results["dataframe"]
    config = MODELCONFIG[model_name]

    fig = go.Figure()
    colors = [config["color_baseline"], config["color_low"], config["color_high"]]

    for scenario, color in zip(SCENARIO_NAMES, colors):
        if scenario in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[scenario],
                    mode="lines+markers",
                    name=scenario,
                    line=dict(color=color, width=3),
                    marker=dict(size=8, symbol="circle"),
                    hovertemplate=(
                        f"<b>{scenario}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        f"Value: %{{y}} {config['unit']}<extra></extra>"
                    )
                )
            )

    fig.update_layout(
        title=dict(text=f"14-Day Forecast - {model_name}", font=dict(size=24, color="#333333")),
        xaxis_title="Date",
        yaxis_title=f"Predicted Value ({config['unit']})",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FAFAFA",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")
    return fig  # [file:1]

def display_metrics(prediction_results: dict, model_name: str):
    details = prediction_results["details"]
    unit = MODELCONFIG[model_name]["unit"]

    st.markdown("### Scenario Comparison")
    cols = st.columns(3)

    for idx, scenario_name in enumerate(SCENARIO_NAMES):
        if scenario_name not in details:
            continue
        d = details[scenario_name]
        with cols[idx]:
            st.metric(label="Total (14 days)", value=f"{d['sum']:,.0f} {unit}")
            st.metric(label="Daily Average", value=f"{d['mean']:,.0f} {unit}")
            st.metric(label="Range", value=f"{d['min']:,.0f} - {d['max']:,.0f}")  # [file:1]

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## Model Selection")
    selected_model = st.selectbox(
        "Choose prediction model",
        options=list(MODELCONFIG.keys()),
        help="Select the metric you want to forecast."
    )  # [file:1]

    #st.markdown("---")
    #st.markdown(f"**Description:** {MODELCONFIG[selected_model]['description']}")
    #st.markdown(f"**Unit:** {MODELCONFIG[selected_model]['unit']}")

    st.markdown("---")
    st.markdown("## Inputs")
    uploaded_txt = st.file_uploader(
        "Upload invoice_layout.txt",
        type=["txt"],
        help="Upload the raw invoice export text file (invoice_layout.txt)."
    )  # [file:1]

    #st.info("Horizon days of prediction depends on how many days of historic data you have in invoice_layout.txt.")

    st.markdown("---")
    run_prediction = st.button("Run Prediction", use_container_width=True, type="primary")  # [file:1]

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This dashboard generates 14-day forecasts using Prophet ML models. "
        "Each prediction runs three scenarios (Baseline, Low, High)."
    )  # [file:1]

# ==================== MAIN ====================
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### Currently Selected: {selected_model}")
with col2:
    st.markdown(f"### Unit: {MODELCONFIG[selected_model]['unit']}")  # [file:1]

st.markdown("---")

if run_prediction:
    if uploaded_txt is None:
        st.error("Please upload invoice_layout.txt first.")
    elif not AVG_WEATHER_XLSX.exists():
        st.error(f"Missing fixed weather file: {AVG_WEATHER_XLSX}")
    else:
        try:
            scenarios = build_scenarios_from_upload(uploaded_txt.getvalue(), str(AVG_WEATHER_XLSX))
            prediction_results = run_multi_scenario_prediction(selected_model, scenarios)

            if prediction_results and not prediction_results["dataframe"].empty:
                st.session_state["last_prediction"] = prediction_results
                st.success("Prediction completed successfully!")

                st.plotly_chart(create_comparison_chart(prediction_results, selected_model), use_container_width=True)
                display_metrics(prediction_results, selected_model)

                st.markdown("---")
                csv_data = prediction_results["dataframe"].reset_index().to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name=f"predictions_{selected_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                with st.expander("View Raw Data"):
                    st.dataframe(prediction_results["dataframe"].reset_index(), use_container_width=True, height=400)
            else:
                st.error("Prediction failed or produced empty results. Check your invoice_layout.txt content.")
        except Exception as e:
            st.error(f"Pipeline/prediction error: {e}")

elif "last_prediction" in st.session_state:
    st.error("Showing last prediction results. Click Run Prediction to generate new forecasts.")
    last = st.session_state["last_prediction"]
    st.plotly_chart(create_comparison_chart(last, last["model_name"]), use_container_width=True)
    display_metrics(last, last["model_name"])

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666666; padding: 1rem;">'
    "<p>TFM UCM - Data Science Project | Powered by Prophet ML</p>"
    "</div>",
    unsafe_allow_html=True
)  # [file:1]

