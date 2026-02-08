# -*- coding: utf-8 -*-
"""
☕ Cafe Sales Prediction Dashboard - Multi-Scenario Forecasting

Consolidated version:
- Uses uploaded invoice_layout.txt when provided (priority).
- Otherwise uses demo dataset: data/2026_01_Invoice_layout_CafeMadrid.txt
- Informs user which source was used.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile

from data_pipeline_etl.prophet_inputs_pipeline import build_prophet_prediction_inputs


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Cafe Sales Prediction Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
section.stMain .block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("CAFE MADRID predictions Dashboard")


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

SCENARIO_NAMES = ["Baseline", "Low Pessimistic", "High Optimistic"]

# Weather file candidates (robust for local/deploy naming differences)
WEATHER_CANDIDATES = [
    Path("data/2026-01-31 Average Weather 2022 to 2026.xlsx"),
    Path("data/2026-01-31-Average-Weather-2022-to-2026.xlsx"),
    Path("2026-01-31 Average Weather 2022 to 2026.xlsx"),
    Path("2026-01-31-Average-Weather-2022-to-2026.xlsx"),
]
AVG_WEATHER_XLSX = next((p for p in WEATHER_CANDIDATES if p.exists()), WEATHER_CANDIDATES[0])

# Demo TXT candidates
DEFAULT_INVOICE_CANDIDATES = [
    Path("data/2026_01_Invoice_layout_CafeMadrid.txt"),
    Path("2026_01_Invoice_layout_CafeMadrid.txt"),
]
DEFAULT_INVOICE_TXT = next((p for p in DEFAULT_INVOICE_CANDIDATES if p.exists()), DEFAULT_INVOICE_CANDIDATES[0])


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
        return None


@st.cache_data(show_spinner=False)
def build_scenarios_from_upload(txt_bytes: bytes, avg_weather_xlsx_path: str):
    """
    Build scenario DataFrames from uploaded txt bytes.
    """
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(txt_bytes)
        tmp_path = tmp.name

    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=tmp_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )

    return {
        "Baseline": baseline_df,
        "Low Pessimistic": low_df,
        "High Optimistic": high_df
    }


@st.cache_data(show_spinner=False)
def build_scenarios_from_demo(demo_txt_path: str, avg_weather_xlsx_path: str):
    """
    Build scenario DataFrames from demo txt path.
    """
    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=demo_txt_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )

    return {
        "Baseline": baseline_df,
        "Low Pessimistic": low_df,
        "High Optimistic": high_df
    }


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
            forecast = model.predict(scenario_df)
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

    if all_predictions.empty:
        return {"dataframe": pd.DataFrame(), "details": {}, "model_name": model_name}

    all_predictions = all_predictions.set_index("Date")
    return {"dataframe": all_predictions, "details": results, "model_name": model_name}


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
    return fig


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
            st.metric(label="Range", value=f"{d['min']:,.0f} - {d['max']:,.0f}")


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## Model Selection")
    selected_model = st.selectbox(
        "Choose prediction model",
        options=list(MODELCONFIG.keys()),
        help="Select the metric you want to forecast."
    )

    st.markdown("---")
    st.markdown("## Inputs")

    # Optional uploader: if provided, it always has priority
    uploaded_txt = st.file_uploader(
        "Upload invoice_layout.txt (optional)",
        type=["txt"],
        help="If uploaded, this file is used. Otherwise, demo dataset is used."
    )

    st.caption(f"Demo dataset fallback: {DEFAULT_INVOICE_TXT.as_posix()}")
    st.caption(f"Weather file: {AVG_WEATHER_XLSX.as_posix()}")

    st.markdown("---")
    run_prediction = st.button("Run Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This dashboard generates 14-day forecasts using Prophet ML models. "
        "Each prediction runs three scenarios (Baseline, Low, High)."
    )


# ==================== MAIN ====================
st.markdown("---")

if run_prediction:
    # Check weather file
    if not AVG_WEATHER_XLSX.exists():
        st.error(
            "Missing weather file. Expected one of:\n"
            + "\n".join([f"- {p.as_posix()}" for p in WEATHER_CANDIDATES])
        )
        st.stop()

    try:
        # Priority rule: uploaded file > demo file
        if uploaded_txt is not None:
            scenarios = build_scenarios_from_upload(uploaded_txt.getvalue(), str(AVG_WEATHER_XLSX))
            source_used = f"Uploaded file: {uploaded_txt.name}"
        else:
            if not DEFAULT_INVOICE_TXT.exists():
                st.error(
                    "Demo dataset not found. Expected one of:\n"
                    + "\n".join([f"- {p.as_posix()}" for p in DEFAULT_INVOICE_CANDIDATES])
                )
                st.stop()

            scenarios = build_scenarios_from_demo(str(DEFAULT_INVOICE_TXT), str(AVG_WEATHER_XLSX))
            source_used = f"Demo dataset: {DEFAULT_INVOICE_TXT.as_posix()}"

        st.info(f"Prediction source: {source_used}")

        prediction_results = run_multi_scenario_prediction(selected_model, scenarios)

        if prediction_results and not prediction_results["dataframe"].empty:
            st.session_state["last_prediction"] = prediction_results
            st.success("Prediction completed successfully!")

            st.plotly_chart(
                create_comparison_chart(prediction_results, selected_model),
                use_container_width=True
            )
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
                st.dataframe(
                    prediction_results["dataframe"].reset_index(),
                    use_container_width=True,
                    height=400
                )
        else:
            st.error("Prediction failed or produced empty results. Check input TXT format/content.")

    except Exception as e:
        st.error(f"Pipeline/prediction error: {e}")
        st.exception(e)

elif "last_prediction" in st.session_state:
    st.warning("Showing last prediction results. Click Run Prediction to generate new forecasts.")
    last = st.session_state["last_prediction"]
    st.plotly_chart(create_comparison_chart(last, last["model_name"]), use_container_width=True)
    display_metrics(last, last["model_name"])

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666666; padding: 1rem;">'
    "<p>TFM UCM - Data Science Project | Powered by Prophet ML</p>"
    "</div>",
    unsafe_allow_html=True
)

