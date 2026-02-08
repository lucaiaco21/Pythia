import streamlit as st
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime

# Uses your existing pipeline import (must exist earlier in your script)
# from data_pipeline_etl.prophet_inputs_pipeline import build_prophet_prediction_inputs

# ------------------ REQUIRED PATHS (DEFINE BEFORE USE) ------------------
# Support both filenames: the one in your repo code and the one you attached (hyphens)
WEATHER_CANDIDATES = [
    Path("data/2026-01-31 Average Weather 2022 to 2026.xlsx"),
    Path("data/2026-01-31-Average-Weather-2022-to-2026.xlsx"),
    Path("2026-01-31 Average Weather 2022 to 2026.xlsx"),
    Path("2026-01-31-Average-Weather-2022-to-2026.xlsx"),
]
AVG_WEATHER_XLSX = next((p for p in WEATHER_CANDIDATES if p.exists()), WEATHER_CANDIDATES[0])

DEFAULT_INVOICE_TXT = Path("data/2026_01_Invoice_layout_CafeMadrid.txt")

SCENARIO_NAMES = ["Baseline", "Low Pessimistic", "High Optimistic"]


# MUST exist before build_scenarios_* functions
try:
    from data_pipeline_etl.prophet_inputs_pipeline import build_prophet_prediction_inputs
except Exception as e:
    st.error("Could not import build_prophet_prediction_inputs from data_pipeline_etl.prophet_inputs_pipeline.")
    st.exception(e)
    st.stop()

# ------------------ SCENARIO BUILDERS ------------------
#DEFAULT_INVOICE_TXT = Path("data/2026_01_Invoice_layout_CafeMadrid.txt")

@st.cache_data(show_spinner=False)
def build_scenarios_from_path(txt_path: str, avg_weather_xlsx_path: str):
    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=txt_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )
    return {
        "Baseline": baseline_df,
        "Low Pessimistic": low_df,
        "High Optimistic": high_df
    }


@st.cache_data(show_spinner=False)
def build_scenarios_from_upload(txt_bytes: bytes, avg_weather_xlsx_path: str):
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(txt_bytes)
        tmp_path = tmp.name

    baseline_df, low_df, high_df = build_prophet_prediction_(
        raw_txt_path=tmp_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )
    return {"Baseline": baseline_df, "Low Pessimistic": low_df, "High Optimistic": high_df

    }

# ------------------ SIDEBAR UI (put inside your existing `with st.sidebar:` block) ------------------
st.markdown("### Inputs")

input_mode = st.radio(
    "Invoice layout source",
    ["Use demo dataset (recommended)", "Upload new .txt"],
    index=0
)

uploaded_txt = None
if input_mode == "Upload new .txt":
    uploaded_txt = st.file_uploader(
        "Upload invoice_layout.txt",
        type=["txt"],
        help="Upload the raw invoice export text file invoice_layout.txt."
    )
else:
    st.caption(f"Using demo file: {DEFAULT_INVOICE_TXT.as_posix()}")

st.caption(f"Weather file: {AVG_WEATHER_XLSX.as_posix()}")

st.markdown("---")
runprediction = st.button("Run Prediction", use_container_width=True, type="primary")


# ------------------ RUN PREDICTION (put where your current `if runprediction:` is) ------------------
if runprediction:
    if not AVG_WEATHER_XLSX.exists():
        st.error(
            "Weather file not found. Put it in /data or project root.\n\n"
            "Expected one of:\n"
            "- data/2026-01-31 Average Weather 2022 to 2026.xlsx\n"
            "- data/2026-01-31-Average-Weather-2022-to-2026.xlsx"
        )
        st.stop()

    if input_mode == "Upload new .txt":
        if uploaded_txt is None:
            st.error("Please upload invoice_layout.txt first (or switch to the demo dataset).")
            st.stop()
        scenarios = build_scenarios_from_upload(uploaded_txt.getvalue(), str(AVG_WEATHER_XLSX))
    else:
        if not DEFAULT_INVOICE_TXT.exists():
            st.error(f"Demo file not found: {DEFAULT_INVOICE_TXT.as_posix()} (upload a .txt instead).")
            st.stop()
        scenarios = build_scenarios_from_path(str(DEFAULT_INVOICE_TXT), str(AVG_WEATHER_XLSX))

    prediction_results = run_multi_scenario_prediction(selected_model, scenarios)

    if prediction_results and not prediction_results["dataframe"].empty:
        st.session_state["last_prediction"] = prediction_results

        st.plotly_chart(create_comparison_chart(prediction_results, selected_model), use_container_width=True)
        display_metrics(prediction_results, selected_model)

        st.markdown("### Scenario comparison")

        df_cmp = prediction_results["dataframe"].copy()
        df_cmp = df_cmp.reindex(columns=[c for c in SCENARIO_NAMES if c in df_cmp.columns])
        df_cmp = df_cmp.rename(columns={
            "Baseline": "Baseline",
            "Low Pessimistic": "Low (pessimistic)",
            "High Optimistic": "High (optimistic)"
        })

        mobile_view = st.toggle("Mobile-friendly view", value=False)
        if mobile_view:
            df_long = (
                df_cmp.reset_index()
                      .melt(id_vars="Date", var_name="Scenario", value_name="Prediction")
                      .sort_values(["Date", "Scenario"])
            )
            st.dataframe(df_long, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_cmp, use_container_width=True)

        csv_data = df_cmp.reset_index().to_csv(index=False)
        st.download_button(
            "Download as CSV",
            data=csv_data,
            filename=f"predictions_{selected_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("Prediction failed or produced empty results. Check your invoice_layout.txt content.")


