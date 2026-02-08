# ====== DROP-IN REPLACEMENT: SIDEBAR INPUTS + RUN PREDICTION + MOBILE-FRIENDLY COMPARISON ======
# Paste this block inside your existing `with st.sidebar:` section (replacing the current Inputs/uploader + run button block)
# AND paste the "MAIN RENDER" part right after the sidebar block (where you currently handle `if runprediction:`).

from pathlib import Path
from tempfile import NamedTemporaryFile
import pandas as pd
import streamlit as st

# --- Demo file in your repo ---
DEFAULT_INVOICE_TXT = Path("data/2026_01_Invoice_layout_CafeMadrid.txt")

# --- Helper: build scenarios from a repo path (demo mode) ---
@st.cache_data(show_spinner=False)
def build_scenarios_from_path(txt_path: str, avg_weather_xlsx_path: str):
    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=txt_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )
    return {"Baseline": baseline_df, "Low Pessimistic": low_df, "High Optimistic": high_df}

# --- Helper: build scenarios from uploaded bytes (upload mode) ---
@st.cache_data(show_spinner=False)
def build_scenarios_from_upload(txt_bytes: bytes, avg_weather_xlsx_path: str):
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(txt_bytes)
        tmp_path = tmp.name
    baseline_df, low_df, high_df = build_prophet_prediction_inputs(
        raw_txt_path=tmp_path,
        avg_weather_xlsx_path=avg_weather_xlsx_path,
        verbose=False
    )
    return {"Baseline": baseline_df, "Low Pessimistic": low_df, "High Optimistic": high_df}

# ===================== SIDEBAR UI (replace your current Inputs block with this) =====================
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

st.info("Horizon days of prediction depends on how many days of historic data you have in invoice_layout.txt.")
st.markdown("---")
runprediction = st.button("Run Prediction", use_container_width=True, type="primary")

# ===================== MAIN RENDER (put this where your current `if runprediction:` block is) =====================
if runprediction:
    if not AVG_WEATHER_XLSX.exists():
        st.error(f"Missing fixed weather file: {AVG_WEATHER_XLSX}")
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
        # Store last prediction (your app already uses this pattern)
        st.session_state["last_prediction"] = prediction_results
        st.success("Prediction completed successfully!")

        # Chart (already labeled via trace names)
        st.plotly_chart(create_comparison_chart(prediction_results, selected_model), use_container_width=True)

        # Your existing metrics
        display_metrics(prediction_results, selected_model)

        # Scenario comparison table: make labels explicit + add mobile-friendly view
        st.markdown("### Scenario comparison")

        df_cmp = prediction_results["dataframe"].copy()

        # Ensure consistent ordering and clearer labels
        desired_order = ["Baseline", "Low Pessimistic", "High Optimistic"]
        df_cmp = df_cmp.reindex(columns=[c for c in desired_order if c in df_cmp.columns])
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

        st.markdown("---")
        csv_data = df_cmp.reset_index().to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            filename=f"predictions_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.error("Prediction failed or produced empty results. Check your invoice_layout.txt content.")

