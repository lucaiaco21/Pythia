# -*- coding: utf-8 -*-
"""
☕ Cafe Sales Prediction Dashboard - Multi-Scenario Forecasting
TFM UCM - Page 4
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Cafe Sales Prediction Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS (Matching Your Image) ====================
st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #F5EFE6;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        color: #F5EFE6;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #F5DEB3;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #8B4513;
    }

    /* Buttons */
    .stButton>button {
        background-color: #8B4513;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #A0522D;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E8DCC4;
    }

    /* Info boxes */
    .stAlert {
        background-color: #FFF8DC;
        border-left: 4px solid #D2691E;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>☕ Cafe Sales Prediction Dashboard</h1>
    <p>Upload historical sales data to predict the next 14 days using Prophet ML model</p>
</div>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
MODEL_CONFIG = {
    "Ventas Totales (€)": {
        "file": "2026_02_01_prophet_total_revenue_model.joblib",
        "unit": "€",
        "color_baseline": "#8B4513",
        "color_low": "#CD853F",
        "color_high": "#654321",
        "description": "Total revenue in Euros"
    },
    "Coffee Clásico (units)": {
        "file": "2026_02_01_prophet_units_classic_coffee.joblib",
        "unit": "units",
        "color_baseline": "#D2691E",
        "color_low": "#DEB887",
        "color_high": "#A0522D",
        "description": "Classic coffee units sold"
    },
    "Pastries & Sweets (units)": {
        "file": "2026_02_01_prophet_units_pastries_and_sweets.joblib",
        "unit": "units",
        "color_baseline": "#CD5C5C",
        "color_low": "#F08080",
        "color_high": "#8B3A3A",
        "description": "Pastries and sweets units sold"
    },
    "Desayunos/Tostadas (units)": {
        "file": "2026_02_01_prophet_units_DESAYUNOS_TOSTADAS_PANES_model.joblib",
        "unit": "units",
        "color_baseline": "#DAA520",
        "color_low": "#F0E68C",
        "color_high": "#B8860B",
        "description": "Breakfast items units sold"
    }
}

SCENARIO_CONFIG = {
    "Baseline": {
        "file": "20260203_all_inputs_cxf_baseline.csv",
        "description": "Expected normal conditions",
        "icon": "📊"
    },
    "Low (Pessimistic)": {
        "file": "20260203_all_inputs_cxf_low.csv",
        "description": "Conservative estimate with adverse conditions",
        "icon": "📉"
    },
    "High (Optimistic)": {
        "file": "20260203_all_inputs_cxf_high.csv",
        "description": "Best case scenario with favorable conditions",
        "icon": "📈"
    }
}

# ==================== CACHING FUNCTIONS ====================
@st.cache_resource
def load_model(model_name):
    """Load and cache Prophet models from GitHub repository"""
    try:
        model_file = MODEL_CONFIG[model_name]["file"]

        # For Streamlit Cloud deployment, models should be in root or models/ folder
        possible_paths = [
            Path(model_file),  # Root directory
            Path("models") / model_file,  # Models subdirectory
            Path("data") / model_file,  # Data subdirectory
        ]

        for path in possible_paths:
            if path.exists():
                loaded_object = joblib.load(path)

                # Handle different joblib save formats
                if isinstance(loaded_object, dict):
                    for key in ['model', 'prophet', 'estimator', 'regressor']:
                        if key in loaded_object:
                            return loaded_object[key]
                return loaded_object

        st.error(f"Model file not found: {model_file}")
        return None

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_scenario_data(scenario_name):
    """Load and cache scenario input data"""
    try:
        scenario_file = SCENARIO_CONFIG[scenario_name]["file"]

        # For Streamlit Cloud deployment
        possible_paths = [
            Path(scenario_file),
            Path("data") / scenario_file,
            Path("scenarios") / scenario_file,
        ]

        for path in possible_paths:
            if path.exists():
                return pd.read_csv(path)

        st.error(f"Scenario file not found: {scenario_file}")
        return None

    except Exception as e:
        st.error(f"Error loading scenario data: {str(e)}")
        return None

# ==================== PREDICTION FUNCTION ====================
def run_multi_scenario_prediction(model_name):
    """
    Run predictions for all three scenarios (Baseline, Low, High)
    Returns a dictionary with results for each scenario
    """
    model = load_model(model_name)
    if model is None:
        return None

    results = {}
    all_predictions = pd.DataFrame()

    with st.spinner(f"Running predictions for {model_name}..."):
        for scenario_name in ["Baseline", "Low (Pessimistic)", "High (Optimistic)"]:
            scenario_data = load_scenario_data(scenario_name)

            if scenario_data is None:
                continue

            try:
                # Run prediction with Prophet
                forecast = model.predict(scenario_data)

                # Extract predictions and dates
                predictions = forecast['yhat'].round(0).astype(int)

                # Handle date column (Prophet uses 'ds')
                if 'ds' in scenario_data.columns:
                    dates = pd.to_datetime(scenario_data['ds'])
                elif 'date' in scenario_data.columns:
                    dates = pd.to_datetime(scenario_data['date'])
                else:
                    dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')

                # Store results
                results[scenario_name] = {
                    'predictions': predictions,
                    'dates': dates,
                    'mean': predictions.mean(),
                    'sum': predictions.sum(),
                    'min': predictions.min(),
                    'max': predictions.max()
                }

                # Build combined dataframe
                if all_predictions.empty:
                    all_predictions['Date'] = dates
                all_predictions[scenario_name] = predictions

            except Exception as e:
                st.error(f"Prediction failed for {scenario_name}: {str(e)}")
                continue

    if not all_predictions.empty:
        all_predictions.set_index('Date', inplace=True)

    return {
        'dataframe': all_predictions,
        'details': results,
        'model_name': model_name
    }

# ==================== VISUALIZATION FUNCTIONS ====================
def create_comparison_chart(prediction_results, model_name):
    """Create an interactive Plotly chart comparing all three scenarios"""
    df = prediction_results['dataframe']
    config = MODEL_CONFIG[model_name]

    fig = go.Figure()

    # Add traces for each scenario
    scenarios = ["Baseline", "Low (Pessimistic)", "High (Optimistic)"]
    colors = [config['color_baseline'], config['color_low'], config['color_high']]

    for scenario, color in zip(scenarios, colors):
        if scenario in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[scenario],
                mode='lines+markers',
                name=scenario,
                line=dict(color=color, width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate=f'<b>{scenario}</b><br>' +
                              'Date: %{x|%Y-%m-%d}<br>' +
                              f'Value: %{{y}} {config["unit"]}<br>' +
                              '<extra></extra>'
            ))

    # Update layout with coffee shop theme
    fig.update_layout(
        title=dict(
            text=f"14-Day Forecast: {model_name}",
            font=dict(size=24, color='#8B4513', family='Arial Black')
        ),
        xaxis_title="Date",
        yaxis_title=f"Predicted Value ({config['unit']})",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#8B4513',
            borderwidth=2
        ),
        plot_bgcolor='#FAFAF0',
        paper_bgcolor='#F5EFE6',
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E8DCC4')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E8DCC4')

    return fig

def display_metrics(prediction_results, model_name):
    """Display key metrics for each scenario"""
    details = prediction_results['details']
    config = MODEL_CONFIG[model_name]
    unit = config['unit']

    st.markdown("### 📊 Scenario Comparison")

    cols = st.columns(3)

    for idx, (scenario_name, scenario_data) in enumerate(details.items()):
        with cols[idx]:
            icon = SCENARIO_CONFIG[scenario_name]["icon"]
            st.markdown(f"#### {icon} {scenario_name}")

            st.metric(
                label=f"Total (14 days)",
                value=f"{scenario_data['sum']:,.0f} {unit}"
            )
            st.metric(
                label="Daily Average",
                value=f"{scenario_data['mean']:,.0f} {unit}"
            )
            st.metric(
                label="Range",
                value=f"{scenario_data['min']:,.0f} - {scenario_data['max']:,.0f}"
            )

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/hot-beverage_2615.png", width=100)
    st.markdown("## 🎯 Model Selection")

    selected_model = st.selectbox(
        "Choose prediction model:",
        options=list(MODEL_CONFIG.keys()),
        help="Select the metric you want to forecast"
    )

    st.markdown("---")
    st.markdown(f"**Description:** {MODEL_CONFIG[selected_model]['description']}")

    st.markdown("---")
    st.markdown("### 📋 Scenario Details")
    for scenario_name, scenario_info in SCENARIO_CONFIG.items():
        with st.expander(f"{scenario_info['icon']} {scenario_name}"):
            st.write(scenario_info['description'])

    st.markdown("---")
    run_prediction = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

# ==================== MAIN CONTENT ====================

# Info box
st.info("👈 Select a model from the sidebar and click **Run Prediction** to generate 14-day forecasts")

# Display model info
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### Currently Selected: **{selected_model}**")
with col2:
    st.markdown(f"**Unit:** {MODEL_CONFIG[selected_model]['unit']}")

st.markdown("---")

# Run prediction when button is clicked
if run_prediction:
    prediction_results = run_multi_scenario_prediction(selected_model)

    if prediction_results and not prediction_results['dataframe'].empty:
        # Store results in session state
        st.session_state['last_prediction'] = prediction_results
        st.success("✅ Prediction completed successfully!")

        # Display chart
        st.plotly_chart(
            create_comparison_chart(prediction_results, selected_model),
            use_container_width=True
        )

        # Display metrics
        display_metrics(prediction_results, selected_model)

        # Download option
        st.markdown("---")
        st.markdown("### 💾 Export Results")

        csv_data = prediction_results['dataframe'].reset_index().to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"predictions_{selected_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Show data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(
                prediction_results['dataframe'].reset_index(),
                use_container_width=True,
                height=400
            )
    else:
        st.error("❌ Prediction failed. Please check model and data files.")

# Display last prediction if exists
elif 'last_prediction' in st.session_state:
    st.info("Showing last prediction results. Click **Run Prediction** to generate new forecasts.")

    last_results = st.session_state['last_prediction']

    st.plotly_chart(
        create_comparison_chart(last_results, last_results['model_name']),
        use_container_width=True
    )

    display_metrics(last_results, last_results['model_name'])

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8B4513; padding: 1rem;'>
    <p>📚 TFM UCM - Data Science Project | ☕ Powered by Prophet ML</p>
</div>
""", unsafe_allow_html=True)
