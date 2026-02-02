import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from joblib import load
import os

# Page config
st.set_page_config(page_title="Predicciones de Ventas", page_icon="💹", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    """Load the pre-trained Prophet model"""
    model_path = '/Users/lucaiacomino/Desktop/Tesi/Streamlit_App/2026_01_28_prophet_sales_model.joblib'
    
    if os.path.exists(model_path):
        try:
            bundle = load(model_path)
            st.success(f"✅ Modelo cargado correctamente")
            return bundle
        except Exception as e:
            st.error(f"❌ Error cargando el modelo: {str(e)}")
            return None
    else:
        st.error(f"❌ Archivo no encontrado: {model_path}")
        return None

# Main app
def main():
    # Title
    st.title("💹 Predicciones de Ventas")
    st.markdown("Genera predicciones de ventas para los próximos 14 días con tu modelo Prophet.")
    st.divider()
    
    # Load model
    bundle = load_model()
    
    if bundle is None:
        st.stop()
    
    # File upload
    st.subheader("📁 Cargar Datos")
    input_file = st.file_uploader(
        "Sube el archivo CSV con los datos de los últimos 28 días",
        type=['csv'],
        help="CSV con columnas: ds y los regresores necesarios"
    )
    
    if input_file is not None:
        try:
            # Read the uploaded file
            df_in = pd.read_csv(input_file)
            
            # Show preview
            st.success(f"✅ Archivo cargado: {df_in.shape[0]} filas, {df_in.shape[1]} columnas")
            with st.expander("👀 Vista previa de los datos"):
                st.dataframe(df_in.head(10))
            
            # Generate predictions button
            if st.button("🚀 Generar Predicciones", type="primary"):
                with st.spinner("Generando predicciones..."):
                    try:
                        # Extract model and regressors
                        m = bundle['model']
                        regressors = bundle['regressors']
                        
                        # Prepare data
                        df_in["ds"] = pd.to_datetime(df_in["ds"])
                        future = df_in[["ds"] + regressors].copy()
                        
                        # Make predictions
                        forecast = m.predict(future)
                        
                        # Format output
                        forecast_output = forecast[["ds", "yhat_lower", "yhat", "yhat_upper"]].copy()
                        forecast_output = forecast_output.rename(columns={
                            "yhat_lower": "revenue_lower_95",
                            "yhat": "revenue_forecast",
                            "yhat_upper": "revenue_upper_95"
                        })
                        
                        # Display results
                        st.divider()
                        st.subheader("📊 Resultados")
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pronóstico Promedio", f"${forecast_output['revenue_forecast'].mean():,.2f}")
                        with col2:
                            st.metric("Mínimo (95%)", f"${forecast_output['revenue_lower_95'].min():,.2f}")
                        with col3:
                            st.metric("Máximo (95%)", f"${forecast_output['revenue_upper_95'].max():,.2f}")
                        
                        # Chart
                        fig = go.Figure()
                        
                        # Forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast_output['ds'],
                            y=forecast_output['revenue_forecast'],
                            mode='lines+markers',
                            name='Pronóstico',
                            line=dict(color='#1f77b4', width=3)
                        ))
                        
                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_output['ds'],
                            y=forecast_output['revenue_upper_95'],
                            mode='lines',
                            name='Límite Superior (95%)',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_output['ds'],
                            y=forecast_output['revenue_lower_95'],
                            mode='lines',
                            name='Límite Inferior (95%)',
                            line=dict(width=0),
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            fill='tonexty',
                            showlegend=True
                        ))
                        
                        fig.update_layout(
                            title="Predicción de Ventas",
                            xaxis_title="Fecha",
                            yaxis_title="Ventas ($)",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data table
                        st.subheader("📋 Tabla de Predicciones")
                        st.dataframe(forecast_output, use_container_width=True)
                        
                        # Download button
                        csv = forecast_output.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Predicciones (CSV)",
                            data=csv,
                            file_name=f"predicciones_ventas_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error al generar predicciones: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {str(e)}")
    
    else:
        st.info("👆 Por favor, sube un archivo CSV para comenzar")

if __name__ == "__main__":
    main()