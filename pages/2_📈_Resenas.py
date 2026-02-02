import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Análisis de Reseñas", page_icon="🗞️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🗞️ Análisis de Reseñas de Clientes")
st.markdown("""
Descubre insights accionables basados en las reseñas de clientes de tu competencia en Madrid.
Estas recomendaciones están respaldadas por análisis de lenguaje natural con IA.
""")

st.divider()

# Insights data structure
insights_data = {
    "☕ Calidad del Café": {
        "icon": "☕",
        "color": "#8B4513",
        "questions": [
            {
                "question": "Among your competitors, the most preferred coffee type is [MASK].",
                "recommendations": [
                    {"text": "among your competitors, the most preferred coffee type is breakfast.", "keyword": "breakfast", "confidence": 0.23},
                    {"text": "among your competitors, the most preferred coffee type is toast.", "keyword": "toast", "confidence": 0.16},
                    {"text": "among your competitors, the most preferred coffee type is iced.", "keyword": "iced", "confidence": 0.06}
                ]
            },
            {
                "question": "Customers highly appreciate coffee that is [MASK].",
                "recommendations": [
                    {"text": "customers highly appreciate coffee that is delicious.", "keyword": "delicious", "confidence": 12.19},
                    {"text": "customers highly appreciate coffee that is breakfast.", "keyword": "breakfast", "confidence": 0.07}
                ]
            },
            {
                "question": "To improve your coffee, focus on making it more [MASK].",
                "recommendations": [
                    {"text": "to improve your coffee, focus on making it more delicious.", "keyword": "delicious", "confidence": 0.64},
                    {"text": "to improve your coffee, focus on making it more friendly.", "keyword": "friendly", "confidence": 0.09}
                ]
            },
            {
                "question": "The most valued coffee characteristic is [MASK].",
                "recommendations": [
                    {"text": "the most valued coffee characteristic is taste.", "keyword": "taste", "confidence": 1.06},
                    {"text": "the most valued coffee characteristic is food.", "keyword": "food", "confidence": 0.10},
                    {"text": "the most valued coffee characteristic is cup.", "keyword": "cup", "confidence": 0.07}
                ]
            }
        ]
    },
    "🍽️ Calidad de Comida": {
        "icon": "🍽️",
        "color": "#FF6347",
        "questions": [
            {
                "question": "Among your competitors, the most preferred food besides coffee is [MASK].",
                "recommendations": [
                    {"text": "among your competitors, the most preferred food besides coffee is breakfast.", "keyword": "breakfast", "confidence": 4.66},
                    {"text": "among your competitors, the most preferred food besides coffee is tea.", "keyword": "tea", "confidence": 2.29},
                    {"text": "among your competitors, the most preferred food besides coffee is chocolate.", "keyword": "chocolate", "confidence": 1.95}
                ]
            },
            {
                "question": "Customers value food that is [MASK].",
                "recommendations": [
                    {"text": "customers value food that is healthy.", "keyword": "healthy", "confidence": 13.21},
                    {"text": "customers value food that is delicious.", "keyword": "delicious", "confidence": 7.82},
                    {"text": "customers value food that is fresh.", "keyword": "fresh", "confidence": 3.26}
                ]
            },
            {
                "question": "The most appreciated food item is [MASK].",
                "recommendations": [
                    {"text": "the most appreciated food item is tea.", "keyword": "tea", "confidence": 0.19},
                    {"text": "the most appreciated food item is cake.", "keyword": "cake", "confidence": 0.04},
                    {"text": "the most appreciated food item is peanut.", "keyword": "peanut", "confidence": 0.03}
                ]
            },
            {
                "question": "Customers frequently praise food for being [MASK].",
                "recommendations": [
                    {"text": "customers frequently praise food for being delicious.", "keyword": "delicious", "confidence": 10.66},
                    {"text": "customers frequently praise food for being healthy.", "keyword": "healthy", "confidence": 4.05},
                    {"text": "customers frequently praise food for being fresh.", "keyword": "fresh", "confidence": 0.53}
                ]
            }
        ]
    },
    "👥 Servicio y Personal": {
        "icon": "👥",
        "color": "#4169E1",
        "questions": [
            {
                "question": "Customers in the location value service aspects such as [MASK].",
                "recommendations": [
                    {"text": "customers in the location value service aspects such as food.", "keyword": "food", "confidence": 1.33},
                    {"text": "customers in the location value service aspects such as service.", "keyword": "service", "confidence": 0.81},
                    {"text": "customers in the location value service aspects such as price.", "keyword": "price", "confidence": 0.11}
                ]
            },
            {
                "question": "Staff should be more [MASK] to improve customer satisfaction.",
                "recommendations": [
                    {"text": "staff should be more friendly to improve customer satisfaction.", "keyword": "friendly", "confidence": 0.10}
                ]
            },
            {
                "question": "Customers expect staff to be [MASK].",
                "recommendations": [
                    {"text": "customers expect staff to be friendly.", "keyword": "friendly", "confidence": 1.90}
                ]
            }
        ]
    },
    "🏮 Atmósfera y Ambiente": {
        "icon": "🏮",
        "color": "#9370DB",
        "questions": [
            {
                "question": "Customers prefer an atmosphere that is [MASK].",
                "recommendations": [
                    {"text": "customers prefer an atmosphere that is comfortable.", "keyword": "comfortable food", "confidence": 6.74},
                    {"text": "customers prefer an atmosphere that is delicious.", "keyword": "delicious", "confidence": 0.09}
                ]
            },
            {
                "question": "The most valued ambiance feature is [MASK].",
                "recommendations": [
                    {"text": "the most valued ambiance feature is seating.", "keyword": "seating", "confidence": 0.06}
                ]
            },
            {
                "question": "Customers appreciate spaces that feel [MASK].",
                "recommendations": [
                    {"text": "customers appreciate spaces that feel delicious.", "keyword": "delicious", "confidence": 0.10}
                ]
            },
            {
                "question": "The ideal cafe environment should be [MASK].",
                "recommendations": [
                    {"text": "the ideal cafe environment should be vegetarian.", "keyword": "vegetarian", "confidence": 0.07}
                ]
            }
        ]
    },
    "📍 Ubicación y Accesibilidad": {
        "icon": "📍",
        "color": "#32CD32",
        "questions": [
            {
                "question": "Customers value locations that are [MASK].",
                "recommendations": [
                    {"text": "customers value locations that are expensive.", "keyword": "expensive", "confidence": 1.11},
                    {"text": "customers value locations that are healthy.", "keyword": "healthy", "confidence": 1.02},
                    {"text": "customers value locations that are friendly.", "keyword": "friendly", "confidence": 0.30}
                ]
            },
            {
                "question": "The most important location feature is [MASK].",
                "recommendations": [
                    {"text": "the most important location feature is city.", "keyword": "city", "confidence": 0.07}
                ]
            },
            {
                "question": "Customers prefer places that are [MASK] to reach.",
                "recommendations": [
                    {"text": "customers prefer places that are expensive to reach.", "keyword": "expensive", "confidence": 0.09},
                    {"text": "customers prefer places that are pleasant to reach.", "keyword": "pleasant", "confidence": 0.01}
                ]
            }
        ]
    },
    "💰 Precio y Valor": {
        "icon": "💰",
        "color": "#FFD700",
        "questions": [
            {
                "question": "Customers expect prices to be [MASK].",
                "recommendations": [
                    {"text": "customers expect prices to be expensive.", "keyword": "expensive", "confidence": 0.16}
                ]
            },
            {
                "question": "The most valued pricing aspect is [MASK].",
                "recommendations": [
                    {"text": "the most valued pricing aspect is price.", "keyword": "price", "confidence": 2.28},
                    {"text": "the most valued pricing aspect is quality.", "keyword": "quality", "confidence": 0.24},
                    {"text": "the most valued pricing aspect is service.", "keyword": "service", "confidence": 0.15}
                ]
            },
            {
                "question": "Customers appreciate when prices are [MASK].",
                "recommendations": [
                    {"text": "customers appreciate when prices are expensive.", "keyword": "expensive", "confidence": 0.15}
                ]
            }
        ]
    }
}

# Overview Dashboard
st.header("📊 Resumen Ejecutivo")

# Calculate overall statistics
total_recommendations = sum(len(cat["questions"]) for cat in insights_data.values())
all_keywords = []
all_confidences = []

for category, data in insights_data.items():
    for question in data["questions"]:
        for rec in question["recommendations"]:
            all_keywords.append(rec["keyword"])
            all_confidences.append(rec["confidence"])

# Top metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🎯</h3>
        <h2>{len(insights_data)}</h2>
        <p>Categorías</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>❓</h3>
        <h2>{total_recommendations}</h2>
        <p>Preguntas Analizadas</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔑</h3>
        <h2>{len(set(all_keywords))}</h2>
        <p>Keywords Únicas</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈</h3>
        <h2>{avg_confidence:.2f}%</h2>
        <p>Confianza Promedio</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Top Keywords Visualization
st.subheader("🔥 Top Keywords por Confianza")

# Create dataframe for top keywords
keyword_df = pd.DataFrame({
    'keyword': all_keywords,
    'confidence': all_confidences
})
top_keywords = keyword_df.groupby('keyword')['confidence'].max().sort_values(ascending=False).head(15)

fig_keywords = px.bar(
    x=top_keywords.values,
    y=top_keywords.index,
    orientation='h',
    labels={'x': 'Confianza (%)', 'y': 'Keyword'},
    title="Top 15 Keywords Más Relevantes",
    color=top_keywords.values,
    color_continuous_scale='Viridis'
)
fig_keywords.update_layout(
    height=500,
    showlegend=False,
    template='plotly_white'
)
st.plotly_chart(fig_keywords, use_container_width=True)

st.divider()

# Madrid Map
st.header("🗺️ Mapa de Madrid - Ubicaciones de Competidores")

# Create Madrid map with sample coffee shop locations
madrid_locations = pd.DataFrame({
    'name': [
        'Café Central', 'La Mallorquina', 'Café Gijón', 'Café Comercial',
        'Federal Café', 'Hanso Café', 'Toma Café', 'Misión Café',
        'Café de Oriente', 'Chocolatería San Ginés'
    ],
    'lat': [
        40.4168, 40.4169, 40.4273, 40.4290,
        40.4215, 40.4312, 40.4198, 40.4256,
        40.4188, 40.4165
    ],
    'lon': [
        -3.7038, -3.7033, -3.6914, -3.7051,
        -3.7025, -3.6895, -3.7089, -3.7015,
        -3.7142, -3.7068
    ],
    'reviews': [450, 380, 520, 410, 290, 310, 275, 190, 340, 580],
    'rating': [4.5, 4.3, 4.7, 4.4, 4.6, 4.5, 4.8, 4.4, 4.3, 4.6]
})

fig_map = px.scatter_mapbox(
    madrid_locations,
    lat='lat',
    lon='lon',
    hover_name='name',
    hover_data={'lat': False, 'lon': False, 'reviews': True, 'rating': True},
    color='rating',
    size='reviews',
    color_continuous_scale='RdYlGn',
    size_max=20,
    zoom=12,
    height=500,
    title="Cafeterías Analizadas en Madrid"
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(lat=40.4168, lon=-3.7038),
    ),
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# Detailed Insights by Category
st.header("📋 Insights Detallados por Categoría")

# Create tabs for each category
tabs = st.tabs(list(insights_data.keys()))

for tab, (category, data) in zip(tabs, insights_data.items()):
    with tab:
        st.markdown(f"## {data['icon']} {category.split(' ', 1)[1]}")

        # Category overview
        category_keywords = []
        category_confidences = []

        for question in data["questions"]:
            for rec in question["recommendations"]:
                category_keywords.append(rec["keyword"])
                category_confidences.append(rec["confidence"])

        # Category metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Preguntas", len(data["questions"]))
        with col2:
            st.metric("Keywords Únicas", len(set(category_keywords)))
        with col3:
            avg_conf = sum(category_confidences) / len(category_confidences) if category_confidences else 0
            st.metric("Confianza Promedio", f"{avg_conf:.2f}%")

        st.markdown("---")

        # Display each question with recommendations
        for idx, question in enumerate(data["questions"], 1):
            st.markdown(f"### ❓ Pregunta {idx}")
            st.info(question["question"])

            st.markdown("**💡 Recomendaciones:**")

            # Create visualization for this question's recommendations
            rec_data = pd.DataFrame(question["recommendations"])

            if len(rec_data) > 1:
                fig_rec = go.Figure()

                fig_rec.add_trace(go.Bar(
                    x=rec_data['keyword'],
                    y=rec_data['confidence'],
                    text=rec_data['confidence'].apply(lambda x: f'{x:.2f}%'),
                    textposition='auto',
                    marker=dict(
                        color=rec_data['confidence'],
                        colorscale='Viridis',
                        showscale=False
                    )
                ))

                fig_rec.update_layout(
                    title=f"Nivel de Confianza por Keyword",
                    xaxis_title="Keyword",
                    yaxis_title="Confianza (%)",
                    height=300,
                    template='plotly_white'
                )

                st.plotly_chart(fig_rec, use_container_width=True)

            # Display recommendations as cards
            for i, rec in enumerate(question["recommendations"], 1):
                confidence_color = "#28a745" if rec["confidence"] > 5 else "#ffc107" if rec["confidence"] > 1 else "#6c757d"

                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="margin: 0; color: #333;">#{i} - {rec['keyword'].upper()}</h4>
                    <p style="margin: 10px 0; color: #666; font-style: italic;">"{rec['text']}"</p>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="flex: 1; background-color: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden;">
                            <div style="background-color: {confidence_color}; height: 100%; width: {min(rec['confidence'] * 5, 100)}%; border-radius: 10px;"></div>
                        </div>
                        <strong style="color: {confidence_color};">{rec['confidence']:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

# Summary & Action Items
st.header("🎯 Plan de Acción Recomendado")

st.markdown("""
### 🏆 Top 3 Prioridades Basadas en los Datos:

1. **Calidad de Comida - "Healthy" (13.21% confianza)**
   - Los clientes valoran enormemente la comida saludable
   - **Acción:** Expandir opciones saludables en el menú y destacarlas

2. **Calidad del Café - "Delicious" (12.19% confianza)**
   - El sabor delicioso del café es altamente apreciado
   - **Acción:** Invertir en granos de alta calidad y formación de baristas

3. **Calidad de Comida - "Delicious" (10.66% confianza)**
   - Los clientes elogian frecuentemente la comida deliciosa
   - **Acción:** Mantener estándares de calidad consistentes

### 📈 Próximos Pasos:

- Revisar el menú actual y alinearlo con las preferencias identificadas
- Capacitar al personal en las áreas prioritarias (amabilidad, servicio)
- Implementar sistema de feedback para monitorear mejoras
- Realizar análisis comparativo mensual con competidores
""")
