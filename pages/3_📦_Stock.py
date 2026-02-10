import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Cafeteria Review Analytics",
    page_icon="☕",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">☕ Cafeteria Review Analytics Dashboard</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 1: REVIEW DISTRIBUTION BY STAR RATING
# ============================================================
st.markdown("---")
st.markdown("## 📊 Review Distribution")

# Star rating data (from your analysis)
star_data = {
    'Rating': ['⭐⭐⭐⭐⭐ Five Stars', '⭐⭐⭐⭐ Four Stars', '⭐⭐⭐ Three Stars', '⭐ One Star', '⭐⭐ Two Stars'],
    'Count': [1588, 497, 228, 113, 84],
    'Stars': [5, 4, 3, 1, 2]
}
df_stars = pd.DataFrame(star_data)
df_stars = df_stars.sort_values('Stars', ascending=False)

# Calculate percentages
total_reviews = df_stars['Count'].sum()
df_stars['Percentage'] = (df_stars['Count'] / total_reviews * 100).round(1)

# Display metrics in columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="⭐⭐⭐⭐⭐ Five Stars",
        value=f"{df_stars[df_stars['Stars']==5]['Count'].values[0]:,}",
        delta=f"{df_stars[df_stars['Stars']==5]['Percentage'].values[0]}%"
    )

with col2:
    st.metric(
        label="⭐⭐⭐⭐ Four Stars",
        value=f"{df_stars[df_stars['Stars']==4]['Count'].values[0]:,}",
        delta=f"{df_stars[df_stars['Stars']==4]['Percentage'].values[0]}%"
    )

with col3:
    st.metric(
        label="⭐⭐⭐ Three Stars",
        value=f"{df_stars[df_stars['Stars']==3]['Count'].values[0]:,}",
        delta=f"{df_stars[df_stars['Stars']==3]['Percentage'].values[0]}%"
    )

with col4:
    st.metric(
        label="⭐⭐ Two Stars",
        value=f"{df_stars[df_stars['Stars']==2]['Count'].values[0]:,}",
        delta=f"{df_stars[df_stars['Stars']==2]['Percentage'].values[0]}%"
    )

with col5:
    st.metric(
        label="⭐ One Star",
        value=f"{df_stars[df_stars['Stars']==1]['Count'].values[0]:,}",
        delta=f"{df_stars[df_stars['Stars']==1]['Percentage'].values[0]}%"
    )

# Visualization
col_left, col_right = st.columns(2)

with col_left:
    # Bar chart
    fig_bar = px.bar(
        df_stars,
        x='Rating',
        y='Count',
        title='Review Count by Star Rating',
        color='Stars',
        color_continuous_scale='RdYlGn',
        text='Count'
    )
    fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_bar.update_layout(showlegend=False, xaxis_title='', yaxis_title='Number of Reviews')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    # Pie chart
    fig_pie = px.pie(
        df_stars,
        values='Count',
        names='Rating',
        title='Review Distribution (%)',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# Total reviews summary
st.info(f"📊 **Total Reviews Analyzed:** {total_reviews:,} reviews | Average Polarity: 0.471")

# ============================================================
# SECTION 2: MAP WITH REVIEWS
# ============================================================
st.markdown("---")
st.markdown("## 🗺️ Restaurant Location & Review Heatmap")

# Placeholder for your map (replace with your actual map code)
# If you have a folium map saved as HTML, you can display it like this:
# with open('mymap.html', 'r') as f:
#     map_html = f.read()
# st.components.v1.html(map_html, height=600)

# For now, creating a sample map placeholder
st.info("📍 **Map Location:** Your restaurant map will be displayed here. Please provide your map HTML file or coordinates.")

# You can also create a simple map using plotly
# Example placeholder map:
import plotly.express as px

# Sample coordinates (replace with your actual coordinates)
map_data = pd.DataFrame({
    'lat': [40.7128],  # Replace with your latitude
    'lon': [-74.0060],  # Replace with your longitude
    'name': ['Your Cafeteria'],
    'reviews': [2510]
})

fig_map = px.scatter_mapbox(
    map_data,
    lat='lat',
    lon='lon',
    hover_name='name',
    hover_data={'reviews': True, 'lat': False, 'lon': False},
    zoom=14,
    height=500,
    size='reviews',
    size_max=30
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# SECTION 3: TABBED ANALYSIS PAGES
# ============================================================
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Your Cafeteria Analysis", "🔍 Detailed Insights"])

# ============================================================
# TAB 1: YOUR CAFETERIA ANALYSIS
# ============================================================
with tab1:
    st.markdown("## 📋 Executive Summary")
    
    # Key metrics at the top
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Best Aspect")
        st.success("**ATMOSPHERE**")
        st.metric("Score", "87.7%", "↑ Top rated")
        
    with col2:
        st.markdown("### ⚠️ Needs Attention")
        st.warning("**PRICE**")
        st.metric("Score", "54.1%", "↓ Lowest rated")
        
    with col3:
        st.markdown("### 📊 Overall")
        st.info("**POLARITY**")
        st.metric("Average", "0.471", "Positive")
    
    st.markdown("---")
    
    # Aspect performance overview
    st.markdown("### 🎯 Aspect Performance Overview")
    
    aspects_data = {
        'Aspect': ['ATMOSPHERE', 'FOOD', 'SERVICE', 'COFFEE', 'GENERAL', 'LOCATION', 'PRICE'],
        'Score': [87.7, 87.3, 84.8, 84.5, 81.9, 80.5, 54.1],
        'Positive %': [90.1, 90.3, 89.7, 88.7, 86.8, 86.2, 70.3],
        'Negative %': [2.5, 3.0, 4.9, 4.3, 4.9, 5.7, 16.2],
        'Total Mentions': [284, 709, 824, 656, 204, 174, 266]
    }
    df_aspects = pd.DataFrame(aspects_data)
    
    # Horizontal bar chart for aspect scores
    fig_aspects = go.Figure()
    
    colors = ['#27ae60' if score >= 80 else '#f39c12' if score >= 60 else '#e74c3c' 
              for score in df_aspects['Score']]
    
    fig_aspects.add_trace(go.Bar(
        y=df_aspects['Aspect'],
        x=df_aspects['Score'],
        orientation='h',
        marker=dict(color=colors),
        text=df_aspects['Score'].apply(lambda x: f'{x}%'),
        textposition='outside'
    ))
    
    fig_aspects.update_layout(
        title='Aspect Satisfaction Scores',
        xaxis_title='Score (%)',
        yaxis_title='',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_aspects, use_container_width=True)
    
    # Detailed breakdown
    st.markdown("---")
    st.markdown("### 📊 Detailed Aspect Breakdown")
    
    # Display the dataframe
    df_display = df_aspects.copy()
    df_display['Score'] = df_display['Score'].apply(lambda x: f"{x}%")
    df_display['Positive %'] = df_display['Positive %'].apply(lambda x: f"{x}%")
    df_display['Negative %'] = df_display['Negative %'].apply(lambda x: f"{x}%")
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Top problems and strengths
    st.markdown("---")
    col_prob, col_strength = st.columns(2)
    
    with col_prob:
        st.markdown("### 🔴 Top 3 Issues to Address")
        st.error("**1. Long Wait Times**")
        st.write("359 mentions across all aspects")
        st.write("💡 Recommendation: Optimize service times - consider more staff")
        
        st.error("**2. Overpriced**")
        st.write("273 mentions")
        st.write("💡 Recommendation: Evaluate pricing strategy vs. competition")
        
        st.error("**3. Rude Staff**")
        st.write("64 mentions")
        st.write("💡 Recommendation: Implement customer service training")
    
    with col_strength:
        st.markdown("### 🟢 Top 3 Strengths to Maintain")
        st.success("**1. Friendly Staff**")
        st.write("990 mentions - Keep up the great work!")
        
        st.success("**2. Fresh Food**")
        st.write("893 mentions - Quality is recognized")
        
        st.success("**3. Cozy Atmosphere**")
        st.write("418 mentions - Ambiance is a strength")

# ============================================================
# TAB 2: DETAILED INSIGHTS
# ============================================================
with tab2:
    st.markdown("## 🔍 Actionable Insights by Aspect")
    
    # Create expandable sections for each aspect
    aspects = [
        {
            'name': 'SERVICE',
            'score': 84.8,
            'priority': 'LOW',
            'pos': 89.7,
            'neg': 4.9,
            'mentions': 824,
            'problems': [
                ('long wait', 133),
                ('overpriced', 42),
                ('rude staff', 28)
            ],
            'strengths': [
                ('friendly staff', 344),
                ('fresh food', 230),
                ('cozy atmosphere', 90)
            ],
            'recommendation': 'Optimize service times - consider more staff'
        },
        {
            'name': 'FOOD',
            'score': 87.3,
            'priority': 'LOW',
            'pos': 90.3,
            'neg': 3.0,
            'mentions': 709,
            'problems': [
                ('long wait', 66),
                ('overpriced', 44),
                ('rude staff', 7)
            ],
            'strengths': [
                ('fresh food', 250),
                ('friendly staff', 209),
                ('cozy atmosphere', 63)
            ],
            'recommendation': 'Optimize service times - consider more staff'
        },
        {
            'name': 'COFFEE',
            'score': 84.5,
            'priority': 'LOW',
            'pos': 88.7,
            'neg': 4.3,
            'mentions': 656,
            'problems': [
                ('long wait', 56),
                ('overpriced', 41),
                ('rude staff', 9)
            ],
            'strengths': [
                ('fresh food', 183),
                ('friendly staff', 169),
                ('great coffee', 72)
            ],
            'recommendation': 'Optimize service times - consider more staff'
        },
        {
            'name': 'ATMOSPHERE',
            'score': 87.7,
            'priority': 'LOW',
            'pos': 90.1,
            'neg': 2.5,
            'mentions': 284,
            'problems': [
                ('long wait', 31),
                ('overpriced', 10),
                ('rude staff', 3)
            ],
            'strengths': [
                ('cozy atmosphere', 106),
                ('friendly staff', 94),
                ('fresh food', 78)
            ],
            'recommendation': 'Optimize service times - consider more staff'
        },
        {
            'name': 'PRICE',
            'score': 54.1,
            'priority': 'HIGH',
            'pos': 70.3,
            'neg': 16.2,
            'mentions': 266,
            'problems': [
                ('overpriced', 97),
                ('long wait', 26),
                ('rude staff', 5)
            ],
            'strengths': [
                ('friendly staff', 61),
                ('fresh food', 57),
                ('good value', 36)
            ],
            'recommendation': 'Evaluate pricing strategy vs. competition'
        },
        {
            'name': 'GENERAL',
            'score': 81.9,
            'priority': 'LOW',
            'pos': 86.8,
            'neg': 4.9,
            'mentions': 204,
            'problems': [
                ('long wait', 4)
            ],
            'strengths': [
                ('fresh food', 22),
                ('friendly staff', 17),
                ('cozy atmosphere', 1)
            ],
            'recommendation': 'Optimize service times - consider more staff'
        },
        {
            'name': 'LOCATION',
            'score': 80.5,
            'priority': 'LOW',
            'pos': 86.2,
            'neg': 5.7,
            'mentions': 174,
            'problems': [
                ('overpriced', 23),
                ('long wait', 21),
                ('rude staff', 4)
            ],
            'strengths': [
                ('friendly staff', 49),
                ('fresh food', 31),
                ('cozy atmosphere', 22)
            ],
            'recommendation': 'Evaluate pricing strategy vs. competition'
        }
    ]
    
    for i, aspect in enumerate(aspects, 1):
        with st.expander(f"**{i}. {aspect['name']}** - Score: {aspect['score']}% | Mentions: {aspect['mentions']}", expanded=(i<=3)):
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Satisfaction Score", f"{aspect['score']}%")
            with col2:
                st.metric("Positive", f"{aspect['pos']}%")
            with col3:
                st.metric("Negative", f"{aspect['neg']}%")
            with col4:
                priority_color = "🔴" if aspect['priority'] == 'HIGH' else "🟡" if aspect['priority'] == 'MEDIUM' else "🟢"
                st.metric("Priority", f"{priority_color} {aspect['priority']}")
            
            # Sentiment distribution chart
            fig_sentiment = go.Figure(data=[
                go.Bar(name='Positive', x=[aspect['name']], y=[aspect['pos']], marker_color='#27ae60'),
                go.Bar(name='Negative', x=[aspect['name']], y=[aspect['neg']], marker_color='#e74c3c'),
                go.Bar(name='Neutral', x=[aspect['name']], y=[100 - aspect['pos'] - aspect['neg']], marker_color='#95a5a6')
            ])
            fig_sentiment.update_layout(
                barmode='stack',
                title=f'{aspect["name"]} - Sentiment Distribution',
                yaxis_title='Percentage (%)',
                showlegend=True,
                height=250
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Problems and Strengths side by side
            col_p, col_s = st.columns(2)
            
            with col_p:
                st.markdown("##### ⚠️ Problems Detected")
                for problem, count in aspect['problems']:
                    st.write(f"• **{problem}**: {count} times")
            
            with col_s:
                st.markdown("##### ✅ Strengths")
                for strength, count in aspect['strengths']:
                    st.write(f"• **{strength}**: {count} times")
            
            # Recommendation
            st.info(f"💡 **Recommendation:** {aspect['recommendation']}")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>📊 Analysis based on 1,621 valid reviews out of 2,510 total reviews</p>
    <p>🔄 Last updated: February 2026</p>
</div>
""", unsafe_allow_html=True)
