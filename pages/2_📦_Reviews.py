import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit.components.v1 as components
import pickle
import hashlib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Reviews analysis",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .sub-header {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PASSWORD PROTECTION & DATA PERSISTENCE
# ============================================================================

PASSWORD = "PATIO"
DATA_DIR = Path("saved_data")
DATA_DIR.mkdir(exist_ok=True)

def hash_password(pwd):
    """Hash password for storage"""
    return hashlib.sha256(pwd.encode()).hexdigest()

def save_data(data, password):
    """Save uploaded data with password hash"""
    pwd_hash = hash_password(password)
    data_file = DATA_DIR / f"{pwd_hash}.pkl"
    
    save_dict = {
        'timestamp': datetime.now(),
        'per_restaurant': data['per_restaurant'],
        'common_insights': data['common_insights'],
        'category_insights': data['category_insights']
    }
    
    with open(data_file, 'wb') as f:
        pickle.dump(save_dict, f)
    
    return True

def load_data(password):
    """Load saved data if it exists"""
    pwd_hash = hash_password(password)
    data_file = DATA_DIR / f"{pwd_hash}.pkl"
    
    if data_file.exists():
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data
    return None

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Enter Password to Access Deatiled Analysis", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info(f"Password is {len(PASSWORD)} letters ")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Enter Password to Access Dashboard", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Check password first
if not check_password():
    st.stop()

# ============================================================================
# DATA LOADING LOGIC
# ============================================================================

# Try to load existing data
saved_data = load_data(PASSWORD)

if saved_data is not None:
    # Data exists - load it
    st.sidebar.success(f"✅ Data loaded (uploaded on {saved_data['timestamp'].strftime('%Y-%m-%d %H:%M')})")
    
    # Option to re-upload
    if st.sidebar.button("🔄 Upload New Data"):
        st.session_state['force_upload'] = True
        st.rerun()
    
    # Use saved data
    if 'force_upload' not in st.session_state:
        per_rest_df = saved_data['per_restaurant']
        common_df = saved_data['common_insights']
        category_df = saved_data['category_insights']
        data_loaded = True
    else:
        data_loaded = False
else:
    data_loaded = False

# If no saved data or user wants to re-upload
if not data_loaded:
    st.sidebar.info("Upload your insights files")
    
    uploaded_per_rest = st.sidebar.file_uploader(
        "Upload: final_top_10_insights.csv",
        type=['csv'],
        key='per_rest'
    )
    
    uploaded_common = st.sidebar.file_uploader(
        "Upload: common_insights_all_restaurants.csv",
        type=['csv'],
        key='common'
    )
    
    uploaded_category = st.sidebar.file_uploader(
        "Upload: top_insights_by_category.csv",
        type=['csv'],
        key='category'
    )
    
    if uploaded_per_rest and uploaded_common and uploaded_category:
        try:
            per_rest_df = pd.read_csv(uploaded_per_rest)
            common_df = pd.read_csv(uploaded_common)
            category_df = pd.read_csv(uploaded_category)
            
            # Save the data
            save_data({
                'per_restaurant': per_rest_df,
                'common_insights': common_df,
                'category_insights': category_df
            }, PASSWORD)
            
            st.sidebar.success("✅ Data uploaded and saved!")
            st.sidebar.info("💾 Data will be available on your next visit with the same password")
            
            if 'force_upload' in st.session_state:
                del st.session_state['force_upload']
            
            data_loaded = True
            
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}")
            st.stop()
    else:
        st.info("👈 Please upload all 3 CSV files using the sidebar")
        st.markdown("""
        **Required files:**
        1. `final_top_10_insights.csv`
        2. `common_insights_all_restaurants.csv`
        3. `top_insights_by_category.csv`
        
        After uploading, the data will be saved and you won't need to upload again when you access with the same password.
        """)
        st.stop()

# ============================================================================
# NAVIGATION
# ============================================================================

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Patio Vertical", "🔍 Competitor Analysis"],
    label_visibility="collapsed"
)

# Main header
st.markdown('<div class="main-header">☕ Cafeteria Review Analytics Dashboard</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 1: YOUR CAFETERIA
# ============================================================
if page == "🏠 Patio Vertical":
    
    st.markdown("---")
    st.markdown("## 📊 Review Distribution")

    # Star rating data
    star_data = {
        'Rating': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐', '⭐⭐'],
        'Count': [1588, 497, 228, 113, 84],
        'Stars': [5, 4, 3, 1, 2]
    }
    df_stars = pd.DataFrame(star_data)
    df_stars = df_stars.sort_values('Stars', ascending=False)

    total_reviews = df_stars['Count'].sum()
    df_stars['Percentage'] = (df_stars['Count'] / total_reviews * 100).round(1)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="⭐⭐⭐⭐⭐",
            value=f"{df_stars[df_stars['Stars']==5]['Count'].values[0]:,}",
            delta=f"{df_stars[df_stars['Stars']==5]['Percentage'].values[0]}%"
        )

    with col2:
        st.metric(
            label="⭐⭐⭐⭐",
            value=f"{df_stars[df_stars['Stars']==4]['Count'].values[0]:,}",
            delta=f"{df_stars[df_stars['Stars']==4]['Percentage'].values[0]}%"
        )

    with col3:
        st.metric(
            label="⭐⭐⭐",
            value=f"{df_stars[df_stars['Stars']==3]['Count'].values[0]:,}",
            delta=f"{df_stars[df_stars['Stars']==3]['Percentage'].values[0]}%"
        )

    with col4:
        st.metric(
            label="⭐⭐",
            value=f"{df_stars[df_stars['Stars']==2]['Count'].values[0]:,}",
            delta=f"{df_stars[df_stars['Stars']==2]['Percentage'].values[0]}%"
        )

    with col5:
        st.metric(
            label="⭐",
            value=f"{df_stars[df_stars['Stars']==1]['Count'].values[0]:,}",
            delta=f"{df_stars[df_stars['Stars']==1]['Percentage'].values[0]}%"
        )

    st.info(f"📊 **Total Reviews Analyzed:** {total_reviews:,} reviews | Average Point: 4.7⭐")


    st.markdown("---")
    st.markdown("## Analysis of Google Maps Reviews")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊General Insights", "🔍 Detailed Insights"])

    with tab1:
        st.markdown("## 📋 Executive Summary")
        
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
            st.info("**SCORE**")
            st.metric("Average", "4.7", "Positive")
        
        st.markdown("---")
        st.markdown("### 🎯 Aspect Performance Overview")
        
        aspects_data = {
            'Aspect': ['ATMOSPHERE', 'FOOD', 'SERVICE', 'COFFEE', 'GENERAL', 'LOCATION', 'PRICE'],
            'Score': [87.7, 87.3, 84.8, 84.5, 81.9, 80.5, 54.1],
            'Positive %': [90.1, 90.3, 89.7, 88.7, 86.8, 86.2, 70.3],
            'Negative %': [2.5, 3.0, 4.9, 4.3, 4.9, 5.7, 16.2],
            'Total Mentions': [284, 709, 824, 656, 204, 174, 266]
        }
        df_aspects = pd.DataFrame(aspects_data)
        
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
        
        st.markdown("---")
        col_prob, col_strength = st.columns(2)
        
        with col_prob:
            st.markdown("### 🔴 Top 3 Issues to Address")
            st.error("**1. Long Wait Times**")
            st.write("359 mentions across all aspects")
            st.write("💡 Recommendation: Optimize service times")
            
            st.error("**2. Overpriced**")
            st.write("273 mentions")
            st.write("💡 Recommendation: Evaluate pricing strategy")
            
            st.error("**3. Rude Staff**")
            st.write("64 mentions")
            st.write("💡 Recommendation: Customer service training")
        
        with col_strength:
            st.markdown("### 🟢 Top 3 Strengths to Maintain")
            st.success("**1. Friendly Staff**")
            st.write("990 mentions - Keep it up!")
            
            st.success("**2. Fresh Food**")
            st.write("893 mentions - Quality recognized")
            
            st.success("**3. Cozy Atmosphere**")
            st.write("418 mentions - Great ambiance")

    with tab2:
        st.markdown("## 🔍 Actionable Insights by Aspect")
        st.info("Detailed aspect analysis available - expand each section below")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>📊 Analysis based on 2,510 reviews | 🔄 February 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: COMPETITOR ANALYSIS
# ============================================================
elif page == "🔍 Competitor Analysis":
    
    st.markdown('<div class="sub-header">Analyze what customers love across Madrid\'s top coffee shops</div>', unsafe_allow_html=True)

    st.markdown("## 📊 Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Restaurants", 
            len(per_rest_df['restaurant'].unique()),
            help="Specialty coffee shops analyzed"
        )

    with col2:
        st.metric(
            "Unique Insights", 
            len(common_df),
            help="Different products/features mentioned"
        )

    with col3:
        top_insight = common_df.iloc[0]
        st.metric(
            "Most Mentioned", 
            top_insight['insight'].title(),
            f"{top_insight['mentions']} times"
        )

    with col4:
        most_common = common_df.nlargest(1, 'num_restaurants').iloc[0]
        st.metric(
            "Most Common", 
            most_common['insight'].title(),
            f"{most_common['num_restaurants']}/{len(per_rest_df['restaurant'].unique())} shops"
        )

    st.markdown("---")

    map_path = Path("images/mapa_cafeterias_seeccionadas_madrid_20260210.html")
    if map_path.exists():
        st.markdown("## 📍Map with average scores of most rated Madrid Cafeterias")
        with open(map_path, 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=400, scrolling=False)
        st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Insights", 
        "🏪 By Restaurant", 
        "📊 Category Analysis",
        "🔍 Comparison"
    ])

    category_colors = {
        'food': '#FF6B6B',
        'coffee': '#8B4513',
        'specialty_drink': '#4ECDC4',
        'service': '#FFE66D',
        'feature': '#A8E6CF',
        'atmosphere': '#C7CEEA'
    }
    
    category_labels = {
        'food': '🍽️ Food',
        'coffee': '☕ Coffee',
        'specialty_drink': '🍵 Specialty Drinks',
        'service': '👥 Service',
        'feature': '⭐ Features',
        'atmosphere': '🏠 Atmosphere'
    }

    with tab1:
        st.markdown("### 🏆 What Customers Love Most")
        st.markdown("*Top mentions across all specialty coffee shops*")
        
        top_20 = common_df.head(20).copy()
        
        fig_top20 = go.Figure()
        
        colors = [category_colors.get(cat, '#95A5A6') for cat in top_20['category']]
        
        fig_top20.add_trace(go.Bar(
            y=top_20['insight'],
            x=top_20['mentions'],
            orientation='h',
            marker=dict(color=colors),
            text=top_20['mentions'],
            textposition='outside'
        ))
        
        fig_top20.update_layout(
            title='Top 20 Most Mentioned Items',
            xaxis_title='Total Mentions',
            yaxis_title='',
            height=700,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_top20, use_container_width=True)
        
        st.markdown("---")
        
        widespread = common_df.nlargest(10, 'num_restaurants')
        
        fig_widespread = go.Figure()
        
        fig_widespread.add_trace(go.Bar(
            x=widespread['insight'],
            y=widespread['num_restaurants'],
            marker=dict(color='#3498db'),
            text=widespread['num_restaurants'],
            textposition='outside'
        ))
        
        fig_widespread.update_layout(
            title='Insights Present in Multiple Restaurants',
            xaxis_title='',
            yaxis_title='Number of Restaurants',
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_widespread, use_container_width=True)

    with tab2:
        st.markdown("### 🏪 Restaurant-Specific Insights")
        
        restaurants = sorted(per_rest_df['restaurant'].unique())
        selected_restaurant = st.selectbox(
            "Select a restaurant:",
            restaurants
        )
        
        rest_data = per_rest_df[per_rest_df['restaurant'] == selected_restaurant].copy()
        rest_data = rest_data.sort_values('mentions', ascending=False)
        
        st.markdown(f"## ☕ {selected_restaurant}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Insight", rest_data.iloc[0]['insight'].title())
        with col2:
            st.metric("Most Mentions", rest_data.iloc[0]['mentions'])
        with col3:
            category_dist = rest_data['category'].value_counts()
            st.metric("Main Category", category_dist.index[0].replace('_', ' ').title())
        
        st.markdown("---")
        
        for category in ['food', 'coffee', 'specialty_drink', 'service']:
            cat_items = rest_data[rest_data['category'] == category]
            
            if len(cat_items) > 0:
                with st.expander(f"{category_labels.get(category)} ({len(cat_items)} items)", expanded=True):
                    for idx, row in cat_items.iterrows():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**{row['insight'].title()}**")
                        with cols[1]:
                            st.markdown(f"*{row['mentions']} mentions*")

    with tab3:
        st.markdown("### 📊 Category Deep Dive")
        
        category_totals = per_rest_df.groupby('category')['mentions'].sum().reset_index()
        category_totals = category_totals.sort_values('mentions', ascending=False)
        
        fig_pie = px.pie(
            category_totals,
            values='mentions',
            names='category',
            title='Total Mentions by Category',
            color='category',
            color_discrete_map=category_colors
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab4:
        st.markdown("### 🔍 Restaurant Comparison")
        
        restaurants = sorted(per_rest_df['restaurant'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            rest1 = st.selectbox("First restaurant:", restaurants, key='r1')
        with col2:
            rest2 = st.selectbox("Second restaurant:", 
                                [r for r in restaurants if r != rest1], 
                                key='r2')
        
        if rest1 and rest2:
            rest1_data = per_rest_df[per_rest_df['restaurant'] == rest1].set_index('insight')['mentions']
            rest2_data = per_rest_df[per_rest_df['restaurant'] == rest2].set_index('insight')['mentions']
            
            all_insights = list(set(rest1_data.index) | set(rest2_data.index))
            
            comparison = pd.DataFrame({
                rest1: [rest1_data.get(i, 0) for i in all_insights],
                rest2: [rest2_data.get(i, 0) for i in all_insights]
            }, index=all_insights)
            
            comparison = comparison[(comparison[rest1] > 0) | (comparison[rest2] > 0)]
            comparison = comparison.sort_values(rest1, ascending=False).head(15)
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name=rest1,
                x=comparison.index,
                y=comparison[rest1],
                marker_color='#3498db'
            ))
            
            fig_comparison.add_trace(go.Bar(
                name=rest2,
                x=comparison.index,
                y=comparison[rest2],
                marker_color='#e74c3c'
            ))
            
            fig_comparison.update_layout(
                title=f'{rest1} vs {rest2}',
                xaxis_title='',
                yaxis_title='Mentions',
                barmode='group',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>📊 Based on customer review insights | 🔄 February 2026</p>
    </div>
    """, unsafe_allow_html=True)
