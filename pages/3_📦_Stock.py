import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
from pathlib import Path
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Cafeteria Review Analytics",
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
    .theme-item {
        background: #fef3c7;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .stExpander {
        background: #fef3c7;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Your Cafeteria", "🔍 Competitor Analysis"],
    label_visibility="collapsed"
)

# Main header
st.markdown('<div class="main-header">☕ Cafeteria Review Analytics Dashboard</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 1: YOUR CAFETERIA
# ============================================================
if page == "🏠 Your Cafeteria":
    
    # ============================================================
    # SECTION 1: REVIEW DISTRIBUTION BY STAR RATING
    # ============================================================
    st.markdown("---")
    st.markdown("## 📊 Review Distribution")

    # Star rating data (from your analysis)
    star_data = {
        'Rating': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐', '⭐⭐'],
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

    # Total reviews summary
    st.info(f"📊 **Total Reviews Analyzed:** {total_reviews:,} reviews | Average Polarity: 0.471")

    # ============================================================
    # SECTION 2: MAP WITH REVIEWS
    # ============================================================
    st.markdown("---")
    st.markdown("## 🗺️ Restaurant Location & Review Heatmap")

    # Check if map file exists
    map_path = Path("images/mapa_cafeterias_seeccionadas_madrid_20260210.html")
    has_map = map_path.exists()

    if has_map:
        with open("images/mapa_cafeterias_seeccionadas_madrid_20260210.html", 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=380, scrolling=False)
    else:
        st.info("📍 **Map Location:** Place your map HTML file in images/ directory to display the map.")

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

# ============================================================
# PAGE 2: COMPETITOR ANALYSIS - COMPLETELY NEW
# ============================================================
elif page == "🔍 Competitor Analysis":
    
    st.markdown('<div class="sub-header">Analyze what customers love across Madrid\'s top coffee shops</div>', unsafe_allow_html=True)

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    @st.cache_data
    def load_insights_data():
        """Load all insights data files"""
        try:
            # Per restaurant insights
            per_restaurant = pd.read_csv('final_top_10_insights.csv')
            
            # Common insights across all restaurants
            common_insights = pd.read_csv('common_insights_all_restaurants.csv')
            
            # Category breakdown
            category_insights = pd.read_csv('top_insights_by_category.csv')
            
            return per_restaurant, common_insights, category_insights
        except FileNotFoundError as e:
            st.error(f"⚠️ Data file not found: {e}")
            st.info("📁 Please ensure these CSV files are in the same directory:")
            st.code("""
- final_top_10_insights.csv
- common_insights_all_restaurants.csv
- top_insights_by_category.csv
            """)
            return None, None, None

    per_rest_df, common_df, category_df = load_insights_data()

    if per_rest_df is None:
        st.stop()

    # ============================================================================
    # TOP SECTION - KEY METRICS
    # ============================================================================

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
            f"{top_insight['mentions']} times",
            help="Most frequently mentioned across all restaurants"
        )

    with col4:
        most_common = common_df.nlargest(1, 'num_restaurants').iloc[0]
        st.metric(
            "Most Common", 
            most_common['insight'].title(),
            f"{most_common['num_restaurants']}/12 shops",
            help="Found in the most restaurants"
        )

    st.markdown("---")

    # ============================================================================
    # LOCATION MAP
    # ============================================================================

    map_path = Path("images/mapa_cafeterias_seeccionadas_madrid_20260210.html")
    if map_path.exists():
        st.markdown("## 📍 Location Map")
        with open(map_path, 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=400, scrolling=False)
        st.markdown("---")

    # ============================================================================
    # MAIN CONTENT - TABS
    # ============================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Insights", 
        "🏪 By Restaurant", 
        "📊 Category Analysis",
        "🔍 Comparison Matrix"
    ])

    # Category colors (used across tabs)
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

    # ============================================================================
    # TAB 1: TOP INSIGHTS ACROSS ALL RESTAURANTS
    # ============================================================================

    with tab1:
        st.markdown("### 🏆 What Customers Love Most")
        st.markdown("*Top mentions across all 12 specialty coffee shops in Madrid*")
        
        # Overall top 20
        st.markdown("#### Overall Top 20 Insights")
        
        top_20 = common_df.head(20).copy()
        
        # Create horizontal bar chart
        fig_top20 = go.Figure()
        
        colors = [category_colors.get(cat, '#95A5A6') for cat in top_20['category']]
        
        fig_top20.add_trace(go.Bar(
            y=top_20['insight'],
            x=top_20['mentions'],
            orientation='h',
            marker=dict(color=colors),
            text=top_20['mentions'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Mentions: %{x}<extra></extra>'
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
        
        # Show legend for categories
        st.markdown("**Category Legend:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🍽️ **Food** | ☕ **Coffee**")
        with col2:
            st.markdown("🍵 **Specialty Drinks** | 👥 **Service**")
        with col3:
            st.markdown("⭐ **Features** | 🏠 **Atmosphere**")
        
        st.markdown("---")
        
        # Most widespread insights
        st.markdown("#### 🌐 Most Common Across Restaurants")
        st.markdown("*Insights found in multiple coffee shops*")
        
        widespread = common_df.nlargest(10, 'num_restaurants')
        
        fig_widespread = go.Figure()
        
        fig_widespread.add_trace(go.Bar(
            x=widespread['insight'],
            y=widespread['num_restaurants'],
            marker=dict(color='#3498db'),
            text=widespread['num_restaurants'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Found in %{y}/12 restaurants<extra></extra>'
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
        
        # Key takeaways
        st.info(f"""
        **💡 Key Takeaways:**
        - **"{widespread.iloc[0]['insight'].title()}"** is mentioned in {widespread.iloc[0]['num_restaurants']} out of 12 restaurants
        - Breakfast items are highly valued across the market
        - Specialty coffee drinks (matcha, iced latte, flat white) are essential offerings
        - Friendly service is consistently mentioned as important
        """)

    # ============================================================================
    # TAB 2: INSIGHTS BY RESTAURANT
    # ============================================================================

    with tab2:
        st.markdown("### 🏪 Restaurant-Specific Insights")
        st.markdown("*Top 10 customer favorites at each coffee shop*")
        
        # Restaurant selector
        restaurants = sorted(per_rest_df['restaurant'].unique())
        selected_restaurant = st.selectbox(
            "Select a restaurant to view details:",
            restaurants,
            key='restaurant_selector'
        )
        
        # Filter data for selected restaurant
        rest_data = per_rest_df[per_rest_df['restaurant'] == selected_restaurant].copy()
        rest_data = rest_data.sort_values('mentions', ascending=False)
        
        # Display restaurant name prominently
        st.markdown(f"## ☕ {selected_restaurant}")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Insight", rest_data.iloc[0]['insight'].title())
        with col2:
            st.metric("Most Mentions", rest_data.iloc[0]['mentions'])
        with col3:
            category_dist = rest_data['category'].value_counts()
            st.metric("Main Category", category_dist.index[0].replace('_', ' ').title())
        
        st.markdown("---")
        
        # Category breakdown for this restaurant
        st.markdown("#### 📊 Insights by Category")
        
        categories_order = ['food', 'coffee', 'specialty_drink', 'service', 'feature', 'atmosphere']
        
        for category in categories_order:
            cat_items = rest_data[rest_data['category'] == category]
            
            if len(cat_items) > 0:
                with st.expander(f"{category_labels.get(category, category)} ({len(cat_items)} items)", expanded=True):
                    for idx, row in cat_items.iterrows():
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**{row['insight'].title()}**")
                        with cols[1]:
                            st.markdown(f"*{row['mentions']} mentions*")
        
        st.markdown("---")
        
        # Visual breakdown
        fig_restaurant = px.bar(
            rest_data.head(10),
            x='mentions',
            y='insight',
            orientation='h',
            color='category',
            color_discrete_map=category_colors,
            title=f'Top 10 Insights at {selected_restaurant}'
        )
        
        fig_restaurant.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_restaurant, use_container_width=True)

    # ============================================================================
    # TAB 3: CATEGORY ANALYSIS
    # ============================================================================

    with tab3:
        st.markdown("### 📊 Category Deep Dive")
        st.markdown("*Understanding what matters most to customers*")
        
        # Overall category distribution
        st.markdown("#### Category Distribution Across All Restaurants")
        
        category_totals = per_rest_df.groupby('category')['mentions'].sum().reset_index()
        category_totals = category_totals.sort_values('mentions', ascending=False)
        
        # Pie chart
        fig_pie = px.pie(
            category_totals,
            values='mentions',
            names='category',
            title='Total Mentions by Category',
            color='category',
            color_discrete_map=category_colors
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### Category Statistics")
            for _, row in category_totals.iterrows():
                pct = (row['mentions'] / category_totals['mentions'].sum() * 100)
                st.metric(
                    category_labels.get(row['category'], row['category']),
                    f"{row['mentions']} mentions",
                    f"{pct:.1f}%"
                )
        
        st.markdown("---")
        
        # Top items per category
        st.markdown("#### Top Items by Category")
        
        category_selector = st.selectbox(
            "Select category:",
            list(category_labels.values()),
            key='category_selector'
        )
        
        # Reverse lookup category key
        selected_cat_key = [k for k, v in category_labels.items() if v == category_selector][0]
        
        # Get top items in this category
        cat_items = category_df[category_df['category'] == selected_cat_key].copy()
        cat_items = cat_items.sort_values('total_mentions', ascending=False).head(10)
        
        if len(cat_items) > 0:
            fig_cat = go.Figure()
            
            fig_cat.add_trace(go.Bar(
                x=cat_items['insight'],
                y=cat_items['total_mentions'],
                marker_color=category_colors.get(selected_cat_key, '#95A5A6'),
                text=cat_items['total_mentions'],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y} mentions<br>Found in %{customdata} restaurants<extra></extra>',
                customdata=cat_items['num_restaurants']
            ))
            
            fig_cat.update_layout(
                title=f'Top Items in {category_selector}',
                xaxis_title='',
                yaxis_title='Total Mentions',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Detailed table
            st.markdown("**Detailed Breakdown:**")
            display_df = cat_items[['insight', 'total_mentions', 'num_restaurants']].copy()
            display_df.columns = ['Item', 'Total Mentions', 'Found in # Restaurants']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ============================================================================
    # TAB 4: COMPARISON MATRIX
    # ============================================================================

    with tab4:
        st.markdown("### 🔍 Restaurant Comparison Matrix")
        st.markdown("*Compare insights across different coffee shops*")
        
        # Select insights to compare
        top_insights_list = common_df.head(15)['insight'].tolist()
        
        selected_insights = st.multiselect(
            "Select insights to compare (up to 8):",
            top_insights_list,
            default=top_insights_list[:5],
            max_selections=8,
            key='insight_comparison'
        )
        
        if selected_insights:
            # Create comparison data
            comparison_data = []
            restaurants = sorted(per_rest_df['restaurant'].unique())
            
            for insight in selected_insights:
                row_data = {'Insight': insight}
                for restaurant in restaurants:
                    rest_insight = per_rest_df[
                        (per_rest_df['restaurant'] == restaurant) & 
                        (per_rest_df['insight'] == insight)
                    ]
                    
                    if len(rest_insight) > 0:
                        row_data[restaurant] = rest_insight.iloc[0]['mentions']
                    else:
                        row_data[restaurant] = 0
                
                comparison_data.append(row_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=comparison_df[restaurants].values,
                x=restaurants,
                y=comparison_df['Insight'],
                colorscale='YlOrRd',
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}<br>Mentions: %{z}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='Insight Comparison Heatmap',
                xaxis_title='',
                yaxis_title='',
                height=max(400, len(selected_insights) * 50),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Show table
            st.markdown("#### Detailed Comparison Table")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data",
                data=csv,
                file_name="restaurant_comparison.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Select insights above to see the comparison")
        
        st.markdown("---")
        
        # Restaurant-to-restaurant comparison
        st.markdown("#### Restaurant Head-to-Head Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rest1 = st.selectbox("Select first restaurant:", restaurants, key='rest1')
        with col2:
            rest2 = st.selectbox("Select second restaurant:", 
                                [r for r in restaurants if r != rest1], 
                                key='rest2')
        
        if rest1 and rest2:
            # Get data for both restaurants
            rest1_data = per_rest_df[per_rest_df['restaurant'] == rest1].set_index('insight')['mentions']
            rest2_data = per_rest_df[per_rest_df['restaurant'] == rest2].set_index('insight')['mentions']
            
            # Combine
            all_insights = list(set(rest1_data.index) | set(rest2_data.index))
            
            comparison = pd.DataFrame({
                rest1: [rest1_data.get(i, 0) for i in all_insights],
                rest2: [rest2_data.get(i, 0) for i in all_insights]
            }, index=all_insights)
            
            # Only show items mentioned by at least one restaurant
            comparison = comparison[(comparison[rest1] > 0) | (comparison[rest2] > 0)]
            comparison = comparison.sort_values(rest1, ascending=False).head(15)
            
            # Grouped bar chart
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
                title=f'{rest1} vs {rest2} - Top Insights',
                xaxis_title='',
                yaxis_title='Mentions',
                barmode='group',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

    # ============================================================================
    # FOOTER
    # ============================================================================

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>📊 Analysis of 12 specialty coffee shops in Madrid</p>
        <p>🔄 Based on customer review insights | February 2026</p>
    </div>
    """, unsafe_allow_html=True)
