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
        'Rating': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐, '⭐⭐],
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
        st.info("📍 **Map Location:** Place your 'my_map.html' file in the same directory to display the map.")

        # Placeholder map using plotly
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

# ============================================================
# PAGE 2: COMPETITOR ANALYSIS
# ============================================================
elif page == "🔍 Competitor Analysis":
    
    st.markdown('<div class="sub-header">Monitor customer satisfaction and competitive position</div>', unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Competitor Insights CSV", 
        type=['csv'],
        help="Upload the CSV file with competitor reviews analysis"
    )

    if uploaded_file is None:
        st.info("👈 Please upload your competitor insights CSV file using the sidebar")
        st.stop()

    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    df = load_data(uploaded_file)

    # Filter out meaningless keywords
    STOPWORDS = {
        'you', 'which', 'they', 'that', 'this', 'these', 'those', 'clasico',
        'everything', 'food', 'place', 'staff', 'madrid', 'coffee', 'also',
        'very', 'great', 'good', 'best', 'really', 'super', 'nice', 'lovely',
        'amazing', 'perfect', 'excellent', 'wonderful', 'fantastic'
    }

    def parse_top_products(top_products_str):
        """Parse the top_products string into a list of tuples (product, count)"""
        if pd.isna(top_products_str) or not top_products_str.strip():
            return []
        
        products = []
        for item in top_products_str.split('),'):
            item = item.strip()
            match = re.match(r'(.+?)\s*\((\d+)\)', item.replace(')', ''))
            if match:
                product_name = match.group(1).strip()
                count = int(match.group(2))
                if product_name.lower() not in STOPWORDS:
                    products.append((product_name, count))
        return products

    def get_overall_insights(df, limit=4):
        """Get general insights from AI summaries"""
        insights = []
        
        for _, row in df.iterrows():
            if row['ai_summary'] and str(row['ai_summary']).strip():
                # Extract keywords from this category to estimate importance
                keywords = parse_top_products(row.get('top_products', ''))
                keyword_count = sum(count for _, count in keywords[:3])  # Top 3 keywords
                
                # Calculate text length as another importance factor
                text_length = len(row['ai_summary'])
                importance_score = keyword_count * 10 + text_length
                
                insights.append({
                    'text': row['ai_summary'],
                    'restaurant': row['restaurant'],
                    'importance': importance_score
                })
        
        # Sort by importance and return top N
        insights.sort(key=lambda x: x['importance'], reverse=True)
        return insights[:limit]

    # Top Section - Map
    map_path = Path("images/mapa_cafeterias_seeccionadas_madrid_20260210.html")
    has_map = map_path.exists()
    
    if has_map:
        st.markdown('<div class="section-title">📍 Location Map</div>', unsafe_allow_html=True)
        with open("images/mapa_cafeterias_seeccionadas_madrid_20260210.html", 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=380, scrolling=False)
    else:
        st.info("💡 Place your 'my_map.html' file in the same directory to display the location map")

    st.markdown("---")

    # Two Column Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Key Customer Themes - General Insights
        st.markdown('<div class="section-title">Key Customer Themes</div>', unsafe_allow_html=True)
        
        # Get top general insights from AI summaries
        top_insights = get_overall_insights(df, limit=4)
        
        for insight in top_insights:
            st.markdown(f"""
            <div class="theme-item">
                <strong>🏪 {insight['restaurant']}</strong><br>
                {insight['text']}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Nearby Competitors
        st.markdown('<div class="section-title">Nearby Competitors</div>', unsafe_allow_html=True)
        
        # Initialize session state for selected restaurant
        if 'selected_restaurant' not in st.session_state:
            st.session_state.selected_restaurant = None
        
        restaurants = sorted(df['restaurant'].unique())
        
        # If no restaurant is selected, show the list
        if st.session_state.selected_restaurant is None:
            for restaurant in restaurants:
                restaurant_data = df[df['restaurant'] == restaurant]
                num_insights = len(restaurant_data)
                
                # Create clickable card
                if st.button(f"☕ {restaurant}", key=f"btn_{restaurant}", use_container_width=True):
                    st.session_state.selected_restaurant = restaurant
                    st.rerun()
                
                # Show quick preview
                st.caption(f"{num_insights} insights available")
                st.markdown("---")
        
        else:
            # Show detailed view for selected restaurant
            selected = st.session_state.selected_restaurant
            restaurant_data = df[df['restaurant'] == selected]
            
            # Back button
            if st.button("← Back to all competitors", key="back_btn"):
                st.session_state.selected_restaurant = None
                st.rerun()
            
            st.markdown(f"### {selected}")
            st.caption(f"{len(restaurant_data)} insights")
            st.markdown("---")
            
            # Collect all insights for this restaurant
            all_insights = []
            for _, row in restaurant_data.iterrows():
                if row['ai_summary'] and str(row['ai_summary']).strip():
                    keywords = parse_top_products(row.get('top_products', ''))
                    keyword_count = sum(count for _, count in keywords[:3])
                    
                    all_insights.append({
                        'text': row['ai_summary'],
                        'keywords': keywords[:5],
                        'importance': keyword_count
                    })
            
            # Sort by importance
            all_insights.sort(key=lambda x: x['importance'], reverse=True)
            
            # Show all insights as numbered items
            for i, insight in enumerate(all_insights, 1):
                st.markdown(f"""
                <div class="theme-item">
                    <strong>{i}.</strong> {insight['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show keywords if available
                if insight['keywords']:
                    keyword_text = ", ".join([f"{prod} ({count})" for prod, count in insight['keywords']])
                    st.caption(f"🔑 {keyword_text}")
                
                st.markdown("")  # Add spacing

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #999; padding: 10px; font-size: 0.85rem;'>
            Competitor Insights Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )
