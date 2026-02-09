import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from pathlib import Path
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Sentiment & Competition Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
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

# File upload section
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload your competitor insights CSV", 
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

# Check if map file exists
map_path = Path("my_map.html")
has_map = map_path.exists()

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

# Main Dashboard
st.markdown('<div class="main-header">Sentiment & Competition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Monitor customer satisfaction and competitive position</div>', unsafe_allow_html=True)

# Top Section - Map
if has_map:
    st.markdown('<div class="section-title">📍 Location Map</div>', unsafe_allow_html=True)
    with open("my_map.html", 'r', encoding='utf-8') as f:
        map_html = f.read()
    components.html(map_html, height=400, scrolling=True)
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
            num_categories = len(restaurant_data)
            
            # Create clickable card
            if st.button(f"☕ {restaurant}", key=f"btn_{restaurant}", use_container_width=True):
                st.session_state.selected_restaurant = restaurant
                st.rerun()
            
            # Show quick preview
            st.caption(f"{num_categories} categories analyzed")
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
        st.caption(f"{len(restaurant_data)} categories")
        st.markdown("---")
        
        # Get top insights for this restaurant
        restaurant_insights = []
        for _, row in restaurant_data.iterrows():
            if row['ai_summary'] and str(row['ai_summary']).strip():
                keywords = parse_top_products(row.get('top_products', ''))
                keyword_count = sum(count for _, count in keywords[:3])
                
                restaurant_insights.append({
                    'text': row['ai_summary'],
                    'category': row['category'],
                    'keywords': keywords[:5],
                    'importance': keyword_count
                })
        
        # Sort by importance
        restaurant_insights.sort(key=lambda x: x['importance'], reverse=True)
        
        # Show top 3 insights
        st.markdown("**🌟 Top Insights**")
        for i, insight in enumerate(restaurant_insights[:3], 1):
            st.markdown(f"""
            <div class="theme-item">
                <strong>{i}. {insight['category']}</strong><br>
                {insight['text']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show keywords if available
            if insight['keywords']:
                keyword_text = ", ".join([f"{prod} ({count})" for prod, count in insight['keywords']])
                st.caption(f"🔑 {keyword_text}")
        
        st.markdown("---")
        
        # Show all categories in expandable sections
        st.markdown("**📁 All Categories**")
        for _, row in restaurant_data.iterrows():
            with st.expander(f"{row['category']}", expanded=False):
                if row['ai_summary'] and str(row['ai_summary']).strip():
                    st.write(row['ai_summary'])
                
                # Show keywords
                if 'top_products' in row and row['top_products']:
                    keywords = parse_top_products(row['top_products'])
                    if keywords:
                        st.markdown("**Keywords:**")
                        cols = st.columns(3)
                        for idx, (prod, count) in enumerate(keywords[:6]):
                            with cols[idx % 3]:
                                st.metric(prod, count, label_visibility="visible")

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
