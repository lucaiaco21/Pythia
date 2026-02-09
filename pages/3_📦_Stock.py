import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="Competitor Insights Dashboard",
    page_icon="☕",
    layout="wide"
)

# File upload section
st.sidebar.title("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your competitor insights CSV", 
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


# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview - All Restaurants", "🏷️ Browse by Category", "🏪 Individual Restaurant"]
)

# Main content
if page == "📊 Overview - All Restaurants":
    st.title("☕ Competitor Insights Dashboard")
    st.markdown("### Overview of All Restaurants")
    
    # Key metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Restaurants", df['restaurant'].nunique())
    with col2:
        st.metric("Total Categories", df['category'].nunique())
    
    st.markdown("---")
    
    # Restaurant comparison - number of categories per restaurant
    st.subheader("📈 Categories Coverage by Restaurant")
    
    restaurant_stats = df.groupby('restaurant').agg({
        'category': 'count'
    }).reset_index()
    restaurant_stats.columns = ['restaurant', 'num_categories']
    
    fig_categories = px.bar(
        restaurant_stats.sort_values('num_categories', ascending=True),
        x='num_categories',
        y='restaurant',
        orientation='h',
        title='Number of Categories Analyzed per Restaurant',
        labels={'num_categories': 'Number of Categories', 'restaurant': 'Restaurant'},
        color='num_categories',
        color_continuous_scale='Blues'
    )
    fig_categories.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_categories, width='stretch')
    
    st.markdown("---")
    
    # AI Summaries for all restaurants
    st.subheader("🤖 AI Insights Summary")
    
    for restaurant in df['restaurant'].unique():
        with st.expander(f"🏪 {restaurant}"):
            restaurant_data = df[df['restaurant'] == restaurant]
            
            # Show key categories
            st.markdown("**Categories Covered:**")
            for _, row in restaurant_data.iterrows():
                st.markdown(f"- {row['category']}")
            
            st.markdown("---")
            
            # Show AI summaries
            st.markdown("**Key Insights:**")
            for _, row in restaurant_data.iterrows():
                if row['ai_summary'] and str(row['ai_summary']).strip():
                    st.markdown(f"**{row['category']}**")
                    st.info(row['ai_summary'])


elif page == "🏷️ Browse by Category":
    st.title("🏷️ Browse by Category")
    
    # Category selector
    categories = sorted(df['category'].unique())
    selected_category = st.selectbox("Select a category to explore:", categories)
    
    # Filter data
    category_df = df[df['category'] == selected_category]
    
    st.markdown(f"## {selected_category.title()}")
    
    # Category overview metrics
    st.metric("Restaurants in Category", category_df['restaurant'].nunique())
    
    st.markdown("---")
    
    # Top keywords for this category (if available)
    if 'top_products' in category_df.columns:
        st.subheader("🔑 Top Keywords in This Category")
        
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
        
        all_products = []
        for products_str in category_df['top_products']:
            all_products.extend(parse_top_products(products_str))
        
        product_counter = Counter()
        for product, count in all_products:
            product_counter[product] += count
        
        keywords = product_counter.most_common(15)
        
        if keywords:
            keywords_df = pd.DataFrame(keywords, columns=['Product', 'Count'])
            
            fig_keywords = px.bar(
                keywords_df,
                x='Count',
                y='Product',
                orientation='h',
                title=f'Most Mentioned Products in {selected_category}',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_keywords.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_keywords, width='stretch')
        else:
            st.info("No meaningful keywords found for this category.")
    
    st.markdown("---")
    
    # AI Summaries for each restaurant in this category
    st.subheader("🤖 AI Insights by Restaurant")
    
    for _, row in category_df.iterrows():
        with st.expander(f"🏪 {row['restaurant']}"):
            if row['ai_summary'] and str(row['ai_summary']).strip():
                st.info(row['ai_summary'])
            else:
                st.warning("No AI summary available for this entry.")
            
            # Show top products if available
            if 'top_products' in row and row['top_products']:
                STOPWORDS = {
                    'you', 'which', 'they', 'that', 'this', 'these', 'those', 'clasico',
                    'everything', 'food', 'place', 'staff', 'madrid', 'coffee', 'also',
                    'very', 'great', 'good', 'best', 'really', 'super', 'nice', 'lovely',
                    'amazing', 'perfect', 'excellent', 'wonderful', 'fantastic'
                }
                
                def parse_top_products(top_products_str):
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
                
                products = parse_top_products(row['top_products'])
                if products:
                    st.markdown("**Top Products Mentioned:**")
                    for product, count in products[:5]:
                        st.markdown(f"- {product} ({count})")



elif page == "🏪 Individual Restaurant":
    st.title("🏪 Individual Restaurant Analysis")
    
    # Restaurant selector
    restaurants = sorted(df['restaurant'].unique())
    selected_restaurant = st.selectbox("Select a restaurant:", restaurants)
    
    # Filter data
    restaurant_df = df[df['restaurant'] == selected_restaurant]
    
    st.markdown(f"## {selected_restaurant}")
    
    # Overview metrics
    st.metric("Total Categories Analyzed", len(restaurant_df))
    
    st.markdown("---")
    
    # Category breakdown
    st.subheader("📊 Categories Distribution")
    
    fig_cat = px.pie(
        restaurant_df,
        names='category',
        title='Categories Covered',
        hole=0.3
    )
    fig_cat.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_cat, width='stretch')
    
    st.markdown("---")
    
    # Top keywords for this restaurant (if available)
    if 'top_products' in restaurant_df.columns:
        st.subheader("🔑 Most Mentioned Products")
        
        # Filter out meaningless keywords
        STOPWORDS = {
            'you', 'which', 'they', 'that', 'this', 'these', 'those', 'clasico',
            'everything', 'food', 'place', 'staff', 'madrid', 'coffee', 'also',
            'very', 'great', 'good', 'best', 'really', 'super', 'nice', 'lovely',
            'amazing', 'perfect', 'excellent', 'wonderful', 'fantastic'
        }
        
        def parse_top_products(top_products_str):
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
        
        all_products = []
        for products_str in restaurant_df['top_products']:
            all_products.extend(parse_top_products(products_str))
        
        product_counter = Counter()
        for product, count in all_products:
            product_counter[product] += count
        
        keywords = product_counter.most_common(12)
        
        if keywords:
            keywords_df = pd.DataFrame(keywords, columns=['Product', 'Count'])
            
            fig_keywords = px.bar(
                keywords_df,
                x='Product',
                y='Count',
                title=f'Top Products at {selected_restaurant}',
                color='Count',
                color_continuous_scale='Plasma'
            )
            fig_keywords.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig_keywords, width='stretch')
    
    st.markdown("---")
    
    # AI Insights by category
    st.subheader("🤖 AI Insights by Category")
    
    for _, row in restaurant_df.iterrows():
        with st.expander(f"📁 {row['category']}"):
            if row['ai_summary'] and str(row['ai_summary']).strip():
                st.info(row['ai_summary'])
            else:
                st.warning("No AI summary available.")
            
            # Show top products for this category (if available)
            if 'top_products' in row and row['top_products']:
                STOPWORDS = {
                    'you', 'which', 'they', 'that', 'this', 'these', 'those', 'clasico',
                    'everything', 'food', 'place', 'staff', 'madrid', 'coffee', 'also',
                    'very', 'great', 'good', 'best', 'really', 'super', 'nice', 'lovely',
                    'amazing', 'perfect', 'excellent', 'wonderful', 'fantastic'
                }
                
                def parse_top_products(top_products_str):
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
                
                products = parse_top_products(row['top_products'])
                if products:
                    st.markdown("**Top Products in This Category:**")
                    cols = st.columns(3)
                    for idx, (product, count) in enumerate(products[:6]):
                        with cols[idx % 3]:
                            st.metric(product, f"{count} mentions")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Competitor Insights Dashboard | Data Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
