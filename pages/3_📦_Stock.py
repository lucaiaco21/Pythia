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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('competitor_insights.csv')
    return df

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
    # Split by comma and parse each product
    for item in top_products_str.split('),'):
        item = item.strip()
        # Extract product name and count
        match = re.match(r'(.+?)\s*\((\d+)\)', item.replace(')', ''))
        if match:
            product_name = match.group(1).strip()
            count = int(match.group(2))
            # Filter out stopwords
            if product_name.lower() not in STOPWORDS:
                products.append((product_name, count))
    
    return products

def get_filtered_keywords(df_subset):
    """Get top keywords from a subset of data, filtered for meaningful words"""
    all_products = []
    for products_str in df_subset['top_products']:
        parsed = parse_top_products(products_str)
        all_products.extend(parsed)
    
    # Aggregate counts
    product_counter = Counter()
    for product, count in all_products:
        product_counter[product] += count
    
    return product_counter.most_common(15)

# Load data
df = load_data()

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Restaurants", df['restaurant'].nunique())
    with col2:
        st.metric("Total Categories", df['category'].nunique())
    with col3:
        st.metric("Total Mentions", df['total_mentions'].sum())
    
    st.markdown("---")
    
    # Restaurant comparison
    st.subheader("📈 Restaurant Performance Comparison")
    
    restaurant_stats = df.groupby('restaurant').agg({
        'total_mentions': 'sum',
        'positive_mentions': 'sum'
    }).reset_index()
    restaurant_stats['positive_rate'] = (
        restaurant_stats['positive_mentions'] / restaurant_stats['total_mentions'] * 100
    ).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mentions = px.bar(
            restaurant_stats.sort_values('total_mentions', ascending=True),
            x='total_mentions',
            y='restaurant',
            orientation='h',
            title='Total Mentions by Restaurant',
            labels={'total_mentions': 'Total Mentions', 'restaurant': 'Restaurant'},
            color='total_mentions',
            color_continuous_scale='Blues'
        )
        fig_mentions.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mentions, use_container_width=True)
    
    with col2:
        fig_positive = px.bar(
            restaurant_stats.sort_values('positive_rate', ascending=True),
            x='positive_rate',
            y='restaurant',
            orientation='h',
            title='Positive Mention Rate by Restaurant (%)',
            labels={'positive_rate': 'Positive Rate (%)', 'restaurant': 'Restaurant'},
            color='positive_rate',
            color_continuous_scale='Greens'
        )
        fig_positive.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_positive, use_container_width=True)
    
    st.markdown("---")
    
    # AI Summaries for all restaurants
    st.subheader("🤖 AI Insights Summary")
    
    for restaurant in df['restaurant'].unique():
        with st.expander(f"🏪 {restaurant}"):
            restaurant_data = df[df['restaurant'] == restaurant]
            
            # Show key categories
            st.markdown("**Top Categories:**")
            top_cats = restaurant_data.nlargest(3, 'total_mentions')[['category', 'total_mentions', 'positive_mentions']]
            for _, row in top_cats.iterrows():
                st.markdown(f"- **{row['category']}**: {row['total_mentions']} mentions ({row['positive_mentions']} positive)")
            
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Restaurants in Category", category_df['restaurant'].nunique())
    with col2:
        st.metric("Total Mentions", category_df['total_mentions'].sum())
    with col3:
        avg_positive = (category_df['positive_mentions'].sum() / category_df['total_mentions'].sum() * 100)
        st.metric("Avg Positive Rate", f"{avg_positive:.1f}%")
    
    st.markdown("---")
    
    # Top keywords for this category
    st.subheader("🔑 Top Keywords in This Category")
    keywords = get_filtered_keywords(category_df)
    
    if keywords:
        # Create visualization
        keywords_df = pd.DataFrame(keywords, columns=['Product', 'Count'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Keywords:**")
            for i, (product, count) in enumerate(keywords[:10], 1):
                st.markdown(f"{i}. **{product}** - {count} mentions")
    else:
        st.info("No meaningful keywords found for this category.")
    
    st.markdown("---")
    
    # Restaurant performance in this category
    st.subheader("🏪 Restaurant Performance")
    
    restaurant_perf = category_df.sort_values('total_mentions', ascending=False)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Bar(
        name='Positive Mentions',
        y=restaurant_perf['restaurant'],
        x=restaurant_perf['positive_mentions'],
        orientation='h',
        marker=dict(color='green', opacity=0.7)
    ))
    fig_perf.add_trace(go.Bar(
        name='Negative Mentions',
        y=restaurant_perf['restaurant'],
        x=restaurant_perf['total_mentions'] - restaurant_perf['positive_mentions'],
        orientation='h',
        marker=dict(color='red', opacity=0.7)
    ))
    
    fig_perf.update_layout(
        barmode='stack',
        title=f'Mention Breakdown by Restaurant - {selected_category}',
        xaxis_title='Number of Mentions',
        yaxis_title='Restaurant',
        height=400
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.markdown("---")
    
    # AI Summaries for each restaurant in this category
    st.subheader("🤖 AI Insights by Restaurant")
    
    for _, row in category_df.iterrows():
        with st.expander(f"🏪 {row['restaurant']} - {row['total_mentions']} mentions ({row['positive_mentions']} positive)"):
            if row['ai_summary'] and str(row['ai_summary']).strip():
                st.info(row['ai_summary'])
            else:
                st.warning("No AI summary available for this entry.")
            
            # Show top products
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Categories", len(restaurant_df))
    with col2:
        st.metric("Total Mentions", restaurant_df['total_mentions'].sum())
    with col3:
        st.metric("Positive Mentions", restaurant_df['positive_mentions'].sum())
    with col4:
        positive_rate = (restaurant_df['positive_mentions'].sum() / restaurant_df['total_mentions'].sum() * 100)
        st.metric("Positive Rate", f"{positive_rate:.1f}%")
    
    st.markdown("---")
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Categories by Mentions")
        fig_cat = px.pie(
            restaurant_df,
            values='total_mentions',
            names='category',
            title='Distribution of Mentions by Category'
        )
        fig_cat.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("📈 Top Categories")
        top_categories = restaurant_df.nlargest(5, 'total_mentions')[['category', 'total_mentions', 'positive_mentions']]
        
        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            name='Positive',
            x=top_categories['category'],
            y=top_categories['positive_mentions'],
            marker=dict(color='green', opacity=0.7)
        ))
        fig_top.add_trace(go.Bar(
            name='Negative',
            x=top_categories['category'],
            y=top_categories['total_mentions'] - top_categories['positive_mentions'],
            marker=dict(color='red', opacity=0.7)
        ))
        
        fig_top.update_layout(
            barmode='stack',
            title='Top 5 Categories',
            xaxis_title='Category',
            yaxis_title='Mentions',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    st.markdown("---")
    
    # Top keywords for this restaurant
    st.subheader("🔑 Most Mentioned Products")
    keywords = get_filtered_keywords(restaurant_df)
    
    if keywords:
        keywords_df = pd.DataFrame(keywords[:12], columns=['Product', 'Count'])
        
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
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    st.markdown("---")
    
    # AI Insights by category
    st.subheader("🤖 AI Insights by Category")
    
    for _, row in restaurant_df.sort_values('total_mentions', ascending=False).iterrows():
        with st.expander(f"📁 {row['category']} ({row['total_mentions']} mentions, {row['positive_mentions']} positive)"):
            if row['ai_summary'] and str(row['ai_summary']).strip():
                st.info(row['ai_summary'])
            else:
                st.warning("No AI summary available.")
            
            # Show top products for this category
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
