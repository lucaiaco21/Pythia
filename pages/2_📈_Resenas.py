
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Review Insights", page_icon="🤖", layout="wide")

# ==================== AI MODEL LOADING ====================

@st.cache_resource
def load_ai_model():
    """Load the Hugging Face review summarization model - FREE, no API key!"""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        model_name = "Manish014/review-summariser-gpt-config1"
        
        st.info("🤖 Loading AI model... (first time takes 1-2 minutes)")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available (faster)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return tokenizer, model, device, None
        
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

def chunk_reviews(reviews, chunk_size=3):
    """Split reviews into smaller chunks for better summaries"""
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i:i + chunk_size]

def summarize_reviews_ai(reviews, tokenizer, model, device):
    """
    Use AI to summarize ALL reviews together
    Returns a comprehensive summary of what people say
    """
    try:
        # Combine all reviews into chunks
        all_summaries = []
        
        for chunk in chunk_reviews(reviews, chunk_size=3):
            # Combine chunk of reviews
            combined = " ".join(chunk)
            
            # Limit length
            if len(combined) > 400:
                combined = combined[:400]
            
            # Prepare for model
            input_text = f"summarize: {combined}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate summary
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_summaries.append(summary)
        
        return all_summaries
        
    except Exception as e:
        return [f"Error: {str(e)}"]

def extract_key_themes(reviews):
    """Extract common themes/topics from reviews"""
    # Common keywords for cafeterias
    themes = {
        'Coffee': ['coffee', 'espresso', 'latte', 'cappuccino', 'brew', 'beans', 'americano'],
        'Service': ['service', 'staff', 'barista', 'friendly', 'helpful', 'quick', 'fast'],
        'Atmosphere': ['atmosphere', 'cozy', 'vibe', 'comfortable', 'relaxing', 'ambiance'],
        'Food': ['food', 'pastry', 'croissant', 'sandwich', 'cake', 'breakfast', 'lunch'],
        'Location': ['location', 'convenient', 'parking', 'easy to find', 'accessible'],
        'Price': ['price', 'value', 'worth', 'affordable', 'reasonable'],
        'Cleanliness': ['clean', 'tidy', 'spotless', 'neat'],
        'WiFi/Work': ['wifi', 'work', 'laptop', 'study', 'quiet', 'plugs', 'outlets']
    }
    
    theme_reviews = {}
    
    for theme_name, keywords in themes.items():
        relevant = []
        for review in reviews:
            if any(kw in str(review).lower() for kw in keywords):
                relevant.append(review)
        
        if len(relevant) >= 2:  # At least 2 reviews mention it
            theme_reviews[theme_name] = relevant
    
    return theme_reviews

# ==================== MAIN APP ====================

st.title("🤖 AI-Powered Competitor Review Insights")
st.markdown("""
**Analyze what customers LOVE about your competitors using FREE AI**  
No API keys needed! Uses Hugging Face's review summarization model.
""")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📋 How It Works")
    st.markdown("""
    This app uses **FREE AI** to analyze positive reviews!
    
    **Model:** review-summariser-gpt  
    **Source:** Hugging Face (open-source)  
    **Cost:** Completely FREE!
    
    ---
    
    **File format:**
    ```csv
    cafeteria,rating,review
    Starbucks,5,Great coffee!
    Local Cafe,4,Love the vibe
    ```
    
    **Tips:**
    - Focus on 4-5⭐ reviews only
    - 10-50 reviews per cafe works best
    - First load takes 1-2 min (downloads AI model)
    """)

# File upload
st.subheader("📁 Upload Competitor Reviews")
file = st.file_uploader("Upload CSV or Excel with positive reviews", type=['csv', 'xlsx'])

if file:
    try:
        # Load data
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.success(f"✅ {len(df)} reviews loaded")
        
        # Normalize columns
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns={
            'cafe': 'cafeteria',
            'shop': 'cafeteria',
            'review': 'comment',
            'text': 'comment'
        }, inplace=True)
        
        # Validate
        if not all(col in df.columns for col in ['cafeteria', 'rating', 'comment']):
            st.error(f"❌ Need columns: cafeteria, rating, comment. Found: {', '.join(df.columns)}")
            st.stop()
        
        # Preview
        with st.expander("👀 Data Preview"):
            st.dataframe(df.head())
        
        # Filter positive reviews only
        df_positive = df[df['rating'] >= 4].copy()
        st.info(f"📊 Analyzing {len(df_positive)} positive reviews (4-5 stars)")
        
        # Select cafeteria
        cafeterias = df_positive['cafeteria'].unique().tolist()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("Select competitor cafeteria:", cafeterias)
        with col2:
            compare_all = st.checkbox("Analyze all")
        
        st.divider()
        
        # Add "Analyze ALL" option
        analyze_all = st.checkbox("🌟 Analyze ALL Competitors Together", 
                                  help="Get insights across all cafeterias")
        
        if analyze_all:
            st.subheader("🌍 Combined Analysis: All Competitors")
            st.caption(f"Analyzing {len(df_positive)} positive reviews across {len(cafeterias)} cafeterias")
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Cafeterias", len(cafeterias))
            col2.metric("Total Positive Reviews", len(df_positive))
            col3.metric("Average Rating", f"{df_positive['rating'].mean():.2f}⭐")
            
            if st.button("🚀 Analyze All Competitors with AI", type="primary", use_container_width=True):
                # Load model
                tokenizer, model, device, error = load_ai_model()
                
                if error:
                    st.error(error)
                    st.stop()
                
                st.success("✅ AI model loaded!")
                
                # Get ALL positive reviews
                all_reviews = df_positive['comment'].tolist()
                
                # 1. OVERALL INSIGHTS
                st.markdown("---")
                st.subheader("📝 Market Insights: What Customers Love")
                
                with st.spinner("🤖 AI analyzing all competitor reviews..."):
                    summaries = summarize_reviews_ai(all_reviews, tokenizer, model, device)
                
                for i, summary in enumerate(summaries, 1):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white; margin: 10px 0;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4>💡 Insight {i}</h4>
                        <p style="font-size: 16px; margin: 0;">{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 2. CATEGORY-BASED INSIGHTS
                st.markdown("---")
                st.subheader("🎯 Insights by Category")
                
                theme_reviews = extract_key_themes(all_reviews)
                
                # Define categories with icons
                category_config = {
                    'Coffee': {'icon': '☕', 'title': 'Coffee Quality', 'color': '#8B4513'},
                    'Food': {'icon': '🍽️', 'title': 'Food Quality', 'color': '#FF6347'},
                    'Service': {'icon': '👥', 'title': 'Service & Staff', 'color': '#4169E1'},
                    'Atmosphere': {'icon': '🏮', 'title': 'Atmosphere', 'color': '#9370DB'},
                    'Location': {'icon': '📍', 'title': 'Location', 'color': '#32CD32'},
                    'Price': {'icon': '💰', 'title': 'Price & Value', 'color': '#FFD700'},
                    'Cleanliness': {'icon': '✨', 'title': 'Cleanliness', 'color': '#00CED1'},
                    'WiFi/Work': {'icon': '💻', 'title': 'Work Environment', 'color': '#FF69B4'}
                }
                
                # Create tabs for categories
                if theme_reviews:
                    category_tabs = st.tabs([f"{category_config.get(theme, {}).get('icon', '📌')} {category_config.get(theme, {}).get('title', theme)}" 
                                            for theme in theme_reviews.keys()])
                    
                    for tab, (theme, theme_revs) in zip(category_tabs, theme_reviews.items()):
                        with tab:
                            config = category_config.get(theme, {'icon': '📌', 'title': theme, 'color': '#666'})
                            
                            # Category header
                            st.markdown(f"## {config['icon']} {config['title']}")
                            
                            # Stats
                            col1, col2 = st.columns(2)
                            col1.metric("Mentions", f"{len(theme_revs)} reviews")
                            col2.metric("Frequency", f"{len(theme_revs)/len(all_reviews)*100:.1f}%")
                            
                            st.markdown("---")
                            
                            # AI Summary for this category
                            st.markdown("### 🤖 AI-Generated Insights")
                            
                            with st.spinner(f"Analyzing {config['title']}..."):
                                category_summaries = summarize_reviews_ai(theme_revs[:6], tokenizer, model, device)
                            
                            for summary in category_summaries:
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; 
                                            border-left: 4px solid {config['color']}; margin: 10px 0;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                    <p style="margin: 0; font-size: 15px; color: #333;">
                                        <strong>💡 {summary}</strong>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Action recommendations
                            st.markdown("### 🎯 Recommended Actions")
                            
                            # Generate action based on category
                            actions = {
                                'Coffee': [
                                    "🔹 Source high-quality beans to match competitor standards",
                                    "🔹 Train baristas on consistency and flavor profiles",
                                    "🔹 Offer signature blends that customers mention positively"
                                ],
                                'Food': [
                                    "🔹 Expand healthy food options based on customer demand",
                                    "🔹 Ensure freshness and quality consistency",
                                    "🔹 Highlight popular items mentioned in reviews"
                                ],
                                'Service': [
                                    "🔹 Train staff on friendliness and customer engagement",
                                    "🔹 Improve service speed during peak hours",
                                    "🔹 Implement feedback system for continuous improvement"
                                ],
                                'Atmosphere': [
                                    "🔹 Create comfortable seating arrangements",
                                    "🔹 Optimize lighting and ambiance",
                                    "🔹 Maintain cleanliness consistently"
                                ],
                                'Location': [
                                    "🔹 Ensure easy accessibility and clear signage",
                                    "🔹 Consider parking availability",
                                    "🔹 Leverage convenient location in marketing"
                                ],
                                'Price': [
                                    "🔹 Align pricing with perceived value",
                                    "🔹 Offer loyalty programs or promotions",
                                    "🔹 Communicate value proposition clearly"
                                ],
                                'Cleanliness': [
                                    "🔹 Implement regular cleaning schedules",
                                    "🔹 Maintain bathroom cleanliness",
                                    "🔹 Keep tables and floors spotless"
                                ],
                                'WiFi/Work': [
                                    "🔹 Provide fast, reliable WiFi",
                                    "🔹 Ensure adequate power outlets",
                                    "🔹 Create quiet work-friendly zones"
                                ]
                            }
                            
                            for action in actions.get(theme, ["🔹 Focus on customer feedback in this area"]):
                                st.markdown(action)
                            
                            # Example reviews
                            with st.expander("📋 Example Customer Comments"):
                                for i, rev in enumerate(theme_revs[:5], 1):
                                    st.markdown(f"{i}. *\"{rev[:200]}...\"*")
                
                # 3. TOP PRIORITIES SECTION
                st.markdown("---")
                st.subheader("🏆 Top Action Priorities")
                
                # Word frequency to identify top priorities
                all_text = " ".join(all_reviews).lower()
                words = all_text.split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                             'of', 'with', 'is', 'was', 'are', 'very', 'really', 'so', 'my', 'i', 'me', 'it', 'this', 'that'}
                words = [w for w in words if len(w) > 3 and w not in stop_words]
                
                from collections import Counter
                top_words = Counter(words).most_common(5)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 25px; border-radius: 10px; color: white; margin: 15px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="margin-top: 0;">📊 Market Priorities Based on Customer Feedback</h3>
                    <p style="font-size: 16px;">Focus your efforts on these areas to match competitor success:</p>
                </div>
                """, unsafe_allow_html=True)
                
                priority_num = 1
                for word, count in top_words:
                    # Map words to actionable insights
                    action_map = {
                        'coffee': 'Invest in premium coffee quality and barista training',
                        'food': 'Expand healthy and delicious food options',
                        'service': 'Enhance staff friendliness and service speed',
                        'great': 'Maintain consistency in what makes competitors great',
                        'good': 'Focus on overall quality across all touchpoints',
                        'place': 'Create an inviting and comfortable atmosphere',
                        'staff': 'Train staff for excellent customer service',
                        'nice': 'Pay attention to ambiance and cleanliness',
                        'friendly': 'Prioritize warm, welcoming customer interactions',
                        'delicious': 'Ensure food and drinks taste exceptional',
                        'love': 'Replicate the elements customers love most',
                        'best': 'Aim to be the best in key areas mentioned'
                    }
                    
                    action = action_map.get(word, f'Focus on improving "{word}" quality')
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 15px; border-radius: 8px; 
                                margin: 10px 0; border-left: 4px solid #f5576c;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="color: #f5576c; margin: 0 0 10px 0;">Priority #{priority_num}: {word.upper()}</h4>
                        <p style="margin: 0; color: #333; font-size: 15px;">
                            <strong>Action:</strong> {action}
                        </p>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 13px;">
                            Mentioned {count} times across all reviews ({count/len(all_reviews)*100:.1f}% frequency)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    priority_num += 1
                
                st.success("✨ Analysis complete! Use these insights to improve your cafeteria strategy.")
        
        elif not compare_all:
            # ===== SINGLE CAFETERIA ANALYSIS =====
            df_cafe = df_positive[df_positive['cafeteria'] == selected].copy()
            
            st.subheader(f"✨ What People Love About: {selected}")
            st.caption(f"Based on {len(df_cafe)} positive reviews")
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive Reviews", len(df_cafe))
            col2.metric("Average Rating", f"{df_cafe['rating'].mean():.2f}⭐")
            col3.metric("5-Star Reviews", len(df_cafe[df_cafe['rating'] == 5]))
            
            # AI ANALYSIS BUTTON
            if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                
                # Load model
                tokenizer, model, device, error = load_ai_model()
                
                if error:
                    st.error(error)
                    st.info("💡 Make sure 'transformers' and 'torch' are in requirements.txt")
                    st.stop()
                
                st.success("✅ AI model loaded successfully!")
                
                # Get reviews
                reviews = df_cafe['comment'].tolist()
                
                # 1. OVERALL AI SUMMARY
                st.markdown("---")
                st.subheader("📝 AI-Generated Summary")
                st.caption("What customers love most about this cafeteria")
                
                with st.spinner("🤖 AI is reading all positive reviews..."):
                    summaries = summarize_reviews_ai(reviews, tokenizer, model, device)
                
                for i, summary in enumerate(summaries, 1):
                    st.success(f"**Summary {i}:** {summary}")
                
                # 2. THEME-BASED ANALYSIS
                st.markdown("---")
                st.subheader("🔍 What People Mention Most")
                
                with st.spinner("Extracting key themes..."):
                    theme_reviews = extract_key_themes(reviews)
                
                if theme_reviews:
                    for theme, theme_revs in theme_reviews.items():
                        with st.expander(f"☕ {theme} - Mentioned in {len(theme_revs)} reviews"):
                            
                            # Show mention count
                            st.caption(f"📊 {len(theme_revs)}/{len(reviews)} reviews ({len(theme_revs)/len(reviews)*100:.0f}%) mention this")
                            
                            # AI summary for this theme
                            with st.spinner(f"AI analyzing {theme} reviews..."):
                                theme_summaries = summarize_reviews_ai(theme_revs[:6], tokenizer, model, device)
                            
                            st.markdown("**🤖 AI Summary:**")
                            for summary in theme_summaries:
                                st.info(summary)
                            
                            # Show example reviews
                            st.markdown("**📋 Example Reviews:**")
                            for rev in theme_revs[:3]:
                                st.markdown(f"• *\"{rev[:150]}...\"*")
                else:
                    st.warning("Not enough themed reviews found")
                
                # 3. OVERALL INSIGHTS
                st.markdown("---")
                st.subheader("💡 Key Takeaways")
                
                # Count most mentioned words
                all_text = " ".join(reviews).lower()
                words = all_text.split()
                # Remove common words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'very', 'really', 'so', 'my', 'i', 'me'}
                words = [w for w in words if len(w) > 3 and w not in stop_words]
                
                from collections import Counter
                top_words = Counter(words).most_common(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔤 Most Mentioned Words:**")
                    for word, count in top_words:
                        st.write(f"• **{word}**: {count} times")
                
                with col2:
                    st.markdown("**📊 Review Stats:**")
                    st.write(f"• Total positive reviews: {len(df_cafe)}")
                    st.write(f"• Average review length: {df_cafe['comment'].str.len().mean():.0f} characters")
                    st.write(f"• Themes identified: {len(theme_reviews)}")
            
            # Show all reviews
            st.markdown("---")
            st.subheader("📋 All Positive Reviews")
            st.dataframe(
                df_cafe[['rating', 'comment']].sort_values('rating', ascending=False),
                use_container_width=True
            )
        
        else:
            # ===== COMPARE ALL CAFETERIAS =====
            st.subheader("🔄 Quick Comparison")
            
            comp_data = []
            for cafe in cafeterias:
                df_c = df_positive[df_positive['cafeteria'] == cafe]
                comp_data.append({
                    'Cafeteria': cafe,
                    'Positive Reviews': len(df_c),
                    'Avg Rating': df_c['rating'].mean(),
                    '5-Star %': len(df_c[df_c['rating'] == 5]) / len(df_c) * 100
                })
            
            comp_df = pd.DataFrame(comp_data).sort_values('Avg Rating', ascending=False)
            
            st.dataframe(
                comp_df.style.background_gradient(subset=['Avg Rating'], cmap='Greens'),
                use_container_width=True
            )
            
            # Chart
            fig = px.bar(
                comp_df,
                x='Cafeteria',
                y='Positive Reviews',
                color='Avg Rating',
                title='Positive Reviews Comparison',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 Select individual cafeterias for detailed AI analysis")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload your competitor review data to start")
    
    # Example
    st.markdown("---")
    st.subheader("📝 Example File Format")
    
    example = pd.DataFrame({
        'cafeteria': ['Starbucks', 'Starbucks', 'Local Coffee Co', 'Blue Bottle'],
        'rating': [5, 4, 5, 5],
        'review': [
            'Great coffee and excellent service! The barista was super friendly and made my latte perfectly.',
            'Good atmosphere for working. WiFi is fast and plenty of seating. Coffee is consistently good.',
            'Best espresso in town! Love the cozy vibe and the staff really knows their coffee.',
            'Amazing pour-over coffee. You can really taste the quality of the beans. Worth the price!'
        ]
    })
    
    st.dataframe(example, use_container_width=True)
    
    st.markdown("""
    ## ✨ What This App Does
    
    ### 🎯 Focus on POSITIVE Reviews
    - Analyzes only 4-5⭐ reviews
    - Discovers what customers LOVE
    - Helps you understand competitor strengths
    
    ### 🤖 FREE AI Analysis
    - Uses Hugging Face's review-summariser model
    - No API keys or payments needed
    - Runs locally in your app
    
    ### 📊 What You Get
    1. **Overall AI Summary** - What people love most
    2. **Theme Analysis** - Coffee, Service, Atmosphere, etc.
    3. **AI Summaries per Theme** - Detailed insights
    4. **Example Reviews** - See actual customer comments
    5. **Word Frequency** - Most mentioned positive words
    
    ### 🚀 Perfect For
    - Understanding competitor strengths
    - Finding market opportunities
    - Improving your own cafeteria
    - Benchmarking against competition
    
    **Upload your data and discover what makes competitors successful!**
    """)
