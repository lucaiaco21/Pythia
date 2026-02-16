import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit.components.v1 as components
import pickle
import hashlib
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reviews analysis",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        color: #2c3e50; margin-bottom: 2rem;
    }
    .sub-header { font-size: 0.9rem; color: #6b7280; margin-bottom: 2rem; }
    .good-box {
        background-color: #d4edda; border-left: 5px solid #28a745;
        border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.4rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .bad-box {
        background-color: #f8d7da; border-left: 5px solid #dc3545;
        border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.4rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .box-label { font-weight: 600; font-size: 0.95rem; }
    .box-count { font-weight: 700; font-size: 0.9rem; white-space: nowrap; margin-left: 1rem; }
    .good-count { color: #155724; }
    .bad-count  { color: #721c24; }
    .cat-card {
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
        display: flex; align-items: center; gap: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .cat-card-green  { background: linear-gradient(135deg,#d4edda,#c3e6cb); border-left: 5px solid #28a745; }
    .cat-card-yellow { background: linear-gradient(135deg,#fff3cd,#ffeeba); border-left: 5px solid #ffc107; }
    .cat-card-red    { background: linear-gradient(135deg,#f8d7da,#f5c6cb); border-left: 5px solid #dc3545; }
    .cat-emoji  { font-size: 1.8rem; min-width: 2.2rem; text-align: center; }
    .cat-info   { flex: 1; }
    .cat-name   { font-weight: 700; font-size: 1rem; color: #2c3e50; }
    .cat-meta   { font-size: 0.8rem; color: #555; margin-top: 0.1rem; }
    .cat-score-wrap { text-align: right; min-width: 60px; }
    .cat-score  { font-size: 1.8rem; font-weight: 800; line-height: 1; }
    .cat-score-label { font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .score-green  { color: #1a7c3e; }
    .score-yellow { color: #856404; }
    .score-red    { color: #842029; }
    .progress-bar-bg {
        background: rgba(0,0,0,0.08); border-radius: 99px;
        height: 6px; margin-top: 0.4rem; overflow: hidden;
    }
    .progress-bar-fill { height: 6px; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST CSV LOADER
# Handles: comma / semicolon / tab separators, decimal-comma avg_rating values
# Just upload any CSV and the app will parse it correctly.
# ─────────────────────────────────────────────────────────────────────────────

def _fix_avg_rating(val):
    """
    Repair avg_rating values mangled by decimal-comma/dot confusion.
    e.g. 46.142 → 4.6142  |  4.296 → 4.296 (already fine)
    Also handles string formats like '4,296' → 4.296
    """
    try:
        s = str(val).strip().replace(',', '.')
        v = float(s)
    except (ValueError, TypeError):
        return None
    if 1.0 <= v <= 5.0:
        return round(v, 4)
    # Value > 5: decimal point was dropped after first digit
    # e.g. 46.142 → digits "46142" → "4.6142"
    digits = s.replace('.', '')
    try:
        fixed = float(digits[0] + '.' + digits[1:])
        return round(fixed, 4) if 1.0 <= fixed <= 5.0 else None
    except (ValueError, IndexError):
        return None

def load_csv(file_obj) -> pd.DataFrame:
    """
    Load a CSV from a file path or Streamlit UploadedFile.
    Auto-detects separator (, ; tab |) and repairs avg_rating if present.
    """
    import io
    # Read raw bytes so we can retry with different separators
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        file_obj.seek(0) if hasattr(file_obj, "seek") else None
    else:
        with open(file_obj, "rb") as f:
            raw = f.read()

    df = None
    for sep in [",", ";", "\t", "|"]:
        try:
            candidate = pd.read_csv(io.BytesIO(raw), sep=sep)
            if len(candidate.columns) > 1:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        df = pd.read_csv(io.BytesIO(raw))  # fallback

    # Repair avg_rating column if present
    if "avg_rating" in df.columns:
        fixed = df["avg_rating"].apply(_fix_avg_rating)
        median_val = fixed.dropna().median()
        df["avg_rating"] = fixed.fillna(median_val).round(4)

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD → CATEGORY LOGIC
# To update categories: edit CATEGORY_KEYWORDS below and re-upload the CSV.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "☕ Coffee":     ["coffee","cafe","cappuccino","espresso","latte","cortado",
                      "flat white","cold brew","americano","macchiato"],
    "🍽️ Food":       ["food","eat","meal","breakfast","lunch","dinner","toast",
                      "sandwich","cake","pastry","bread","croissant","avocado",
                      "egg","juice","menu","dish","plate","tapa","tortilla",
                      "salad","yogurt","delicious","fresh","cakes","foods"],
    "👥 Service":    ["staff","service","waiter","waitress","friendly","rude",
                      "attentive","helpful","slow","fast","quick","polite",
                      "customer","attend"],
    "🏠 Atmosphere": ["atmosphere","decor","cozy","ambiance","terrace","space",
                      "quiet","noise","music","interior","design","garden",
                      "place","location","seat","table","view"],
    "💶 Price":      ["price","expensive","cheap","value","worth","euro","cost",
                      "overpriced","pricey"],
    "⭐ Features":   ["wifi","parking","dog","pet","outdoor","indoor",
                      "accessible","toilet","bathroom","gluten free","vegan"],
}

# Single-word noise to filter out
NOISE_WORDS = {
    "good","nice","back","bit","tiny","super","highly","amazing","perfect",
    "lovely","loved","love","beautiful","wonderful","don","won","told","water",
    "table","serve","served","bar","people","girl","thing","tasted","seated",
    "leave","big","didn","time","menus","product","recommend","return",
    "excellent","delicious","terrible","pretty","poor","slice","understand",
    "minutes","rude","piece",
}

def assign_category(text: str) -> str:
    t = str(text).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keywords):
            return cat
    return "🔹 General"

def is_meaningful(text: str) -> bool:
    words = str(text).strip().split()
    if len(words) == 1 and str(text).lower() in NOISE_WORDS:
        return False
    if len(str(text).strip()) < 4:
        return False
    return True

def build_my_local_insights(df: pd.DataFrame):
    """
    From global_insights CSV -> auto-generate good & bad insight lists
    plus a categorized dataframe for the detail tab.

    Accepts two formats:
    - New (pre-categorized): columns insight, category, mentions, avg_rating, sentiment
    - Old (raw):             columns normalized_item, mentions, avg_rating, sentiment
    Just re-upload the CSV via the sidebar to refresh everything.
    """
    df = df.copy()

    # Normalise column name
    if "normalized_item" in df.columns and "insight" not in df.columns:
        df = df.rename(columns={"normalized_item": "insight"})

    # Use existing category column or auto-assign
    if "category" not in df.columns:
        df["category"] = df["insight"].apply(assign_category)

    # Map raw category names to emoji labels for display
    CAT_LABEL = {
        "atmosphere": "Atmosphere", "coffee": "Coffee",
        "food": "Food", "service": "Service",
        "location": "Location", "price": "Price",
        "features": "Features", "general": "General",
    }
    df["category"] = df["category"].map(
        lambda c: CAT_LABEL.get(str(c).strip().lower(), str(c).strip())
    )

    good = df[df["sentiment"].isin(["Muy Positivo", "Positivo"])].sort_values("mentions", ascending=False)
    bad  = df[df["sentiment"] == "Negativo"].sort_values("mentions", ascending=False)

    # Top 3 per category for good (pandas 3.0 compatible)
    good_top_idx = (
        good.groupby("category", group_keys=False)
        .apply(lambda g: g.head(3), include_groups=False)
        .sort_values("mentions", ascending=False)
        .head(15)
    )
    good_top = good.loc[good_top_idx.index]
    bad_top  = bad.head(12)

    def to_list(frame):
        return [
            {
                "insight":    row["insight"].title(),
                "mentions":   int(row["mentions"]),
                "category":   row["category"],
                "sentiment":  row["sentiment"],
                "avg_rating": round(float(row["avg_rating"]), 2),
            }
            for _, row in frame.iterrows()
        ]

    return to_list(good_top), to_list(bad_top), df

# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD & PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
PASSWORD = "PATIO"
DATA_DIR = Path("saved_data")
DATA_DIR.mkdir(exist_ok=True)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def save_data(data, password):
    pwd_hash = hash_password(password)
    with open(DATA_DIR / f"{pwd_hash}.pkl", "wb") as f:
        pickle.dump({**data, "timestamp": datetime.now()}, f)

def load_data(password):
    path = DATA_DIR / f"{hash_password(password)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def check_password():
    def _entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password to Access Detailed Analysis",
                      type="password", on_change=_entered, key="password")
        st.info(f"Password is {len(PASSWORD)} letters")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("Enter Password to Access Dashboard",
                      type="password", on_change=_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

if not check_password():
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
saved = load_data(PASSWORD)

if saved is not None:
    st.sidebar.success(f"✅ Data loaded (uploaded {saved['timestamp'].strftime('%Y-%m-%d %H:%M')})")
    if st.sidebar.button("🔄 Upload New Data"):
        st.session_state["force_upload"] = True
        st.rerun()

if saved is not None and "force_upload" not in st.session_state:
    per_rest_df = saved["per_restaurant"]
    common_df   = saved["common_insights"]
    category_df = saved["category_insights"]
    global_df   = saved.get("global_insights", pd.DataFrame())
    data_loaded = True
else:
    data_loaded = False

if not data_loaded:
    st.sidebar.info("Upload your insights files")

    up_per  = st.sidebar.file_uploader("📄 final_top_10_insights.csv",            type=["csv"], key="per_rest")
    up_com  = st.sidebar.file_uploader("📄 common_insights_all_restaurants.csv",   type=["csv"], key="common")
    up_cat  = st.sidebar.file_uploader("📄 top_insights_by_category.csv",          type=["csv"], key="category")
    up_glob = st.sidebar.file_uploader("📄 global_insights_v2.csv  *(My Local)*",  type=["csv"], key="global")

    if up_per and up_com and up_cat:
        try:
            per_rest_df = load_csv(up_per)
            common_df   = load_csv(up_com)
            category_df = load_csv(up_cat)
            global_df   = load_csv(up_glob) if up_glob else pd.DataFrame()

            save_data({
                "per_restaurant":    per_rest_df,
                "common_insights":   common_df,
                "category_insights": category_df,
                "global_insights":   global_df,
            }, PASSWORD)

            st.sidebar.success("✅ Saved! Won't need to re-upload next visit.")
            if "force_upload" in st.session_state:
                del st.session_state["force_upload"]
            data_loaded = True
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.stop()
    else:
        st.info("👈 Please upload the 3 required CSV files using the sidebar")
        st.markdown("""
        **Required (Competitor Analysis):**
        1. `final_top_10_insights.csv`
        2. `common_insights_all_restaurants.csv`
        3. `top_insights_by_category.csv`

        **Optional — enables My Local insight boxes & category detail:**
        4. `global_insights_v2.csv`
        *(columns: normalized_item, mentions, avg_rating, sentiment)*
        """)
        st.stop()

# Build My Local insights from CSV
if not global_df.empty:
    MY_LOCAL_GOOD, MY_LOCAL_BAD, global_df_cat = build_my_local_insights(global_df)
else:
    MY_LOCAL_GOOD, MY_LOCAL_BAD, global_df_cat = [], [], pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Menu")
page = st.sidebar.radio("Select Page",
    ["🏠 Patio Vertical", "🔍 Competitor Analysis"],
    label_visibility="collapsed")

st.markdown(
    '<div class="main-header">☕ Analysis of Online Reviews (Google Maps & Trip Advisor)</div>',
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PATIO VERTICAL
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Patio Vertical":
    st.markdown("---")
    st.markdown("## 📊 Review Distribution")
    star_data = {
        "Rating": ["⭐⭐⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐","⭐⭐","⭐"],
        "Count":  [1588, 497, 228, 84, 113],
        "Stars":  [5, 4, 3, 2, 1],
    }
    df_stars = pd.DataFrame(star_data).sort_values("Stars", ascending=False)
    total    = df_stars["Count"].sum()
    df_stars["Pct"] = (df_stars["Count"] / total * 100).round(1)
    cols = st.columns(5)
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: inherit !important; }
    [data-testid="stMetricDelta"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)
    for col, star in zip(cols, [5, 4, 3, 2, 1]):
        row = df_stars[df_stars["Stars"] == star]
        with col:
            st.metric("⭐" * star,
                      f"{row['Count'].values[0]:,}",
                      f"{row['Pct'].values[0]}%",
                      delta_color = "off")
    st.info(f"📊 **Total Reviews Analyzed:** {total:,} | Average Score: 4.7⭐")
    st.markdown("---")
    chart_sentiment_png = Path("images/GRAPHIC_monthly_sentiment_analysis.png")
    if chart_sentiment_png.exists():
        st.markdown("### 📈 Monthly Sentiment Trend") 
        with st.expander("ℹ️ How to read this chart"):
            st.markdown("""
            **Sentiment vs. Total Reviews** shows two things at once:
    
            - 🟢 **Green (Positive)**, 🟡 **Yellow (Neutral)**, 🔴 **Red (Negative)**  
            → The stacked bars show the **% breakdown** of review sentiment each month *(left axis)*
    
            - ⚫ **Black dashed line** → The **total number of reviews** received that month *(right axis)*
    
            **How to interpret it:**
            - A tall green bar = most reviews were positive that month
            - A spike in the line = unusually high review volume
            - Large red sections = months with more negative feedback to investigate
            """)
        st.image(str(chart_sentiment_png), use_container_width=True)

    st.markdown("## Analysis of Google Maps Reviews")
    st.caption("Text Analysis of your coffee shop review on Google Maps to extract insights about what customers think and say about you.
                Each score is calculated as a weighted average of star ratings, based on how often customers mention each aspect in their reviews.")
    st.info("""
**How it works — example:**
- 'Great price!' → mentioned in ⭐⭐⭐⭐⭐ review → pushes score **up** 
- 'Too expensive' → mentioned in ⭐ review → pushes score **down**
- Final score = weighted average of all those signals
""")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 General Insights", "🔍 View Insights by Category"])

    # ── TAB 1: GENERAL ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 📋 Executive Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Best Aspect")
            st.success("**Food**")
            st.metric("Score", "92/100", "Top rated", delta_color="off")
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] { color: green; }
                </style>
                """, unsafe_allow_html=True)
        with c2:
            st.markdown("### ⚠️ Needs Attention")
            st.warning("**PRICE**")
            st.metric("Score", "47/100", "⚠️ Lowest rated", delta_color="off")
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] { color: red; }
                </style>
                """, unsafe_allow_html=True)
        
        # ── CATEGORY SCORES ──────────────────────────────────────────────────────────
        st.markdown("---")
        with st.popover("ℹ️ About categories"):
            st.markdown("""
            **🔹 General**  
            Mentions that don't fit a specific category — overall impressions, 
            vague compliments/complaints, or multi-topic comments.  
            *Examples: "great experience", "wouldn't recommend", "came back again"*
    
             ---
    
            **⭐ Features**  
            Specific amenities and special attributes of the café.  
            *Examples: gluten-free options, WiFi, dog-friendly, outdoor terrace, 
            takeaway, vegan menu*
            """)
        st.markdown("### 🏆 Category Performance Scores")
        st.caption(
        "Score 0–100 combining **volume of mentions** (50%) and "
        "**average customer rating** (50%) per category. "
        "Higher = more talked about AND better rated."
        )

        CAT_EMOJI = {
        "food": "🍽️", "coffee": "☕", "service": "👥",
        "atmosphere": "🏠", "location": "📍", "price": "💶",
        "features": "⭐", "general": "🔹",
        }

        cat_summary = (
        global_df.groupby("category")
        .agg(
        total_mentions=("mentions", "sum"),
        avg_rating=("avg_rating", "mean"),
        score=("category_score", "first"),
        )
        .reset_index()
        .sort_values("score", ascending=False)
        )

        col_a, col_b = st.columns(2)
        for i, row in enumerate(cat_summary.itertuples()):
            score = int(row.score)
            if score >= 75:
                card_cls, score_cls, bar_color = "cat-card-green",  "score-green",  "#28a745"
            elif score >= 50:
                card_cls, score_cls, bar_color = "cat-card-yellow", "score-yellow", "#ffc107"
            else:
                card_cls, score_cls, bar_color = "cat-card-red",    "score-red",    "#dc3545"

            emoji    = CAT_EMOJI.get(str(row.category).lower(), "🔹")
            cat_name = str(row.category).title()
            mentions = int(row.total_mentions)
            rating   = round(float(row.avg_rating), 1)

            card_html = f"""
            <div class="cat-card {card_cls}">
            <div class="cat-emoji">{emoji}</div>
            <div class="cat-info">
            <div class="cat-name">{cat_name}</div>
            <div class="cat-meta">
            {mentions:,} mentions &nbsp;·&nbsp; {rating}⭐ avg rating
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill"
                     style="width:{score}%; background:{bar_color};"></div>
            </div>
            </div>
            <div class="cat-score-wrap">
            <div class="cat-score {score_cls}">{score}</div>
            <div class="cat-score-label">/ 100</div>
            </div>
        </div>"""

            with (col_a if i % 2 == 0 else col_b):
                st.markdown(card_html, unsafe_allow_html=True)

                st.caption(
                "🟢 Score ≥ 75 · Strong &nbsp;&nbsp; "
                "🟡 50–74 · Moderate &nbsp;&nbsp; "
                "🔴 < 50 · Needs attention"
                )

        st.markdown("---")
        cp, cs = st.columns(2)
        with cp:
            st.markdown("### 🔴 Top 3 Issues to Address")
            st.error("**1. Long Wait Times** — 359 mentions\n\n💡 Optimize service times")
            st.error("**2. Overpriced** — 273 mentions\n\n💡 Evaluate pricing strategy")
            st.error("**3. Rude Staff** — 64 mentions\n\n💡 Customer service training")
        with cs:
            st.markdown("### 🟢 Top 3 Strengths to Maintain")
            st.success("**1. Friendly Staff** — 990 mentions\n\nKeep it up!")
            st.success("**2. Fresh Food** — 893 mentions\n\nQuality recognized")
            st.success("**3. Cozy Atmosphere** — 418 mentions\n\nGreat ambiance")

        # ── GOOD / BAD BOXES ─────────────────────────────────────────────────
        if MY_LOCAL_GOOD or MY_LOCAL_BAD:
            st.markdown("---")
            st.markdown("## 💬 What Customers Say About Us")
            
            cg, cb = st.columns(2)
            with cg:
                st.markdown("### ✅ What We Do Well")
                for item in MY_LOCAL_GOOD:
                    st.markdown(
                        f'<div class="good-box">'
                        f'<span class="box-label">{item["category"]} - {item["insight"]}</span>'
                        f'<span class="box-count good-count">{item["mentions"]:,} mentions</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with cb:
                st.markdown("### ⚠️ What We Need to Improve")
                for item in MY_LOCAL_BAD:
                    st.markdown(
                        f'<div class="bad-box">'
                        f'<span class="box-label">{item["category"]} - {item["insight"]}</span>'
                        f'<span class="box-count bad-count">{item["mentions"]:,} mentions</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── TAB 2: DETAILED BY CATEGORY ──────────────────────────────────────────
    with tab2:
        st.markdown("## 🔍 Insights by Category")
        with st.expander("ℹ️ How to use this page"):
            st.markdown(
            """
### 🔍 Insights by Category — Guide

This page lets you explore **what customers mention most**, broken down by topic and sentiment.

---

#### 🎛️ Filters

| Filter | What it does |
|--------|-------------|
| **Filter by Category** | Select one or more topics to analyze (e.g. Atmosphere, Coffee, Food, Service...) |
| **Filter by Sentiment** | Show only insights tagged as Muy Positivo, Positivo, Neutro or Negativo |
| **Top N per category** | Controls how many insights are shown per category (slider, default = 8) |

> 💡 You can remove any tag by clicking the **×** next to it, or clear all with the **⊗** button.

---

#### 📊 Chart — Horizontal Bar
- Each bar = one **insight phrase** (e.g. "Atmosphere Cool", "Cozy Decor")
- Bar **length** = number of **mentions** across all reviews
- Bar **color** reflects sentiment:
    - 🟢 Dark green → Very Positive
    - 🟢 Light green → Positive
    - 🟡 Yellow → Neutral
    - 🔴 Red → Negative
- Bars are ranked **top to bottom** by mention count
            """
        )
        if global_df_cat.empty:
            st.warning(
                "Upload `global_insights_v2.csv` via the sidebar to see "
                "the full category breakdown."
            )
        else:
            # Filters row
            fc1, fc2, fc3 = st.columns([2, 2, 1])
            with fc1:
                all_cats  = sorted(global_df_cat["category"].unique())
                sel_cats  = st.multiselect("Filter by category",  all_cats,  default=all_cats)
            with fc2:
                all_sents = sorted(global_df_cat["sentiment"].unique())
                sel_sents = st.multiselect("Filter by sentiment", all_sents, default=all_sents)
            with fc3:
                top_n = st.slider("Top N per category", 3, 20, 8)

            filtered = global_df_cat[
                global_df_cat["category"].isin(sel_cats) &
                global_df_cat["sentiment"].isin(sel_sents)
            ]

            for cat in sel_cats:
                cat_data = (
                    filtered[filtered["category"] == cat]
                    .sort_values("mentions", ascending=False)
                    .head(top_n)
                )
                if cat_data.empty:
                    continue

                with st.expander(f"{cat}  —  {len(cat_data)} items shown", expanded=True):
                    # Bar chart coloured by sentiment
                    sent_color = {
                        "Muy Positivo": "#28a745",
                        "Positivo":     "#85c985",
                        "Neutro":       "#ffc107",
                        "Negativo":     "#dc3545",
                    }
                    fig_cat = go.Figure(go.Bar(
                        y=cat_data["insight"].str.title(),
                        x=cat_data["mentions"],
                        orientation="h",
                        marker=dict(color=[
                            sent_color.get(s, "#aaa") for s in cat_data["sentiment"]
                        ]),
                        text=cat_data["mentions"],
                        textposition="outside",
                    ))
                    fig_cat.update_layout(
                        height=max(250, len(cat_data) * 38),
                        margin=dict(l=10, r=40, t=10, b=10),
                        showlegend=False,
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title="Mentions",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

                    # Searchable table
                    display = cat_data[
                        ["insight","mentions","avg_rating","sentiment"]
                    ].copy()
                    display.columns = ["Insight","Mentions","Avg Rating","Sentiment"]
                    display["Insight"] = display["Insight"].str.title()
                    st.dataframe(
                        display.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#7f8c8d;padding:2rem;'>"
        "📊 Analysis based on 2,510 reviews | 🔄 February 2026</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPETITOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Competitor Analysis":

    st.markdown(
        '<div class="sub-header">Analysis of reviews of main competitors. '
        "Maps show most-rated cafes in Madrid; in-depth analysis covers "
        "those within 3 km of Cafe Madrid.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## 📊 Market Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Cafes", len(per_rest_df["restaurant"].unique()))
    with c2:
        st.metric("Unique Insights", len(common_df))
    with c3:
        top = common_df.iloc[0]
        st.metric("Most Mentioned", top["insight"].title(), f"{top['mentions']} times")
    with c4:
        mc = common_df.nlargest(1, "num_restaurants").iloc[0]
        st.metric("Most Common", mc["insight"].title(),
                  f"{mc['num_restaurants']}/{len(per_rest_df['restaurant'].unique())} shops")

    st.markdown("---")
    map_path = Path("images/mapa_cafeterias_seeccionadas_madrid_20260210.html")
    if map_path.exists():
        st.markdown("## 📍 Map — Average Scores of Most-Rated Madrid Cafeterias")
        with open(map_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=400, scrolling=False)
        st.markdown("---")

    category_colors = {
        "food": "#FF6B6B", "coffee": "#8B4513", "specialty_drink": "#4ECDC4",
        "service": "#FFE66D", "feature": "#A8E6CF", "atmosphere": "#C7CEEA",
    }
    category_labels = {
        "food": "🍽️ Food", "coffee": "☕ Coffee",
        "specialty_drink": "🍵 Specialty Drinks", "service": "👥 Service",
        "feature": "⭐ Features", "atmosphere": "🏠 Atmosphere",
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Insights", "🏪 By Cafes", "📊 Category Analysis", "🔍 Comparison"
    ])

    # ── TAB 1 ────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🏆 What Customers Love Most")
        top_20 = common_df.head(20).copy()
        fig = go.Figure(go.Bar(
            y=top_20["insight"], x=top_20["mentions"], orientation="h",
            marker=dict(color=[category_colors.get(c, "#95A5A6") for c in top_20["category"]]),
            text=top_20["mentions"], textposition="outside",
        ))
        fig.update_layout(title="Top 20 Most Mentioned Items",
                          xaxis_title="Total Mentions", height=700,
                          showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        wide = common_df.nlargest(10, "num_restaurants")
        fig2 = go.Figure(go.Bar(
            x=wide["insight"], y=wide["num_restaurants"],
            marker=dict(color="#3498db"),
            text=wide["num_restaurants"], textposition="outside",
        ))
        fig2.update_layout(title="Insights Present Across Multiple Cafes",
                           yaxis_title="Number of Restaurants",
                           height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2 ────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🏪 Cafe-Specific Insights")
        restaurants = sorted(per_rest_df["restaurant"].unique())
        sel = st.selectbox("Select a restaurant:", restaurants)
        rd  = per_rest_df[per_rest_df["restaurant"] == sel].sort_values("mentions", ascending=False)

        st.markdown(f"## ☕ {sel}")
        rc1, rc2, rc3 = st.columns(3)
        with rc1: st.metric("Top Insight",   rd.iloc[0]["insight"].title())
        with rc2: st.metric("Most Mentions", rd.iloc[0]["mentions"])
        with rc3:
            top_cat = rd["category"].value_counts().index[0]
            st.metric("Main Category", top_cat.replace("_", " ").title())

        st.markdown("---")
        for cat in ["food","coffee","specialty_drink","service"]:
            items = rd[rd["category"] == cat]
            if not items.empty:
                with st.expander(f"{category_labels.get(cat)} ({len(items)} items)", expanded=True):
                    for _, row in items.iterrows():
                        cc1, cc2 = st.columns([3, 1])
                        with cc1: st.markdown(f"**{row['insight'].title()}**")
                        with cc2: st.markdown(f"*{row['mentions']} mentions*")

    # ── TAB 3 ────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Category Deep Dive")
        cat_totals = per_rest_df.groupby("category")["mentions"].sum().reset_index()
        fig_pie = px.pie(cat_totals, values="mentions", names="category",
                         title="Total Mentions by Category",
                         color="category", color_discrete_map=category_colors)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 4: COMPARISON ────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 Cafe Comparison")
        st.caption(
            "ℹ️ Competitor data contains **positive reviews only**. "
            "My Local shows both positive and negative feedback."
        )

        restaurants    = sorted(per_rest_df["restaurant"].unique())
        MY_LOCAL_LABEL = "⭐ My Local (Patio Vertical)"

        # Only add My Local option if we actually have insight data for it
        has_my_local = bool(MY_LOCAL_GOOD)
        all_options  = ([MY_LOCAL_LABEL] if has_my_local else []) + restaurants

        # My Local series — positive mentions only (fair comparison with competitors)
        if has_my_local:
            my_local_comp = pd.DataFrame([
                {"insight": i["insight"].lower(), "mentions": i["mentions"]}
                for i in MY_LOCAL_GOOD
            ])
        else:
            my_local_comp = pd.DataFrame(columns=["insight", "mentions"])
            st.info("Upload `global_insights_v2.csv` via the sidebar to enable "
                    "My Local in the comparison.")

        col1, col2 = st.columns(2)
        with col1:
            rest1 = st.selectbox("First cafe:", all_options, key="r1")
        with col2:
            rest2 = st.selectbox("Second cafe:",
                                 [r for r in all_options if r != rest1], key="r2")

        def get_series(name):
            if name == MY_LOCAL_LABEL:
                if my_local_comp.empty:
                    return pd.Series(dtype=float)
                return my_local_comp.set_index("insight")["mentions"]
            return per_rest_df[per_rest_df["restaurant"] == name].set_index("insight")["mentions"]

        s1, s2   = get_series(rest1), get_series(rest2)
        all_ins  = list(set(s1.index) | set(s2.index))
        comp     = pd.DataFrame(
            {rest1: [s1.get(i, 0) for i in all_ins],
             rest2: [s2.get(i, 0) for i in all_ins]},
            index=all_ins,
        )
        comp = comp[(comp[rest1] > 0) | (comp[rest2] > 0)]
        comp = comp.sort_values(rest1, ascending=False).head(15)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name=rest1, x=comp.index, y=comp[rest1],
                                  marker_color="#3498db"))
        fig_comp.add_trace(go.Bar(name=rest2, x=comp.index, y=comp[rest2],
                                  marker_color="#e74c3c"))
        fig_comp.update_layout(title=f"{rest1} vs {rest2}",
                               yaxis_title="Mentions", barmode="group",
                               height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Full good/bad breakdown when My Local is in the comparison
        if MY_LOCAL_LABEL in (rest1, rest2) and (MY_LOCAL_GOOD or MY_LOCAL_BAD):
            st.markdown("---")
            st.markdown("#### 💬 My Local — Full Review Breakdown")
            st.caption(
                "The chart above uses positive mentions only for a fair comparison. "
                "Here's the complete picture for Patio Vertical:"
            )
            bg, bb = st.columns(2)
            with bg:
                st.markdown("**✅ What We Do Well**")
                for item in MY_LOCAL_GOOD:
                    st.markdown(
                        f'<div class="good-box">'
                        f'<span class="box-label">{item["category"]} &nbsp; {item["insight"]}</span>'
                        f'<span class="box-count good-count">{item["mentions"]:,}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with bb:
                st.markdown("**⚠️ What We Need to Improve**")
                for item in MY_LOCAL_BAD:
                    st.markdown(
                        f'<div class="bad-box">'
                        f'<span class="box-label">{item["category"]} &nbsp; {item["insight"]}</span>'
                        f'<span class="box-count bad-count">{item["mentions"]:,}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#7f8c8d;padding:2rem;'>"
        "📊 Based on customer review insights | 🔄 February 2026</div>",
        unsafe_allow_html=True,
    )
