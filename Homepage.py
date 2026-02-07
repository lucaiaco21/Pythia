import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Master Data Science Project",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS: Injects styles for fonts, spacing, and the "modern" look
st.markdown("""
    <style>
    /* Center the main title and change font */
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        color: #0E1117;
        text-align: center;
        font-weight: 700;
        margin-bottom: 10px;
    }
    /* Style for the description text */
    .description {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 1.2rem;
        color: #4F4F4F;
        text-align: center;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    /* Style for the group section */
    .group-header {
        color: #D32F2F; /* Accent color similar to UCM red */
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin-top: 20px;
    }
    .student-name {
        text-align: center;
        font-size: 1.0rem;
        color: #262730;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Logo Section: Using columns to emphasize UCM
# We create 3 columns: Left spacer, Center (Logos), Right spacer
# Inside the center, we create sub-columns for the two logos
col1, col2 = st.columns([1.5, 1])  # Ratio 1.5:1 gives UCM more space

with col1:
    # Use 'use_column_width' to make it responsive, or set explicit width
    # Assuming 'LOGO_UCM.jpg' is in the root directory
    try:
        image_ucm = Image.open('LOGO_UCM.png')
        st.image(image_ucm, width=280) # Adjust width as needed for emphasis
    except FileNotFoundError:
        st.error("UCM Logo not found. Please upload 'LOGO_UCM.png'")

with col2:
    # We add vertical padding or st.write("") if needed to align vertically
    st.write("") 
    st.write("") # Spacer to push logo down slightly if needed
    try:
        image_ntic = Image.open('LOGO_ntic.png')
        st.image(image_ntic, width=150) # Smaller width for NTIC
    except FileNotFoundError:
        st.error("NTIC Logo not found. Please upload 'LOGO_ntic.png'")

st.write("---") # A subtle divider line

# 4. Title and Description
st.markdown('<div class="main-title">Master data science, big data & business analytics 2024-2025</div>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
    Development of an intelligent application (APP) for a Coffee Shop analyzing its market 
    and predicting 14-day of revenues (€) and quantities for its 3 key categories of products 
    (Cafes Clasicos, Dulces & Reposteria, Desayunos & Tostadas).
</div>
""", unsafe_allow_html=True)

# 5. Group and Students Section

st.write("---") # A subtle divider line

# Using a container withOUT a border for a cleaner, modern look (Streamlit 1.30+)
with st.container(border=False):
    st.markdown('<div class="group-header">GRUPO 6</div>', unsafe_allow_html=True)
    
    # Grid layout for students (2 columns of 3 names) for better spacing
    s_col1, s_col2= st.columns(3)
    
    students = [
        "Anabel Jose Baéz Rodríguez",  "Ilan Alexander Arvelo Yagua",
        "Fatima Tawfik Vázquez", "Luca Iacomino", 
        "Genesis Karollay Hernandéz Gallegos", "Marcio Yassuhiro Iha"
    ]
    
    # Distribute students across two columns
    for i, student in enumerate(students):
        if i % 2 == 0:
            with s_col1:
                st.markdown(f'<div class="student-name">{student}</div>', unsafe_allow_html=True)
        else:
            with s_col2:
                st.markdown(f'<div class="student-name">{student}</div>', unsafe_allow_html=True)

#st.title("Main Page")
st.sidebar.success("Select one of the options")
