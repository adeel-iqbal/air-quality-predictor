import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("dt_pipeline.joblib")
    except FileNotFoundError:
        st.error("⚠️ Model file 'dt_pipeline.joblib' not found. Please ensure the model file is in the correct directory.")
        return None
# ------------------------
# App Configuration
# ------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Enhanced CSS with intuitive colors and clean layout
st.markdown("""
<style>
    /* Main styling */
    .main-title {
        text-align: center;
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Prediction box styling with category-specific colors */
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-good {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
    }
    
    .prediction-moderate {
        background: linear-gradient(135deg, #eab308, #ca8a04);
        color: white;
    }
    
    .prediction-unhealthy-sensitive {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
    }
    
    .prediction-unhealthy {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    .prediction-very-unhealthy {
        background: linear-gradient(135deg, #a855f7, #9333ea);
        color: white;
    }
    
    .prediction-hazardous {
        background: linear-gradient(135deg, #991b1b, #7f1d1d);
        color: white;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .prediction-desc {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    .prediction-bar {
        width: 100%;
        height: 8px;
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    .prediction-bar-fill {
        height: 8px;
        border-radius: 4px;
        background-color: rgba(255, 255, 255, 0.7);
    }
    
    /* Right sidebar styling */
    .right-sidebar {
        background-color: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        height: fit-content;
        position: sticky;
        top: 1rem;
    }
    
    .right-sidebar h3 {
        color: #1e40af;
        margin-top: 0;
        border-bottom: 2px solid #bfdbfe;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .info-box:hover {
        transform: translateY(-2px);
    }
    
    .info-box h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #1e40af;
    }
    
    .info-box p {
        margin: 0;
        color: #4b5563;
    }
    
    /* AQI Categories Reference with intuitive colors */
    .category-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        height: 140px; /* Fixed height for equal boxes */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .category-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .category-good {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 4px solid #22c55e;
    }
    
    .category-moderate {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 4px solid #eab308;
    }
    
    .category-unhealthy-sensitive {
        background: linear-gradient(135deg, #fed7aa, #fdba74);
        border-left: 4px solid #f97316;
    }
    
    .category-unhealthy {
        background: linear-gradient(135deg, #fecaca, #fca5a5);
        border-left: 4px solid #ef4444;
    }
    
    .category-very-unhealthy {
        background: linear-gradient(135deg, #e9d5ff, #d8b4fe);
        border-left: 4px solid #a855f7;
    }
    
    .category-hazardous {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 4px solid #991b1b;
    }
    
    .category-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .category-good .category-title { color: #16a34a; }
    .category-moderate .category-title { color: #ca8a04; }
    .category-unhealthy-sensitive .category-title { color: #ea580c; }
    .category-unhealthy .category-title { color: #dc2626; }
    .category-very-unhealthy .category-title { color: #9333ea; }
    .category-hazardous .category-title { color: #991b1b; }
    
    .category-desc {
        font-size: 0.9rem;
        color: #4b5563;
    }
    
    /* Health recommendations */
    .health-rec {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Hide Streamlit default styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Responsive adjustments */
    @media (max-width: 992px) {
        .right-sidebar {
            position: relative;
            top: 0;
            margin-top: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)
# Load the model
dt_pipeline = load_model()
# ------------------------
# Constants with Intuitive Colors
# ------------------------
CATEGORY_COLORS = {
    "Good": "#22c55e",  # Green
    "Moderate": "#eab308",  # Yellow
    "Unhealthy for Sensitive Groups": "#f97316",  # Orange
    "Unhealthy": "#ef4444",  # Red
    "Very Unhealthy": "#a855f7",  # Purple
    "Hazardous": "#991b1b"  # Dark Red
}
CATEGORY_CLASSES = {
    "Good": "prediction-good",
    "Moderate": "prediction-moderate",
    "Unhealthy for Sensitive Groups": "prediction-unhealthy-sensitive",
    "Unhealthy": "prediction-unhealthy",
    "Very Unhealthy": "prediction-very-unhealthy",
    "Hazardous": "prediction-hazardous"
}
CATEGORY_DESCRIPTIONS = {
    "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable. However, sensitive individuals may experience effects.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. General public less likely affected.",
    "Unhealthy": "Some members of the general public may experience health effects; sensitive groups may experience more serious effects.",
    "Very Unhealthy": "Health alert: Risk of health effects is increased for everyone.",
    "Hazardous": "Health warning of emergency conditions: everyone is more likely to be affected."
}
# Complete list of countries
country_options = [
    'Russian Federation', 'Brazil', 'Italy', 'Poland', 'France',
    'United States of America', 'Germany', 'Belgium', 'Egypt', 'China',
    'Netherlands', 'India', 'Pakistan', 'Republic of North Macedonia', 'Colombia',
    'Romania', 'Indonesia', 'Finland', 'South Africa',
    'United Kingdom of Great Britain and Northern Ireland',
    'United Republic of Tanzania', 'Haiti', 'Somalia', 'Philippines', 'Latvia',
    'Chad', 'New Zealand', 'Tunisia', 'Viet Nam', 'Iran (Islamic Republic of)',
    'Mexico', 'Japan', 'El Salvador', 'Bulgaria', 'Nigeria', 'South Sudan',
    'Guatemala', 'Ireland', 'Turkey', 'Peru', 'Democratic Republic of the Congo',
    'Canada', 'Switzerland', 'Denmark', 'Cameroon', 'Australia', 'Portugal',
    "Côte d'Ivoire", 'Sweden', 'Ethiopia', 'Thailand', 'Hungary', 'Kazakhstan',
    'Israel', 'Spain', 'Myanmar', 'Papua New Guinea', 'Madagascar', 'Lithuania',
    'Ghana', 'Azerbaijan', 'Armenia', 'Ukraine', 'Malaysia', 'Serbia', 'Slovakia',
    'Gambia', 'Ecuador', 'Bosnia and Herzegovina', 'Czechia', 'Argentina',
    'Dominican Republic', 'Guinea', 'Bolivia (Plurinational State of)',
    'Bangladesh', 'Sudan', 'Chile', 'Panama', 'Congo', 'Kyrgyzstan', 'Mauritius',
    'Greece', 'Malawi', 'Cuba', 'Saudi Arabia', 'Benin', 'Sierra Leone', 'Lebanon',
    'Uruguay', 'Namibia', 'Albania', 'Guyana', 'Senegal', 'Lesotho', 'Mongolia',
    'Venezuela (Bolivarian Republic of)', 'Solomon Islands', 'Paraguay',
    'Zimbabwe', 'Austria', 'Croatia', 'Honduras', 'Cambodia', 'Uganda',
    'Republic of Moldova', 'Angola', 'Kingdom of Eswatini', 'Afghanistan',
    'Uzbekistan', 'Zambia', 'Morocco', 'Belarus', 'Norway', 'Malta', 'Rwanda',
    'Sri Lanka', 'Botswana', 'Burundi', 'Jamaica', 'Central African Republic',
    'Kenya', 'Niger', 'Mali', 'Slovenia', 'Costa Rica', 'Nicaragua',
    'Republic of Korea', 'Burkina Faso', 'Cabo Verde', 'Mozambique', 'Mauritania',
    'Guinea-Bissau', 'United Arab Emirates', 'Eritrea', 'Tajikistan', 'Barbados',
    'Algeria', 'Iraq', 'Syrian Arab Republic', 'Gabon', 'Liberia',
    "Lao People's Democratic Republic", 'Bhutan', 'Yemen', 'Togo', 'Turkmenistan',
    'Saint Lucia', 'Kuwait', 'Libya', 'Georgia', 'Nepal', 'Estonia',
    'Trinidad and Tobago', 'Jordan', 'Cyprus', 'Montenegro', 'Comoros', 'Iceland',
    'Andorra', 'Oman', 'Equatorial Guinea', 'Luxembourg', 'Vanuatu', 'Aruba',
    'Belize', 'Qatar', 'Palau', 'Suriname', 'Singapore', 'Maldives', 'Bahrain',
    'Seychelles', 'State of Palestine', 'Saint Kitts and Nevis', 'Monaco'
]
# ------------------------
# Header
# ------------------------
st.markdown('<h1 class="main-title">🌫️ Air Quality Index Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict AQI category based on air quality measurements</p>', unsafe_allow_html=True)
if dt_pipeline is None:
    st.stop()
# ------------------------
# Layout with Right Sidebar
# ------------------------
left_col, main_col, right_col = st.columns([1, 3, 1])
# ------------------------
# Left Sidebar - Input Parameters
# ------------------------
with left_col:
    st.header("Input Parameters")
    
    aqi_value = st.slider("AQI Value", min_value=6, max_value=500, value=50)
    co_aqi = st.slider("CO AQI Value", min_value=0, max_value=133, value=10)
    ozone_aqi = st.slider("Ozone AQI Value", min_value=0, max_value=235, value=20)
    no2_aqi = st.slider("NO2 AQI Value", min_value=0, max_value=91, value=15)
    country = st.selectbox(
        "Select Country",
        options=country_options,
        index=country_options.index('Pakistan') if 'Pakistan' in country_options else 0
    )
    
    # Current values display
    st.markdown("---")
    st.markdown("**Current Values**")
    st.markdown(f"🌡️ AQI Value: **{aqi_value}**")
    st.markdown(f"🚗 CO AQI: **{co_aqi}**")
    st.markdown(f"☀️ Ozone AQI: **{ozone_aqi}**")
    st.markdown(f"🏭 NO2 AQI: **{no2_aqi}**")
    st.markdown(f"🌍 Country: **{country}**")
# ------------------------
# Right Sidebar - About AQI Parameters
# ------------------------
with right_col:
    st.markdown('<h3 style="margin-top: 0;">About AQI Parameters</h3>', unsafe_allow_html=True)  # Added style to remove top margin
    
    st.markdown("""
    <div class="info-box">
        <h4>🌡️ AQI Value</h4>
        <p>Overall air quality index (6-500 range)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🚗 CO AQI</h4>
        <p>Carbon Monoxide levels from vehicles</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>☀️ Ozone AQI</h4>
        <p>Ground-level ozone pollution</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🏭 NO2 AQI</h4>
        <p>Nitrogen Dioxide from industry</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
# ------------------------
# Main Content
# ------------------------
with main_col:
    # Prediction button
    if st.button("🔮 Predict AQI Category", type="primary", use_container_width=True):
        with st.spinner("Analyzing air quality data..."):
            # Prepare input data
            input_data = pd.DataFrame({
                'AQI Value': [aqi_value],
                'CO AQI Value': [co_aqi],
                'Ozone AQI Value': [ozone_aqi],
                'NO2 AQI Value': [no2_aqi],
                'Country': [country]
            })
            
            try:
                # Make prediction
                prediction = dt_pipeline.predict(input_data)[0]
                description = CATEGORY_DESCRIPTIONS[prediction]
                css_class = CATEGORY_CLASSES[prediction]
                
                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.description = description
                st.session_state.css_class = css_class
                
                st.success("✅ Prediction completed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display prediction if available
    if hasattr(st.session_state, 'prediction'):
        prediction = st.session_state.prediction
        description = st.session_state.description
        css_class = st.session_state.css_class
        
        # Main prediction result with category-specific styling
        st.markdown(f"""
        <div class="prediction-box {css_class}">
            <h2 class="prediction-title">{prediction}</h2>
            <p class="prediction-desc">{description}</p>
            <div class="prediction-bar">
                <div class="prediction-bar-fill" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Health recommendations
        st.markdown("### Health Recommendations")
        if prediction == "Good":
            st.markdown("""
            <div class="health-rec" style="background-color: #dcfce7; color: #166534; border-left: 4px solid #22c55e;">
                🌟 Perfect for outdoor activities! No health concerns.
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Moderate":
            st.markdown("""
            <div class="health-rec" style="background-color: #fef3c7; color: #a16207; border-left: 4px solid #eab308;">
                ⚠️ Generally safe. Sensitive individuals should monitor symptoms.
            </div>
            """, unsafe_allow_html=True)
        elif "Sensitive" in prediction:
            st.markdown("""
            <div class="health-rec" style="background-color: #fed7aa; color: #9a3412; border-left: 4px solid #f97316;">
                🚨 Sensitive groups should limit outdoor activities.
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Unhealthy":
            st.markdown("""
            <div class="health-rec" style="background-color: #fecaca; color: #991b1b; border-left: 4px solid #ef4444;">
                ⛔ Everyone should limit outdoor activities.
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "Very Unhealthy":
            st.markdown("""
            <div class="health-rec" style="background-color: #e9d5ff; color: #6b21a8; border-left: 4px solid #a855f7;">
                ☣️ Avoid outdoor activities. Close windows and use air purifiers.
            </div>
            """, unsafe_allow_html=True)
        else:  # Hazardous
            st.markdown("""
            <div class="health-rec" style="background-color: #fee2e2; color: #7f1d1d; border-left: 4px solid #991b1b;">
                ☢️ Emergency conditions! Stay indoors, avoid physical exertion.
            </div>
            """, unsafe_allow_html=True)
    # AQI Categories Reference
    st.markdown("### AQI Categories Reference")
    
    # Create 3 columns for categories
    col1, col2, col3 = st.columns(3)
    
    categories = [
        ("Good", "category-good", "0-50: No health risk"),
        ("Moderate", "category-moderate", "51-100: Sensitive groups monitor"),
        ("Unhealthy for Sensitive Groups", "category-unhealthy-sensitive", "101-150: Sensitive groups limit outdoor time"),
        ("Unhealthy", "category-unhealthy", "151-200: Everyone limit outdoor time"),
        ("Very Unhealthy", "category-very-unhealthy", "201-300: Avoid outdoor activities"),
        ("Hazardous", "category-hazardous", "301+: Emergency conditions")
    ]
    
    for i, (category, css_class, desc) in enumerate(categories):
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(f"""
            <div class="category-box {css_class}">
                <h4 class="category-title">{category}</h4>
                <p class="category-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("**AQI Predictor** - Powered by Machine Learning | Built with Streamlit")