import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/best_churn_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Header
st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Choose a section:", ["🔮 Prediction", "📊 Model Info", "❓ Help"])

# Load model
model = load_model()

if model is None:
    st.stop()

if page == "🔮 Prediction":
    st.markdown("### 📝 Enter Customer Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"], help="Customer's gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], help="Is the customer 65 or older?")
        partner = st.selectbox("Has Partner", ["No", "Yes"], help="Does the customer have a partner?")
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], help="Does the customer have dependents?")
        
        st.markdown("#### 📞 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        
        st.markdown("#### 🛡️ Add-on Services")
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    
    with col2:
        st.markdown("#### 🎬 Entertainment")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        
        st.markdown("#### 💼 Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with the company?")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        st.markdown("#### 💰 Financial")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure), step=50.0)

    # Prediction button
    if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = np.array([[
            1 if gender == "Male" else 0,
            1 if senior == "Yes" else 0,
            1 if partner == "Yes" else 0,
            1 if dependents == "Yes" else 0,
            tenure,
            1 if phone_service == "Yes" else 0,
            1 if multiple_lines == "Yes" else 0,
            1 if internet_service == "Fiber optic" else 0,
            1 if online_security == "Yes" else 0,
            1 if online_backup == "Yes" else 0,
            1 if device_protection == "Yes" else 0,
            1 if tech_support == "Yes" else 0,
            1 if streaming_tv == "Yes" else 0,
            1 if streaming_movies == "Yes" else 0,
            1 if contract == "Month-to-month" else 0,
            1 if paperless_billing == "Yes" else 0,
            1 if payment_method == "Electronic check" else 0,
            monthly_charges,
            total_charges
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown(f'<div class="risk-high"><h3>⚠️ HIGH RISK</h3><p>Customer likely to churn</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><h3>✅ LOW RISK</h3><p>Customer likely to stay</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}", delta=f"{probability-0.5:.1%}")
        
        with col3:
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if probability > 0.7:
            st.warning("🚨 **Immediate Action Required:**")
            recommendations = [
                "Contact customer immediately to understand concerns",
                "Offer retention incentives or discounts",
                "Consider upgrading their service plan",
                "Provide personalized customer support"
            ]
        elif probability > 0.3:
            st.info("⚠️ **Monitor Closely:**")
            recommendations = [
                "Regular check-ins with customer",
                "Offer additional services that add value",
                "Ensure customer satisfaction through surveys",
                "Provide proactive customer support"
            ]
        else:
            st.success("✅ **Customer is Stable:**")
            recommendations = [
                "Continue providing excellent service",
                "Consider upselling opportunities",
                "Use as reference for testimonials",
                "Maintain regular communication"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

elif page == "📊 Model Info":
    st.markdown("### 🔍 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Model Details
        - **Algorithm**: Machine Learning Classifier
        - **Features**: 19 customer attributes
        - **Target**: Binary classification (Churn/No Churn)
        - **Output**: Probability score (0-1)
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Key Features
        - Demographics (Age, Gender, Family)
        - Service Usage (Internet, Phone, Streaming)
        - Contract Details (Type, Billing, Payment)
        - Financial Metrics (Charges, Tenure)
        """)
    
    # Feature importance (mock data for demonstration)
    st.markdown("#### 📊 Feature Importance")
    features = ['Contract Type', 'Tenure', 'Payment Method', 'Monthly Charges', 'Internet Service',
               'Tech Support', 'Online Security', 'Total Charges', 'Paperless Billing', 'Senior Citizen']
    importance = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
    
    fig = px.bar(x=importance, y=features, orientation='h', 
                 title="Top 10 Most Important Features",
                 color=importance, color_continuous_scale='viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

else:  # Help page
    st.markdown("### ❓ How to Use This App")
    
    st.markdown("""
    #### 🎯 Purpose
    This app predicts whether a customer is likely to churn (stop using the service) based on their profile and usage patterns.
    
    #### 📝 How to Use
    1. **Navigate to the Prediction tab**
    2. **Fill in customer information** in the form
    3. **Click "Predict Churn Risk"** to get results
    4. **Review recommendations** based on the risk level
    
    #### 📊 Understanding Results
    - **Churn Probability**: Percentage likelihood of customer leaving
    - **Risk Levels**:
      - 🟢 **Low (0-30%)**: Customer is likely to stay
      - 🟡 **Medium (30-70%)**: Monitor customer closely
      - 🔴 **High (70-100%)**: Immediate retention action needed
    
    #### 🔧 Tips for Best Results
    - Ensure all information is accurate and up-to-date
    - Pay special attention to contract type and payment method
    - Consider the customer's service usage patterns
    - Use tenure and charges data to understand customer value
    """)
    
    st.markdown("---")
    st.markdown("#### 📞 Need Help?")
    st.info("For technical support or questions about the model, please contact your data science team.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🏢 Customer Churn Prediction System | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)