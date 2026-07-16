import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Heart Disease Risk Stratification", layout="wide")

# Title
st.title("🫀 Heart Disease Risk Stratification Using Stacked Ensemble & SHAP")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["🏠 Home", "🧠 Predict Heart Disease", "ℹ️ About Project"])

# Load models and scalers (cached for performance)
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('../models/scaler.pkl')
        stacking_model = joblib.load('../models/stacking.pkl')
        xgb_model = stacking_model.named_estimators_['xgb']
        explainer = shap.Explainer(xgb_model)
        return scaler, stacking_model, explainer, xgb_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have run the notebooks first.")
        return None, None, None, None

scaler, stack_model, explainer, xgb_model = load_models()

if page == "🏠 Home":
    st.write("### Welcome to the Heart Disease Risk Stratification App")
    st.write("""
    This application uses a Stacked Ensemble Machine Learning model (Random Forest, XGBoost, Gradient Boosting, Logistic Regression)
    to predict the risk of heart disease based on clinical features.
    
    It also utilizes **Explainable AI (SHAP)** to provide a transparent breakdown of why the model made a specific prediction for a patient.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Heart_diagram-en.svg/1024px-Heart_diagram-en.svg.png", width=400)

elif page == "🧠 Predict Heart Disease":
    st.header("Enter Patient Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (cp) [0-3]", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg) [0-2]", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
        
    with col3:
        slope = st.selectbox("Slope of ST Segment (slope) [0-2]", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca) [0-4]", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal) [0-3]", [0, 1, 2, 3])

    if st.button("Predict Risk & Explain"):
        if stack_model is not None:
            # Prepare input
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            # Scale
            input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            
            # Predict
            prob = stack_model.predict_proba(input_scaled)[0][1]
            
            st.write("---")
            st.subheader("Prediction Result")
            
            if prob < 0.4:
                st.success(f"**Low Risk** (Probability: {prob:.2f})")
            elif prob < 0.7:
                st.warning(f"**Medium Risk** (Probability: {prob:.2f})")
            else:
                st.error(f"**High Risk** (Probability: {prob:.2f})")
                
            st.write("---")
            st.subheader("SHAP Local Explanation (Waterfall Plot)")
            st.write("This plot explains how each feature contributed to this specific patient's risk.")
            
            shap_values_local = explainer(input_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_local[0], show=False)
            st.pyplot(fig)
            plt.close()

elif page == "ℹ️ About Project":
    st.header("About")
    st.write("This is a comprehensive academic project for Heart Disease Risk Stratification.")
    st.write("- **Models**: Logistic Regression, RandomForest, GradientBoosting, XGBoost, Stacking")
    st.write("- **Explainability**: SHAP (SHapley Additive exPlanations)")
