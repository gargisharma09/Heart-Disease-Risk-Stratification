# Presentation: Heart Disease Risk Stratification Using Stacked Ensemble Learning with Explainable AI

---
## Slide 1: Title Slide
**Heart Disease Risk Stratification**
*Using Stacked Ensemble Learning with Explainable AI (SHAP)*
- Presenter Name
- Date

---
## Slide 2: Problem Statement
- Heart disease is a leading global cause of death.
- Early detection saves lives.
- Traditional ML models act as "Black Boxes" – clinicians can't trust what they can't understand.
- **Goal**: Build a highly accurate predictive model that is also fully transparent and interpretable.

---
## Slide 3: Dataset Overview
- **Source**: UCI Heart Disease Dataset
- **Features**: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, etc.
- **Target**: Presence (1) or Absence (0) of Heart Disease.

---
## Slide 4: Data Preprocessing
- **Missing Values**: Handled via median imputation.
- **Outlier Detection**: IQR and Z-score methods applied to BP and Cholesterol.
- **Scaling**: StandardScaler applied for uniformity.
- **Encoding**: Processed categorical values for ML compatibility.

---
## Slide 5: Exploratory Data Analysis
- Analyzed feature distributions and correlations.
- Strong correlations found between Chest Pain Type, Max Heart Rate, and Heart Disease risk.

---
## Slide 6: Modeling Strategy (Stacked Ensemble)
- **Base Models**:
  1. Random Forest
  2. Gradient Boosting
  3. XGBoost
- **Meta-Learner**:
  - Logistic Regression
- **Why?**: Combines the strengths of individual trees while reducing variance and bias.

---
## Slide 7: Model Performance
- Evaluated models using Accuracy, Precision, Recall, F1, and ROC-AUC.
- Stacked Ensemble achieved the highest robustness and recall.
- Vital to minimize False Negatives in medical settings.

---
## Slide 8: Explainable AI with SHAP
- **SHAP**: SHapley Additive exPlanations
- Solves the "Black Box" problem.
- **Global Explainability**: Identifies the most important features across the entire dataset.
- **Local Explainability**: Explains the exact reasoning for a single patient's prediction.

---
## Slide 9: Web Application Integration
- Developed a user-friendly Streamlit Web App.
- Allows clinicians to input patient data easily.
- Provides real-time risk stratification (Low, Medium, High).
- Displays a SHAP Waterfall Plot instantly for trust and transparency.

---
## Slide 10: Conclusion
- Created an end-to-end, highly accurate Heart Disease prediction system.
- Ensured clinical viability through Explainable AI.
- The Stacked approach provides top-tier accuracy, while SHAP provides top-tier interpretability.

---
## Slide 11: Questions?
- Thank you for your time.
- Any questions?
