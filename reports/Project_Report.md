# Project Report: Heart Disease Risk Stratification Using Stacked Ensemble Learning with Explainable AI (SHAP)

## 1. Introduction
Heart disease remains one of the leading causes of mortality worldwide. Early detection and accurate risk stratification can significantly improve patient outcomes. Traditional machine learning models often suffer from a lack of interpretability (the "black box" problem), making it difficult for healthcare professionals to trust the predictions. This project aims to address these challenges by developing a robust **Stacked Ensemble Machine Learning Model** combined with **SHAP (SHapley Additive exPlanations)** for full transparency.

## 2. Dataset and Preprocessing
We utilized the UCI Heart Disease dataset, which contains clinical, demographic, and physiological data points for patients.

### Data Cleaning and Feature Engineering
- **Missing Values**: Handled using median imputation for numerical features to avoid skewness.
- **Outlier Detection**: Implemented Interquartile Range (IQR) and Z-score methods to detect and cap outliers in continuous variables such as resting blood pressure (trestbps) and cholesterol (chol).
- **Scaling**: All numerical features were standardized using StandardScaler to ensure that features like max heart rate and cholesterol do not disproportionately dominate the models.
- **Encoding**: Categorical features were appropriately encoded.

## 3. Exploratory Data Analysis (EDA)
Comprehensive EDA was conducted to identify underlying patterns:
- Analyzed the distribution of the target variable to ensure class balance.
- Investigated the correlation matrix, identifying key predictors like chest pain type, exercise-induced angina, and maximum heart rate.
- Pairplots and distribution plots revealed relationships between age, cholesterol, and heart disease risk.

## 4. Modeling Approach
Rather than relying on a single classifier, we employed a Stacking Classifier architecture to leverage the strengths of multiple algorithms:
1. **Base Learners**:
   - Random Forest
   - Gradient Boosting
   - XGBoost
2. **Meta-Learner**:
   - Logistic Regression
The base models capture different non-linear patterns, and the meta-learner optimally combines their predictions, leading to improved accuracy and robustness.

## 5. Model Evaluation
The models were evaluated using multiple metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- The Stacked Ensemble demonstrated superior performance, proving highly reliable at minimizing false negatives (crucial for medical diagnostics).
- Confusion matrices and ROC curves were generated to visualize performance.

## 6. Explainability using SHAP
To ensure clinical applicability, SHAP was integrated to explain predictions:
- **Global Explainability (Summary Plots)**: Showed the overall importance of features. Typically, chest pain type, max heart rate, and ST depression are the strongest predictors.
- **Local Explainability (Waterfall Plots)**: For any individual patient, the model provides a visual breakdown of how specific factors (e.g., high cholesterol or old age) contributed positively or negatively to their risk score.

## 7. Streamlit Web Application
A user-friendly web interface was built using Streamlit. It allows users to:
- Input patient clinical parameters.
- Receive real-time predictions of heart disease risk (Low, Medium, High).
- View a dynamically generated SHAP waterfall plot explaining the prediction for that specific patient.

## 8. Conclusion
This project successfully combined advanced ensemble machine learning techniques with explainable AI to create a powerful, interpretable diagnostic support tool. The final model is robust, and the transparent SHAP explanations bridge the gap between AI and clinical trust.
