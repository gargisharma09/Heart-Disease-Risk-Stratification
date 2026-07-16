# Heart Disease Risk Stratification Using Stacked Ensemble Learning with Explainable AI (SHAP)

## Overview
This project aims to accurately predict the risk of heart disease in patients using a robust machine learning pipeline. Instead of relying on a single algorithm, we employ a **Stacked Ensemble Learning** approach (combining Random Forest, XGBoost, Gradient Boosting, and Logistic Regression). To ensure that our predictions are not a "black box", we use **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable explanations for every prediction.

## Project Structure
```
HeartDiseaseRiskStratification/
│
├── data/                  # Contains raw and preprocessed datasets
├── notebooks/             # Jupyter Notebooks for EDA, Preprocessing, Modeling, and SHAP
├── models/                # Saved serialized models (scaler, stacking model)
├── app/                   # Streamlit web application
├── reports/               # Project report and PPT content
├── images/                # Visualizations and plots generated
├── requirements.txt       # Python dependencies
├── download_data.py       # Script to fetch UCI dataset
├── generate_notebooks.py  # Script to generate the Jupyter notebooks
└── README.md              # Project documentation
```

## How to Run the Project

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   ```bash
   python download_data.py
   ```

3. **Generate Notebooks** (If not already present)
   ```bash
   python generate_notebooks.py
   ```

4. **Run Jupyter Notebooks**
   Run the notebooks in `notebooks/` directory sequentially from 01 to 04 to explore data, train models, and generate explainability plots.
   ```bash
   jupyter notebook
   ```

5. **Run the Streamlit Application**
   Once models are trained and saved in the `models/` folder, run the web application:
   ```bash
   streamlit run app/app.py
   ```

## Key Technologies
- **Python** for Data Science
- **Pandas & NumPy** for data manipulation
- **Scikit-Learn & XGBoost** for machine learning models
- **SHAP** for Model Explainability
- **Streamlit** for the web application dashboard
- **Matplotlib & Seaborn** for data visualization
