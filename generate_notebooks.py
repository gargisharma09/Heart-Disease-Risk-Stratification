import nbformat as nbf
import os

# Ensure directories exist
os.makedirs('notebooks', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    cells = []
    for c_type, content in cells_content:
        if c_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif c_type == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

# 01_EDA.ipynb
eda_cells = [
    ('markdown', '# Heart Disease Risk Stratification - Exploratory Data Analysis (EDA)'),
    ('markdown', '## 1. Import Libraries'),
    ('code', '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
'''),
    ('markdown', '## 2. Load Dataset'),
    ('code', '''df = pd.read_csv('../data/heart.csv')
display(df.head())
print(df.info())
'''),
    ('markdown', '## 3. Data Inspection & Cleaning'),
    ('code', '''# Target Variable might have values 0, 1, 2, 3, 4 where >0 means presence of heart disease.
# We will convert this into a binary classification problem: 0 = No Heart Disease, 1 = Heart Disease
if df['target'].max() > 1:
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# The UCI dataset has '?' for missing values in 'ca' and 'thal' occasionally, if imported directly.
# Let's check for any missing values or '?'
df.replace('?', np.nan, inplace=True)
print("Missing values per column:\\n", df.isnull().sum())

# We will save this for preprocessing step, for now just drop them to do EDA
df_eda = df.dropna()
'''),
    ('markdown', '## 4. Exploratory Data Analysis'),
    ('code', '''# 1. Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df_eda, x='target', palette='Set2')
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.show()

# 2. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df_eda, x='age', hue='target', kde=True, palette='Set1')
plt.title('Age Distribution by Target')
plt.show()

# 3. Sex and Heart Disease (0 = Female, 1 = Male)
plt.figure(figsize=(6,4))
sns.countplot(data=df_eda, x='sex', hue='target', palette='Set2')
plt.title('Heart Disease Frequency by Sex')
plt.show()

# 4. Chest Pain Type vs Target
plt.figure(figsize=(8,5))
sns.countplot(data=df_eda, x='cp', hue='target', palette='Set1')
plt.title('Chest Pain Type vs Target')
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(12,10))
# convert object types to numeric if any for correlation
corr_df = df_eda.apply(pd.to_numeric)
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 6. Pairplot for continuous variables
cont_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
sns.pairplot(corr_df[cont_vars], hue='target', palette='husl')
plt.show()
''')
]

# 02_Preprocessing.ipynb
prep_cells = [
    ('markdown', '# Preprocessing & Feature Engineering'),
    ('code', '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('../data/heart.csv')

# Handle binary target
if df['target'].max() > 1:
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Impute with median for all columns
df.fillna(df.median(), inplace=True)

print("Missing values after imputation:\\n", df.isnull().sum())
'''),
    ('markdown', '## Outlier Detection (IQR & Z-score)'),
    ('code', '''def handle_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap outliers
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

num_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
for col in num_cols:
    df = handle_outliers_iqr(df, col)
'''),
    ('markdown', '## Train Test Split & Scaling'),
    ('code', '''X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Save preprocessed data
X_train_scaled.to_csv('../data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('../data/X_test_scaled.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

# Save scaler for later use
import joblib
joblib.dump(scaler, '../models/scaler.pkl')
print("Preprocessing complete. Scaler and data saved.")
''')
]

# 03_ModelBuilding.ipynb
model_cells = [
    ('markdown', '# Model Building and Stacking Classifier'),
    ('code', '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

X_train = pd.read_csv('../data/X_train_scaled.csv')
X_test = pd.read_csv('../data/X_test_scaled.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_test = pd.read_csv('../data/y_test.csv').squeeze()
'''),
    ('markdown', '## Train Base Models'),
    ('code', '''models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })
    
results_df = pd.DataFrame(results)
display(results_df)
'''),
    ('markdown', '## Stacking Classifier'),
    ('code', '''estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('gb', GradientBoostingClassifier(random_state=42))
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)
y_prob_stack = stack_model.predict_proba(X_test)[:, 1]

stack_results = {
    'Model': 'Stacking Classifier',
    'Accuracy': accuracy_score(y_test, y_pred_stack),
    'Precision': precision_score(y_test, y_pred_stack),
    'Recall': recall_score(y_test, y_pred_stack),
    'F1-Score': f1_score(y_test, y_pred_stack),
    'ROC-AUC': roc_auc_score(y_test, y_prob_stack)
}

results_df = pd.concat([results_df, pd.DataFrame([stack_results])], ignore_index=True)
display(results_df)

# Save Stacking model
joblib.dump(stack_model, '../models/stacking.pkl')
'''),
    ('markdown', '## Evaluation Visualizations'),
    ('code', '''# Confusion Matrix for Stacking
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Stacking Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('../images/confusion_matrix.png')
plt.show()
''')
]

# 04_SHAP.ipynb
shap_cells = [
    ('markdown', '# Model Explainability using SHAP'),
    ('code', '''import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# We'll use XGBoost as the explainer model since SHAP TreeExplainer supports it well,
# or we can use the XGBoost component of our Stacking model.
X_train = pd.read_csv('../data/X_train_scaled.csv')
X_test = pd.read_csv('../data/X_test_scaled.csv')

# Load the stacking model
stack_model = joblib.load('../models/stacking.pkl')

# Extract the XGBoost model from the stacking estimators
xgb_model = stack_model.named_estimators_['xgb']

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
'''),
    ('markdown', '## Summary Plot'),
    ('code', '''plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('../images/shap_summary.png', bbox_inches='tight')
plt.show()
'''),
    ('markdown', '## Waterfall Plot (Local Explanation)'),
    ('code', '''# For the first patient in the test set
idx = 0
plt.figure()
# shap.plots.waterfall is preferred for modern SHAP, requires Explanation object
explainer_v2 = shap.Explainer(xgb_model)
shap_values_v2 = explainer_v2(X_test)

shap.plots.waterfall(shap_values_v2[idx], show=False)
plt.savefig('../images/shap_waterfall.png', bbox_inches='tight')
plt.show()
''')
]

create_notebook('notebooks/01_EDA.ipynb', eda_cells)
create_notebook('notebooks/02_Preprocessing.ipynb', prep_cells)
create_notebook('notebooks/03_ModelBuilding.ipynb', model_cells)
create_notebook('notebooks/04_SHAP.ipynb', shap_cells)

print("Successfully generated all notebooks!")
