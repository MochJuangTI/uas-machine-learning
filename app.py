import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = './DATA RUMAH (1).xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Remove duplicate entries
data = data.drop_duplicates()

# Select the relevant columns
data = data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']]

# Normalize the data
scaler = MinMaxScaler()
data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']] = scaler.fit_transform(data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']])

# Classify HARGA (RUPIAH) into quartiles
data['PRICE_QUARTILE'] = pd.qcut(data['HARGA (RUPIAH)'], q=4, labels=False)

# Display the number of houses in each quartile
quartile_counts = data['PRICE_QUARTILE'].value_counts().sort_index()

# Visualize the quartile distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=quartile_counts.index, y=quartile_counts.values, palette='viridis')
plt.xlabel('Price Quartile')
plt.ylabel('Number of Houses')
plt.title('Distribution of Houses in Each Price Quartile')
plt.show()

# Define the independent and dependent variables
X = data[['LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']]
y = data['PRICE_QUARTILE']

# Handle missing values using imputation and standardize the features
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X = pipeline.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter tuning function
def hyperparameter_tuning(model, params, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Define hyperparameters for tuning
params = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Decision Tree': {'max_depth': [None, 10, 20, 30, 40, 50]},
    'Random Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'Support Vector Machine': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
}

# Train, tune, and evaluate each model
results = {}
feature_importances = {}

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    best_model = hyperparameter_tuning(model, params[model_name], X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    mcc = matthews_corrcoef(y_test, y_pred)

    # For ROC-AUC score, we need the predicted probabilities
    y_prob = best_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovo')

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'MCC': mcc,
        'ROC-AUC': roc_auc
    }

    print(f'{model_name} Best Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Q{i}' for i in range(1, 5)], yticklabels=[f'Q{i}' for i in range(1, 5)])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Collect feature importances for models that support it
    if hasattr(best_model, 'feature_importances_'):
        feature_importances[model_name] = best_model.feature_importances_

# Compare the models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[results[model][metric] for model in results], palette='viridis')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.title(f'Comparison of Machine Learning Models by {metric}')
    plt.ylim(0, 1)
    plt.show()

# Feature importance analysis
for model_name, importances in feature_importances.items():
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances, y=['LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI'], palette='viridis')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importances for {model_name}')
    plt.show()
