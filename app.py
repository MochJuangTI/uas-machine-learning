import pandas as pd
import numpy as np
import streamlit as st
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

st.title('House Price Prediction and Analysis')

uploaded_file = st.file_uploader("./DATA RUMAH (1).xlsx", type="xlsx")
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, engine='openpyxl')

    data = data.drop_duplicates()
    data = data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']]

    scaler = MinMaxScaler()
    data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']] = scaler.fit_transform(data[['HARGA (RUPIAH)', 'LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']])

    data['PRICE_QUARTILE'] = pd.qcut(data['HARGA (RUPIAH)'], q=4, labels=False)

    quartile_counts = data['PRICE_QUARTILE'].value_counts().sort_index()

    st.subheader('Distribution of Houses in Each Price Quartile')
    fig, ax = plt.subplots()
    sns.barplot(x=quartile_counts.index, y=quartile_counts.values, palette='viridis', ax=ax)
    ax.set_xlabel('Price Quartile')
    ax.set_ylabel('Number of Houses')
    st.pyplot(fig)

    X = data[['LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI']]
    y = data['PRICE_QUARTILE']

    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    def hyperparameter_tuning(model, params, X_train, y_train):
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    params = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
        'Decision Tree': {'max_depth': [None, 10, 20, 30, 40, 50]},
        'Random Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]},
        'Support Vector Machine': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]},
        'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }

    results = {}
    feature_importances = {}

    for model_name, model in models.items():
        st.write(f"Tuning {model_name}...")
        best_model = hyperparameter_tuning(model, params[model_name], X_train, y_train)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
        recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
        f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        mcc = matthews_corrcoef(y_test, y_pred)

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

        st.write(f'{model_name} Best Accuracy: {accuracy:.2f}')
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Q{i}' for i in range(1, 5)], yticklabels=[f'Q{i}' for i in range(1, 5)], ax=ax)
        ax.set_title(f'Confusion Matrix for {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        if hasattr(best_model, 'feature_importances_'):
            feature_importances[model_name] = best_model.feature_importances_

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC']
    for metric in metrics:
        fig, ax = plt.subplots()
        sns.barplot(x=list(results.keys()), y=[results[model][metric] for model in results], palette='viridis', ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'Comparison of Machine Learning Models by {metric}')
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    for model_name, importances in feature_importances.items():
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=['LUAS BANGUNAN', 'LUAS TANAH', 'KAMAR TIDUR', 'KAMAR MANDI', 'GARASI'], palette='viridis', ax=ax)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Feature Importances for {model_name}')
        st.pyplot(fig)
else:
    st.write("Please upload an Excel file.")
