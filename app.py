import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, f1_score

# App Title and Introduction
st.title("Credit Card Fraud Detection System")
st.markdown("""
This app detects fraudulent transactions using various machine learning models.
Explore the dataset, visualize fraud distribution, and predict fraud using different models.
""")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

df = load_data()

# Sidebar for user input
st.sidebar.header('Filter Dataset')

# Transaction amount range filter
amount_range = st.sidebar.slider("Transaction Amount Range", 0, int(df["Amount"].max()), (0, 2000))
df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]

# Dataset Overview
st.subheader("Dataset Overview")
if st.checkbox("Show Dataset"):
    st.write(df_filtered.head())

# Class distribution (Fraud vs Non-Fraud)
st.subheader("Class Distribution (Fraud vs Non-Fraud)")
class_counts = df_filtered['Class'].value_counts()
st.bar_chart(class_counts)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_filtered.corr(), cmap='coolwarm', annot=False)
    st.pyplot(plt)
    plt.clf()  # Clear the figure after plotting

# Scatter Plot: Amount vs V1 (Interactive)
st.subheader("Interactive Visualization")
st.markdown("Transaction Amount vs V1 for Fraud and Non-Fraud Transactions")
fig = px.scatter(df_filtered, x="Amount", y="V1", color="Class", title="Amount vs V1 by Class")
st.plotly_chart(fig)

# Sidebar for Model Selection
st.sidebar.subheader("Model Selection")
model_choices = st.sidebar.multiselect("Choose Machine Learning Models", 
                                        ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost"],
                                        default=["Logistic Regression"])

# Preprocessing data for ML models
X = df_filtered.drop(columns=['Class'], axis=1)
y = df_filtered['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Training and Prediction
def train_and_predict(models):
    results = {}
    
    for model_name in models:
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "XGBoost":
            model = XGBClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[model_name] = (accuracy, f1)
        
    return results

# Train the selected models
if st.sidebar.button("Train Models"):
    with st.spinner("Training the models..."):
        results = train_and_predict(model_choices)
        st.success("Model training completed!")

        # Display Results in a DataFrame
        results_df = pd.DataFrame(results).T
        results_df.columns = ['Accuracy', 'F1 Score']
        st.subheader("Model Performance Comparison")
        st.write(results_df)

        # Visualization of Model Performance
        st.subheader("Model Performance Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        results_df[['Accuracy', 'F1 Score']].plot(kind='bar', ax=ax)
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        st.pyplot(fig)
        plt.clf()  # Clear the figure after plotting
        
        # ROC Curve for each model
        for model_name in model_choices:
            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "XGBoost":
                model = XGBClassifier()

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, lw=2, label='{} (area = {:.2f})'.format(model_name, roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot(plt)
            plt.clf()  # Clear the figure after plotting