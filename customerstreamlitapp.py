#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Customer Segmentation Streamlit App

# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation Using Behavioral Patterns")
st.write(
    "This application uses a pre-trained K-Means model to segment customers "
    "based on spending behavior and purchase patterns."
)

# Load Pickle Files (Model & Scaler)
# Loading trained K-Means model
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

# Loading scaler used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Dataset Uploading
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Reading dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Data Cleaning
    # Handling missing Income values
    df['Income'].fillna(df['Income'].median(), inplace=True)

    # Feature Engineering
    # Creating Total_Spend feature
    df['Total_Spend'] = (
        df['MntWines'] +
        df['MntMeatProducts'] +
        df['MntFishProducts'] +
        df['MntSweetProducts'] +
        df['MntGoldProds']
    )

    # Creating Total_Purchases feature
    df['Total_Purchases'] = df['NumWebPurchases'] + df['NumStorePurchases']

    # Inverting Recency so higher = better engagement
    df['Engagement'] = 1 / (df['Recency'] + 1)

    # Feature Selection
    features = df[
        ['Income', 'Total_Spend', 'Total_Purchases','Engagement']
    ]

    # Feature Scaling (Using Loaded Scaler)
    scaled_features = scaler.transform(features)

    # Predicting Clusters Using Pickle Model
    df['Cluster'] = kmeans_model.predict(scaled_features)

    # Cluster Summary
    st.subheader("📌 Cluster Summary (Average Values)")

    cluster_summary = df.groupby('Cluster')[
        ['Income', 'Total_Spend', 'Engagement', 'Total_Purchases']
    ].mean()

    st.dataframe(cluster_summary)

    # PCA for Cluster Visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'Cluster': df['Cluster']
    })

    # Plotting Clusters
    st.subheader("📈 Customer Segments (PCA Visualization)")

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='tab10',
        ax=ax
    )

    ax.set_title("Customer Segments Visualized Using PCA")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)
    
    # Download Final Clustered Dataset
    st.subheader("⬇️ Download Clustered Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered Data",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a customer dataset to start segmentation.")


# In[ ]:




