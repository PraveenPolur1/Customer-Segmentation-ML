# 📊 Customer Segmentation Using Machine Learning

## 🚀 Project Overview

This project builds a **Customer Segmentation System** using the **K-Means Clustering algorithm** to group customers based on purchasing behavior and engagement metrics.

The final model is deployed using a **Streamlit Web Application**, allowing users to upload their dataset and generate customer segments instantly.

---

## 🎯 Business Problem

Marketing teams need to identify:

- High-value customers
- Potential churn customers
- Low-engagement customers
- Premium spenders

Instead of mass marketing, businesses can use segmentation to:

✔ Improve targeting  
✔ Increase conversion rates  
✔ Optimize marketing budget  
✔ Personalize campaigns  

---

## 📂 Project Structure

```
Customer-Segmentation-ML/
│
├── CustomerSegmentationModelCode.ipynb   # EDA & Model Training
├── customerstreamlit1.py                 # Streamlit App
├── marketing_campaign.xlsx               # Dataset
├── kmeans_model.pkl                      # Trained Model
├── scaler.pkl                            # Feature Scaler
├── requirements.txt
└── README.md
```

---

## 🧠 Model Used: K-Means Clustering

### Why K-Means?

K-Means is an unsupervised algorithm used to:

- Partition data into K clusters
- Minimize within-cluster variance
- Group similar behavioral patterns

---

### Mathematical Objective

K-Means minimizes:

Σ Σ || xi - μk ||²

Where:

- xi = data point
- μk = centroid of cluster k

---

## ⚙️ Features Used for Segmentation

The following engineered features were used:

- Income
- Total_Spend
- Total_Purchases
- Engagement (derived from Recency)

### Feature Engineering:

- Created `Total_Spend`
- Created `Total_Purchases`
- Inverted Recency into Engagement score

---

## 🔬 Data Processing Pipeline

1. Handle missing Income values
2. Feature Engineering
3. Feature Scaling using StandardScaler
4. K-Means Prediction
5. PCA Visualization
6. Cluster Summary Analysis

---

## 📊 Visualization

- PCA used to reduce dimensionality to 2D
- Cluster visualization using seaborn scatter plot
- Cluster averages displayed in dashboard

---

## 🌐 Streamlit Deployment

The app allows:

✔ Upload CSV or Excel dataset  
✔ Automatic segmentation  
✔ Cluster summary display  
✔ PCA visualization  
✔ Download clustered dataset  

Run locally:

```bash
streamlit run customerstreamlit1.py
```

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit

---

## 📈 Skills Demonstrated

- Unsupervised Learning
- Feature Engineering
- Data Preprocessing
- PCA Dimensionality Reduction
- Model Serialization (Pickle)
- Web App Deployment
- End-to-End ML Pipeline

---

## ⚠️ Limitations

- K-Means assumes spherical clusters
- Sensitive to number of clusters (K)
- Scaling required
- No real-time data streaming

---

## 🔮 Future Improvements

- Compare with DBSCAN & Hierarchical Clustering
- Add 3D PCA visualization
- Add cluster interpretation dashboard
- Deploy on Streamlit Cloud

---

## 👨‍💻 Author

Praveen Poluri  
Machine Learning & Data Science Enthusiast

---

⭐ If you found this project helpful, consider giving it a star!

