# 📦 Product Return Behaviour & Return-Risk Intelligence

### Machine Learning-Based E-Commerce Return Risk Prediction System

A machine learning application designed to predict the probability of an e-commerce product being returned based on customer behaviour, product details, purchase patterns, and shipping information.

The project provides return-risk classification, risk indicators, and business recommendations through an interactive Streamlit dashboard.

## 🎯 Project Objectives

- Predict the probability of product returns.
- Classify orders into Low, Medium, and High return-risk categories.
- Analyze customer return history and order characteristics.
- Compare multiple machine learning algorithms.
- Provide interpretable risk indicators and business recommendations.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Serialization:** Joblib
- **Version Control:** Git and GitHub

## 🤖 Machine Learning Models

The project trains and compares three classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Models are evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

The model with the highest ROC-AUC score is selected and saved for predictions.

## 📊 Dataset

The project uses a synthetically generated dataset containing 5,000 e-commerce orders.

Features include:

- Product category and price
- Discount percentage and quantity
- Customer age and gender
- Payment method
- Shipping duration
- Previous orders and returns
- Customer and product ratings

Additional engineered features include previous return ratio, order value, and discounted order value.

**Note:** The dataset is synthetic. Predictions are intended for educational and demonstration purposes, not validated real-world business decisions.

## ✨ Key Features

- Interactive order information form
- Product return probability prediction
- Low, Medium, and High risk classification
- Customer return-history analysis
- Risk indicator identification
- Business recommendations
- Machine learning model comparison

## 🚀 Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Yetesh-sudu/product-return-risk-intelligence.git
```

Navigate to the project:

```bash
cd product-return-risk-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the dataset and trained model are not already included, run:

```bash
python generate_dataset.py
python preprocessing.py
python compare_models.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## 🔄 Project Workflow

1. Generate synthetic e-commerce order data.
2. Clean and preprocess the dataset.
3. Engineer additional predictive features.
4. Train and compare classification models.
5. Select the best-performing model using ROC-AUC.
6. Save the trained model using Joblib.
7. Predict product return probability through Streamlit.
8. Display risk indicators and business recommendations.

## 🔮 Future Enhancements

- Train and validate models using real-world datasets.
- Add explainable AI using SHAP.
- Implement advanced feature engineering.
- Introduce customer-level analytics.
- Add dashboard visualizations and downloadable reports.
- Deploy the application publicly.

## 👨‍💻 Developer

**Yetesh**

B.Tech — Artificial Intelligence & Machine Learning  
Amity University Haryana

GitHub: https://github.com/Yetesh-sudu

## 📌 Disclaimer

This is an academic machine learning project. Its current model uses synthetic data and should not be used as the sole basis for real-world customer or business decisions.
