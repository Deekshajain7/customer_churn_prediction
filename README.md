📊 Customer Churn Prediction App
📌 Project Overview

This project is an end-to-end Machine Learning application that predicts whether a customer is likely to churn (leave the company) or stay.

The model is trained on the Telco Customer Churn dataset from Kaggle and deployed using Streamlit to create an interactive web application.

The goal of this project is to help businesses identify customers at risk of churn and take proactive retention measures.

🎯 Problem Statement

Customer churn is a major business problem in telecom and subscription-based industries. Acquiring new customers is more expensive than retaining existing ones.

This project builds a Machine Learning classification model to:

Predict customer churn (Yes/No)

Calculate churn probability

Provide business insights

📂 Dataset Information

Dataset: Telco Customer Churn

Source: Kaggle

Total Records: ~7000 customers

Features: 20+ customer attributes

Target Variable: Churn (Yes/No)

Important features include:

Tenure

Monthly Charges

Total Charges

Contract Type

Internet Service

Payment Method

🧠 Machine Learning Algorithms Used

The following classification algorithms were implemented and compared:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Final Selected Model:
Random Forest (based on performance evaluation)

📊 Model Evaluation Metrics

The models were evaluated using:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

These metrics help measure prediction performance and balance between false positives and false negatives.

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

Streamlit

GitHub

🖥️ Streamlit Application Features

Interactive user input form

Real-time churn prediction

Churn probability display

Clean and responsive UI

Model integration using pickle

📁 Project Structure
Customer_Churn_Project/
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
🚀 How to Run the Project Locally

Clone the repository

Install required libraries:

pip install -r requirements.txt

Run Streamlit app:

streamlit run app.py
💡 Business Impact

This system helps organizations:

Identify high-risk customers

Reduce revenue loss

Improve customer retention strategies

Make data-driven decisions

📌 Future Improvements

Add ROC Curve visualization

Deploy using Streamlit Cloud

Add model comparison dashboard

Integrate real-time database support

👩‍💻 Author

Deeksha Jain
MSc Big Data Analytics
