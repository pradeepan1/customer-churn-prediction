# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview
Customer churn is a critical challenge for telecom companies, where customers discontinue services due to dissatisfaction, high costs, or better alternatives.  
This project builds an **end-to-end machine learning system** to predict customer churn and help businesses take preventive actions.

The system includes data preprocessing, model training, evaluation, and an interactive web interface built using Streamlit.

---

## 🎯 Objectives
- Predict whether a customer will churn or stay
- Identify key factors influencing churn
- Provide a user-friendly prediction interface
- Visualize important churn-driving features

---

## 📂 Dataset
- **Dataset:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Target Variable:** `Churn` (Yes / No)

### Key Features
- Demographics: gender, senior citizen
- Services: streaming TV, internet service, tech support, online security
- Billing: monthly charges, total charges, payment method
- Contract details and tenure

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed churn distribution
- Studied churn behavior across contract type, tenure, and monthly charges
- Visualized key trends using graphs

### 2. Data Preprocessing
- Removed irrelevant columns
- Handled missing values
- Converted categorical variables into numerical form

### 3. Feature Encoding
- Encoded all categorical attributes
- Prepared ML-ready dataset

### 4. Model Training
- **Logistic Regression** used as baseline model
- **Random Forest** used for improved performance
- Evaluated using accuracy, precision, recall, and confusion matrix

### 5. Model Performance
- Random Forest Accuracy: ~79%
- Identified top churn influencing features

### 6. Streamlit Web Application
- Human-friendly dropdown inputs
- Real-time churn prediction
- Feature importance visualization for explainability

---

## 📊 Key Insights
- Customers with **month-to-month contracts** churn more
- **Higher monthly charges** increase churn probability
- **Shorter tenure** customers are more likely to churn
- Total charges and contract type are strong churn indicators

---

## 🖥️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install pandas scikit-learn streamlit

2️⃣ Run EDA
python churn_eda.py

3️⃣ Run Data Preprocessing
python churn_data_cleaning.py
python churn_feature_encoding.py

4️⃣ Run ML Models
python churn_model_logistic.py
python churn_model_random_forest.py

5️⃣ Run Streamlit UI
python -m streamlit run app.py

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Matplotlib, Seaborn

Git & GitHub


🚀 Future Enhancements

Deploy application on cloud platform

Hyperparameter tuning for improved accuracy

Add real-time customer data integration

Include cost-based churn reduction strategies

👤 Author

Pradeepan L
BE CSE (AI & ML)
Passionate about Artificial Intelligence and real-world ML applications

⭐ Final Note

This project demonstrates a complete machine learning pipeline with business relevance, explainable predictions, and interactive deployment.


---

## 🚀 STEP 3: COMMIT 7 (FINAL COMMIT)

Run these commands **exactly**:

```bash
git status
git add README.md
git commit -m "Day 7: Project documentation and final submission readiness"
git push