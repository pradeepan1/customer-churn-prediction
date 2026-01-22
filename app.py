import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer will churn and understand key influencing factors.")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("encoded_churn_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------------------
# Train model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Enter Customer Details")

binary_map = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
internet_map = {"DSL": 0, "Fiber optic": 1, "No Internet": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Electronic check": 2,
    "Mailed check": 3,
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1
}

user_input = {}

user_input["gender"] = gender_map[st.sidebar.selectbox("Gender", gender_map.keys())]
user_input["SeniorCitizen"] = binary_map[st.sidebar.selectbox("Senior Citizen", binary_map.keys())]
user_input["Partner"] = binary_map[st.sidebar.selectbox("Has Partner", binary_map.keys())]
user_input["Dependents"] = binary_map[st.sidebar.selectbox("Has Dependents", binary_map.keys())]

user_input["tenure"] = st.sidebar.slider(
    "Tenure (months)",
    int(X["tenure"].min()),
    int(X["tenure"].max()),
    12
)

user_input["PhoneService"] = binary_map[st.sidebar.selectbox("Phone Service", binary_map.keys())]
user_input["MultipleLines"] = st.sidebar.selectbox("Multiple Lines (Encoded)", [0, 1, 2])
user_input["InternetService"] = internet_map[st.sidebar.selectbox("Internet Service", internet_map.keys())]
user_input["OnlineSecurity"] = st.sidebar.selectbox("Online Security (Encoded)", [0, 1, 2])
user_input["OnlineBackup"] = st.sidebar.selectbox("Online Backup (Encoded)", [0, 1, 2])
user_input["DeviceProtection"] = st.sidebar.selectbox("Device Protection (Encoded)", [0, 1, 2])
user_input["TechSupport"] = st.sidebar.selectbox("Tech Support (Encoded)", [0, 1, 2])
user_input["StreamingTV"] = st.sidebar.selectbox("Streaming TV (Encoded)", [0, 1, 2])
user_input["StreamingMovies"] = st.sidebar.selectbox("Streaming Movies (Encoded)", [0, 1, 2])

user_input["Contract"] = contract_map[st.sidebar.selectbox("Contract Type", contract_map.keys())]
user_input["PaperlessBilling"] = binary_map[st.sidebar.selectbox("Paperless Billing", binary_map.keys())]
user_input["PaymentMethod"] = payment_map[st.sidebar.selectbox("Payment Method", payment_map.keys())]

user_input["MonthlyCharges"] = st.sidebar.number_input(
    "Monthly Charges",
    float(X["MonthlyCharges"].min()),
    float(X["MonthlyCharges"].max()),
    float(X["MonthlyCharges"].mean())
)

user_input["TotalCharges"] = st.sidebar.number_input(
    "Total Charges",
    float(X["TotalCharges"].min()),
    float(X["TotalCharges"].max()),
    float(X["TotalCharges"].mean())
)

input_df = pd.DataFrame([user_input])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
        st.write(f"**Churn Probability:** {probability:.2f}")
    else:
        st.success("✅ Customer is likely to STAY")
        st.write(f"**Churn Probability:** {probability:.2f}")

    # ---------------------------
    # Feature Importance Graph
    # ---------------------------
    st.subheader("📊 Key Factors Influencing Churn")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(8)

    st.bar_chart(importance_df.set_index("Feature"))
