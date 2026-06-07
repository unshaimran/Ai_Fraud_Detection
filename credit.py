import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(page_title="Fraud Detection AI", layout="wide")

model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 AI Fraud Detection System ")

st.sidebar.header("Enter Transaction Details")

# Inputs
transaction_id = st.sidebar.number_input("Transaction ID", 1, 999999, 1)
amount = st.sidebar.number_input("Transaction Amount", 0.0, 100000.0, 1000.0)
transaction_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    ["grocery", "electronics", "fashion", "travel", "others"]
)

foreign_transaction = st.sidebar.selectbox("Foreign Transaction", ["No", "Yes"])
location_mismatch = st.sidebar.selectbox("Location Mismatch", ["No", "Yes"])

device_trust_score = st.sidebar.slider("Device Trust Score", 0.0, 1.0, 0.5)
velocity_last_24h = st.sidebar.number_input("Velocity Last 24h", 0.0, 10.0, 1.0)
cardholder_age = st.sidebar.number_input("Cardholder Age", 18, 100, 30)

# Encoding
merchant_map = {"grocery": 0, "electronics": 1, "fashion": 2, "travel": 3, "others": 4}

merchant_category = merchant_map[merchant_category]
foreign_transaction = 1 if foreign_transaction == "Yes" else 0
location_mismatch = 1 if location_mismatch == "Yes" else 0

# Predict button
if st.sidebar.button("🔍 Detect Fraud"):

    input_data = pd.DataFrame([[
        transaction_id,
        amount,
        transaction_hour,
        merchant_category,
        foreign_transaction,
        location_mismatch,
        device_trust_score,
        velocity_last_24h,
        cardholder_age
    ]], columns=model.feature_names_in_)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("🔎 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGIT TRANSACTION")

    with col2:
        st.metric("Fraud Probability", f"{prob[1]*100:.2f}%")

    # 📊 PIE CHART - Risk Distribution
    chart_data = pd.DataFrame({
        "Type": ["Legit", "Fraud"],
        "Probability": [prob[0]*100, prob[1]*100]
    })

    fig = px.pie(
        chart_data,
        names="Type",
        values="Probability",
        title="Transaction Risk Distribution",
        color_discrete_sequence=["green", "red"]
    )

    st.plotly_chart(fig, use_container_width=True)

# 📊 System Features
st.markdown("---")
st.subheader("📊 System Features")
st.write("""
✔ Real-time Fraud Detection  
✔ Machine Learning Model  
✔ Probability-based Risk Scoring  
✔ Interactive Graphs (Plotly)  
✔ Streamlit Dashboard  
""")