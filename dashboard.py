import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Bank Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Customer Risk Analysis Dashboard")
st.markdown("---")

col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.subheader("📝 Customer Details")
    
    with st.form("churn_prediction_form"):
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: credit_score = st.number_input("Credit Score", 300, 850, value=None)
        with r1_c2: geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=None)
        
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: gender = st.selectbox("Gender", ["Female", "Male"], index=None)
        with r2_c2: age = st.number_input("Age", 18, 100, value=None)
            
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1: tenure = st.number_input("Tenure", 0, 10, value=None)
        with r3_c2: num_products = st.selectbox("Num of Products", [1, 2, 3, 4], index=None)

        r4_c1, r4_c2 = st.columns(2)
        with r4_c1: balance = st.number_input("Balance", 0.0, step=1000.0, value=None)
        with r4_c2: estimated_salary = st.number_input("Estimated Salary", 0.0, step=1000.0, value=None)

        r5_c1, r5_c2 = st.columns(2)
        with r5_c1: card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "DIAMOND", "PLATINUM"], index=None)
        with r5_c2:
            st.write("")
            has_cr_card = st.checkbox("Has Credit Card?", value=False)
            is_active = st.checkbox("Is Active Member?", value=False)
            
        st.markdown("---")
        submit_btn = st.form_submit_button("ANALYZE RISK", type="primary", use_container_width=True)


with col_right:
    tab_result, tab_history = st.tabs(["Analysis Result", "Prediction History"])

    with tab_result:
        result_container = st.container()
        if not submit_btn:
             result_container.info("Enter details on the left and click Analyze.")

    with tab_history:
        if st.button("Refresh Data", use_container_width=True):
            pass
        try:
            hist_resp = requests.get(f"{API_URL}/logs?limit=10")
            if hist_resp.status_code == 200:
                df = pd.DataFrame(hist_resp.json())
                st.dataframe(df, use_container_width=True, hide_index=True, height=400)
            else:
                st.warning("No records found.")
        except:
            st.error("Connection failed.")


if submit_btn:
    missing = []
    if credit_score is None: missing.append("Credit Score")
    if geography is None: missing.append("Geography")
    if gender is None: missing.append("Gender")
    if age is None: missing.append("Age")
    if tenure is None: missing.append("Tenure")
    if num_products is None: missing.append("Products")
    if balance is None: missing.append("Balance")
    if estimated_salary is None: missing.append("Salary")
    if card_type is None: missing.append("Card Type")

    if missing:
        result_container.warning(f"Missing fields: {', '.join(missing)}")
    else:
        result_container.info("Analyzing...")
        
        payload = {
            "CustomerId": 0, "Surname": "Unknown",
            "CreditScore": int(credit_score), 
            "Geography": geography, 
            "Gender": gender,
            "Age": int(age), 
            "Tenure": int(tenure), 
            "Balance": float(balance),
            "NumOfProducts": int(num_products), 
            "HasCrCard": 1 if has_cr_card else 0,
            "IsActiveMember": 1 if is_active else 0, 
            "EstimatedSalary": float(estimated_salary),
            "CardType": card_type
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                res = response.json()
                
                if 'prediction' in res and 'churn_probability' in res:
                    label = res['prediction']
                    prob = res['churn_probability']
                    
                    result_container.empty()
                    
                    if label == "CHURN":
                        result_container.error(f"🚨 **HIGH RISK (CHURN)**")
                        result_container.progress(prob, text=f"Probability: %{prob*100:.2f}")
                        
                    elif label == "LOYAL":
                        result_container.success(f"✅ **LOW RISK (LOYAL)**")
                        result_container.progress(prob, text=f"Probability: %{prob*100:.2f}")
                        
                    else:
                        result_container.error("Model returned an unrecognizable response.")
                else:
                    result_container.error("Invalid response format from API.")
            else:
                result_container.error(f"Server Error: {response.status_code}")
                
        except Exception as e:
            result_container.error(f"Connection Error: {e}")