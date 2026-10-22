import streamlit as st, pandas as pd, joblib, sys
sys.path.append("src")
from preprocess import add_features

model = joblib.load("models/model.pkl")
st.title("🏦 Loan Default Risk Predictor")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual income", 1000, 1_000_000, 50000)
    emp = st.number_input("Years employed", 0, 50, 5)
    home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    hist = st.number_input("Credit history (years)", 0, 50, 5)
with col2:
    amount = st.number_input("Loan amount", 500, 50000, 10000)
    rate = st.number_input("Interest rate (%)", 5.0, 25.0, 11.0)
    intent = st.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "VENTURE",
                          "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    grade = st.selectbox("Loan grade", list("ABCDEFG"))
    prev = st.selectbox("Previous default?", ["N", "Y"])

if st.button("Predict risk"):
    row = pd.DataFrame([{
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
        "loan_amnt": amount, "loan_int_rate": rate,
        "loan_percent_income": amount / income,
        "cb_person_default_on_file": prev, "cb_person_cred_hist_length": hist,
    }])
    prob = model.predict_proba(add_features(row))[0, 1]
    st.metric("Default probability", f"{prob:.1%}")
    if prob > 0.5:   st.error("High risk ❌")
    elif prob > 0.25: st.warning("Medium risk ⚠️")
    else:            st.success("Low risk ✅")