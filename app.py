import streamlit as st
import pandas as pd
import joblib

# Load the model and features
model = joblib.load("titanic_model.pkl")
features = joblib.load("titanic_features.pkl")

st.set_page_config(page_title="⛴️Titanic Survival Predictor", layout="centered")

st.title("⛴️Titanic Survival Prediction App")
st.markdown("""
Enter passenger details to predict the probability of survival🏊‍♀️.
""")

# Collect user input
user_data = {}

for feature in features:
    label = feature.replace("_", " ")
    if feature in ['Pclass', 'SibSp', 'Parch']:
        user_data[feature] = st.number_input(label, min_value=0, value=1, step=1)
    else:
        user_data[feature] = st.number_input(label, min_value=0.0, value=0.0)

# Predict button
if st.button("PREDICT SURVIVAL 🕵️"):
    input_df = pd.DataFrame([user_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of survival

    if prediction == 1:
        st.success(f"✅ Perfect! This passenger is likely to survive ({probability*100:.2f}% probability)😁💃")
    else:
        st.error(f"❌ Aw,dang! This passenger is unlikely to survive ({(1-probability)*100:.2f}% probability)😔💔")