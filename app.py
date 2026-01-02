import streamlit as st
import pandas as pd
import joblib

# Load the model and features
model = joblib.load("titanic_model.pkl")
features = joblib.load("titanic_features.pkl")

# Page configuration
st.set_page_config(page_title="⛴️ Titanic Survival Predictor", layout="centered")

st.title("⛴️ Titanic Survival Prediction App")
st.markdown("""
Enter passenger details to predict the probability of survival 🏊‍♀️.
""")

# --- USER INPUT ---
user_data = {}

# Organize layout in 2 columns
col1, col2 = st.columns(2)

# Passenger Class, Age, SibSp
with col1:
    user_data["Pclass"] = st.slider("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", 1, 3, 1)
    user_data["Age"] = st.slider("Age", 0, 100, 30)
    user_data["SibSp"] = st.slider("Siblings/Spouses aboard", 0, 8, 0)

# Parch, Fare, Sex
with col2:
    user_data["Parch"] = st.slider("Parents/Children aboard", 0, 6, 0)
    user_data["Fare"] = st.slider("Fare ($)", 0, 500, 50)

    # Sex dropdown → convert to dummy variable
    sex_input = st.selectbox("Sex", ["Male", "Female"])
    user_data["Sex_male"] = 1 if sex_input == "Male" else 0

# Embarked dropdown → convert to dummy variables
embarked_input = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
user_data["Embarked_Q"] = 1 if embarked_input == "Q" else 0
user_data["Embarked_S"] = 1 if embarked_input == "S" else 0
# Note: Embarked_C is dropped (drop_first)

# --- PREDICTION ---
if st.button("PREDICT SURVIVAL 🕵️"):
    input_df = pd.DataFrame([user_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Perfect! This passenger is likely to survive ({probability*100:.2f}% probability) 😁💃")
    else:
        st.error(f"❌ Aw, dang! This passenger is unlikely to survive ({(1-probability)*100:.2f}% probability) 😔💔")

    # Optional: small celebration for survivors
    if prediction == 1:
        st.balloons()
