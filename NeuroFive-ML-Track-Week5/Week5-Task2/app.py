import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# App Title & Description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict their survival probability using the trained ML pipeline.")

# Load the saved joblib pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# User Input Form
with st.form("passenger_form"):
    st.subheader("Passenger Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2)
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.number_input("Age", min_value=0.42, max_value=80.0, value=22.0, step=1.0)
        embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])
        
    with col2:
        fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=512.33, value=7.25, step=1.0)
        sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=1, step=1)
        parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0, step=1)
    
    submit_button = st.form_submit_button("Predict Survival")

# Inference & Result Display
if submit_button:
    # 1. Feature Engineering
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # 2. Match exact DataFrame schema expected by the pipeline
    raw_sample = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'FamilySize': family_size,
        'IsAlone': is_alone
    }])
    
    # 3. Predict directly using the loaded pipeline
    prediction = pipeline.predict(raw_sample)[0]
    probabilities = pipeline.predict_proba(raw_sample)[0]
    confidence = probabilities[prediction]
    
    # 4. Show Output
    st.divider()
    if prediction == 1:
        st.success(f"### 🎉 Prediction: Survived (1)")
    else:
        st.error(f"### ⚠️ Prediction: Did Not Survive (0)")
        
    st.info(f"**Model Confidence:** {confidence:.2%}")
    import os
import streamlit as st
import pandas as pd
import joblib

# Dynamically construct the path to the folder containing this app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'titanic_pipeline.joblib')

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()
