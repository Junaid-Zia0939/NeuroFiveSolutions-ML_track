import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Train and cache the end-to-end pipeline directly (Zero serialization issues)
@st.cache_resource
def get_trained_pipeline():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    target = 'Survived'
    
    X = df[features]
    y = df[target]
    
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'IsAlone']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X, y)
    return pipeline

pipeline = get_trained_pipeline()

# App Title & Description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict their survival probability using the trained ML pipeline.")

# User Input Form
with st.form("passenger_form"):
    st.subheader("Passenger Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2)
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.number_input("Age", min_value=0.42, max_value=80.0, value=22.0, step=1.0)
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
        )
        
    with col2:
        fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=512.33, value=7.25, step=1.0)
        sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=1, step=1)
        parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0, step=1)
    
    submit_button = st.form_submit_button("Predict Survival")

# Inference & Result Display
if submit_button:
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
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
    
    prediction = pipeline.predict(raw_sample)[0]
    probabilities = pipeline.predict_proba(raw_sample)[0]
    confidence = probabilities[prediction]
    
    st.divider()
    if prediction == 1:
        st.success("### 🎉 Prediction: Survived (1)")
    else:
        st.error("### ⚠️ Prediction: Did Not Survive (0)")
        
    st.info(f"**Model Confidence:** {confidence:.2%}")
