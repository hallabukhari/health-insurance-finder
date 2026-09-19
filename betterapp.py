import joblib
import pandas as pd
import streamlit as st


model = joblib.load("model.pkl")

st.title("Insurance Charges Prediction")
st.write("Enter your information below:")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
sex = st.selectbox("Sex", ["male", "female"])
children = st.number_input("Number of Children", min_value=0, max_value=20)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"],
)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[
            age,
            bmi,
            children,
            age * bmi,
            1 if sex == "male" else 0,
            1 if smoker == "yes" else 0,
            1 if region == "northwest" else 0,
            1 if region == "southeast" else 0,
            1 if region == "southwest" else 0,
        ]],
        columns=model.feature_names_in_,
    )

    prediction = model.predict(input_data)[0]
    st.write("Estimated insurance charges:", round(float(prediction), 2))
