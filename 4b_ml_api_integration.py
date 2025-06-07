import streamlit as st
import requests

st.title("Iris Classifier (via FastAPI)")
st.write("Make predictions using a FastAPI backend")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        result = response.json()
        st.success(f"Prediction: {result['label']}")
        st.write("Probabilities:")
        st.json(result["probabilities"])
    except Exception as e:
        st.error("Failed to connect to prediction API")
        st.text(str(e))
