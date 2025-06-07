import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Train model
X = iris.data
y = iris.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose page", ["Home", "Data", "Predict"])

# Page: Home
if page == "Home":
    st.title("🌸 Iris Classifier App")
    st.write("Welcome to the Iris Flower Classifier using Streamlit!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Flower")

# Page: Data
elif page == "Data":
    st.title("📊 Iris Dataset")
    st.dataframe(df)

# Page: Predict
elif page == "Predict":
    st.title("🧠 Predict Iris Species")

    # Input sliders
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.write(f"### 🔍 Prediction: **{iris.target_names[prediction]}**")
    
    # Display probabilities
    st.write("### 📈 Prediction Probabilities:")
    proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.bar_chart(proba_df.T)

    # Display accuracy
    st.write("### 📊 Model Accuracy (on training data):")
    st.info(f"{accuracy:.2%}")
