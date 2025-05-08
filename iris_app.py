import streamlit as st
import joblib
import numpy as np
import os

# Load model
model = joblib.load("model_iris.pkl")

# Species mapping with fallback to local image for Setosa
species_map = {
    0: ("Setosa", "local"),  # Use uploaded image for Setosa
    1: ("Versicolor", "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"),
    2: ("Virginica", "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"),
    "setosa": ("Setosa", "local"),
    "versicolor": ("Versicolor", "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"),
    "virginica": ("Virginica", "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg")
}

# Title
st.title("Iris Species Predictor 🌸")
st.markdown("Adjust the measurements below to predict the species of Iris flower.")

# Feature sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]

# Get species info
species_name, image_source = species_map.get(prediction, ("Unknown", None))

# Display prediction
st.subheader("Predicted Species:")
st.success(f"🌼 {species_name}")

# Display image
if image_source == "local":
    st.image("Irissetosa1.jpg", caption=species_name, use_container_width=True)
elif image_source:
    st.image(image_source, caption=species_name, use_container_width=True)
