import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
@st.cache_resource
def load_trained_model():
    model = load_model("Saved_model/convo_lm_keras.h5") 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # Get the input shape from the model
    input_shape = model.input_shape[1:]  # Remove the batch dimension
    print("Input shape:", input_shape)




    return model

model = load_trained_model()

# Title
st.title("Deep Learning Model Prediction (Keras .h5)")

st.write("This app uses your trained Keras model to make predictions.")

# === Example Input Form ===
# ⚠️ Replace with the actual input shape your model expects
st.header("Enter Input Features")

# For example: if your model expects 4 features
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input (adjust shape according to your model)
        input_data = np.array([[f1, f2, f3, f4]])  

        # Run prediction
        prediction = model.predict(input_data)

        # Show results
        st.subheader("Prediction Result")
        st.write(prediction)

    except Exception as e:
        st.error(f"Error: {e}")
