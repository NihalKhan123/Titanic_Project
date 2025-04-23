import streamlit as st
import pickle
import numpy as np

# Load your trained model using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the Streamlit app layout
st.title("Model Prediction App")

# Add input fields (for example, assuming your model needs two inputs)
input1 = st.number_input("Input Feature 1", min_value=0.0, max_value=100.0, value=0.0)
input2 = st.number_input("Input Feature 2", min_value=0.0, max_value=100.0, value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare input data as a numpy array (adjust as per your model's input shape)
    inputs = np.array([[input1, input2]])
    
    # Make prediction
    prediction = model.predict(inputs)
    
    # Show result
    st.write(f"The prediction is: {prediction[0]}")
