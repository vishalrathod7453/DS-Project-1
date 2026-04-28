import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Performance Predictor", page_icon="🎓")

# Custom CSS for animation and styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    .main-container {
        animation: fadeIn 1.5s;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('Dspproject1.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# UI Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🎓 Student Performance Predictor")
st.subheader("Predict your performance category based on daily habits.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    student_id = st.number_input("Student ID", min_value=0, value=1)
    study_hours = st.slider("Study Hours per Day", 0.0, 24.0, 5.0)
    sleep_hours = st.slider("Sleep Hours per Day", 0.0, 24.0, 8.0)

with col2:
    social_media_hours = st.slider("Social Media Hours", 0.0, 24.0, 2.0)
    exam_score = st.number_input("Current Exam Score", min_value=0.0, max_value=100.0, value=75.0)

# Prediction Logic
if st.button("Predict Result 🚀"):
    # Create feature array in the exact order the model expects
    features = pd.DataFrame([[student_id, study_hours, sleep_hours, social_media_hours, exam_score]], 
                            columns=['id', 'study_hours', 'sleep_hours', 'social_media_hours', 'exam_score'])
    
    with st.spinner('Analyzing patterns...'):
        prediction = model.predict(features)
        
    st.success(f"### Predicted Performance Category: {prediction[0]}")

st.markdown('</div>', unsafe_allow_html=True)
