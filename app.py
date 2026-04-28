import streamlit as st
import numpy as np
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🚀",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("Dsproject1.pkl", "rb"))

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
    }
    .main {
        background: linear-gradient(to right, #1f4037, #99f2c8);
        border-radius: 15px;
        padding: 20px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #ff416c;
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<p class="title">📊 Data Science Predictor</p>', unsafe_allow_html=True)
st.write("### Enter your data below 👇")

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", min_value=0.0, step=0.1)
    feature2 = st.number_input("Feature 2", min_value=0.0, step=0.1)

with col2:
    feature3 = st.number_input("Feature 3", min_value=0.0, step=0.1)
    feature4 = st.number_input("Feature 4", min_value=0.0, step=0.1)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)

        st.success(f"✅ Prediction: {prediction[0]}")

        # 🎉 Animation effect
        st.balloons()

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
