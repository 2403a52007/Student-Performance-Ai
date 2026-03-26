import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 Student Performance Predictor")

f1 = st.number_input("Study Hours")
f2 = st.number_input("Attendance (%)")
f3 = st.number_input("Previous Marks")

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    result = model.predict(data)
    st.success(f"Predicted Score: {result[0]}")
