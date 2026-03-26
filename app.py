import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Train model directly (no model.pkl needed)
X = np.array([
    [2, 60, 50],
    [3, 65, 55],
    [4, 70, 60],
    [5, 75, 65],
    [6, 80, 70],
    [7, 85, 75],
    [8, 90, 80]
])

y = np.array([55, 60, 65, 70, 75, 80, 85])

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Predictor")

f1 = st.number_input("Study Hours")
f2 = st.number_input("Attendance (%)")
f3 = st.number_input("Previous Marks")

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    result = model.predict(data)
    st.success(f"Predicted Score: {result[0]}")
