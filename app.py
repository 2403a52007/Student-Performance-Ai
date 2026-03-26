import gradio as gr
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

def predict(f1, f2, f3):
    data = np.array([[f1, f2, f3]])
    result = model.predict(data)
    return f"Predicted Score: {result[0]}"

interface = gr.Interface(
    fn=predict,
    inputs=["number","number","number"],
    outputs="text",
    title="Student Performance Predictor"
)

interface.launch()
