from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])

    return render_template('index.html', prediction_text=f"Predicted Score: {prediction[0]}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
