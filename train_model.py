from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

try:
    # Load the model and scaler
    with open("classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    model = None
    scaler = None
    print("❌ Error loading model or scaler:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return "Model or scaler not loaded properly."

    try:
        # Get form values
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"]),
        ]
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
