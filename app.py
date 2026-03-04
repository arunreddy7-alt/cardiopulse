# in this the complete backend code will be there, after the model selection

from flask import Flask, render_template, request
import numpy as np
import joblib

# -----------------------------------
# Initialize App
# -----------------------------------
app = Flask(__name__, template_folder='frontend')

# -----------------------------------
# Load Model
# -----------------------------------
model = joblib.load("knn_model.pkl")

# -----------------------------------
# Encoding Dictionaries
# -----------------------------------

gender_map = {"Male": 0, "Female": 1}

age_map = {
    "18-34": 1,
    "35-50": 2,
    "51-64": 3,
    "65+": 4
}

yes_no_map = {"No": 0, "Yes": 1}

when_diagnosed_map = {
    "<1 Year": 1,
    "1 - 5 Years": 2,
    ">5 Years": 3
}

severity_map = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

systolic_bp_map = {
    "100 - 110": 0,
    "111 - 120": 1,
    "121 - 130": 2,
    "130+": 3
}

diastolic_bp_map = {
    "70 - 80": 0,
    "81 - 90": 1,
    "91 - 100": 2,
    "100+": 3
}

# -----------------------------------
# Clinical Recommendations
# -----------------------------------

recommendations = {
    0: {
        "stage": "Normal",
        "advice": [
            "Maintain healthy lifestyle",
            "Regular exercise",
            "Balanced diet",
            "Annual BP checkup"
        ]
    },
    1: {
        "stage": "Stage 1 Hypertension",
        "advice": [
            "Reduce salt intake",
            "Increase physical activity",
            "Weight management",
            "Consult physician regularly"
        ]
    },
    2: {
        "stage": "Stage 2 Hypertension",
        "advice": [
            "Medical consultation required",
            "Strict diet monitoring",
            "Medication adherence",
            "Frequent BP monitoring"
        ]
    },
    3: {
        "stage": "Hypertensive Crisis",
        "advice": [
            "Immediate medical attention",
            "Emergency evaluation required",
            "Continuous monitoring",
            "Hospital admission recommended"
        ]
    }
}

# -----------------------------------
# Home Route
# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Prediction Route
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # -------- Get Form Data --------
        gender = request.form["Gender"]
        age = request.form["Age"]
        history = request.form["History"]
        patient = request.form["Patient"]
        medication = request.form["TakeMedication"]
        severity = request.form["Severity"]
        breath = request.form["BreathShortness"]
        visual = request.form["VisualChanges"]
        nose = request.form["NoseBleeding"]
        when_diagnosed = request.form["WhenDiagnosed"]
        systolic = request.form["Systolic"]
        diastolic = request.form["Diastolic"]
        diet = request.form["ControlledDiet"]

        # -------- Encode Inputs --------
        encoded = [
            gender_map[gender],
            age_map[age],
            yes_no_map[history],
            yes_no_map[patient],
            yes_no_map[medication],
            severity_map[severity],
            yes_no_map[breath],
            yes_no_map[visual],
            yes_no_map[nose],
            when_diagnosed_map[when_diagnosed],
            systolic_bp_map[systolic],
            diastolic_bp_map[diastolic],
            yes_no_map[diet]
        ]

        # -------- Convert to Array --------
        input_array = np.array(encoded).reshape(1, -1)

        # -------- Prediction --------
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]

        confidence = round(np.max(probability) * 100, 2)

        result = recommendations[prediction]

        # -------- Send Result --------
        return render_template(
            "index.html",
            prediction=result["stage"],
            confidence=confidence,
            advice=result["advice"]
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


# -----------------------------------
# Run Server
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
