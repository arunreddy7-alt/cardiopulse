import numpy as np
import joblib

# Load model
model = joblib.load("knn_model.pkl")

# Encoding dictionaries
gender_map = {"Male": 0, "Female": 1}
age_map = {"18-34": 1, "35-50": 2, "51-64": 3, "65+": 4}
yes_no_map = {"No": 0, "Yes": 1}
when_diagnosed_map = {"<1 Year": 1, "1 - 5 Years": 2, ">5 Years": 3}
severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
systolic_bp_map = {"100 - 110": 0, "111 - 120": 1, "121 - 130": 2, "130+": 3}
diastolic_bp_map = {"70 - 80": 0, "81 - 90": 1, "91 - 100": 2, "100+": 3}

stage_map = {0: "NORMAL", 1: "Stage-1", 2: "Stage-2", 3: "CRISIS"}

def predict_patient(data_dict):
    """Predict hypertension stage for a patient"""
    encoded = [
        gender_map[data_dict["Gender"]],
        age_map[data_dict["Age"]],
        yes_no_map[data_dict["History"]],
        yes_no_map[data_dict["Patient"]],
        yes_no_map[data_dict["TakeMedication"]],
        severity_map[data_dict["Severity"]],
        yes_no_map[data_dict["BreathShortness"]],
        yes_no_map[data_dict["VisualChanges"]],
        yes_no_map[data_dict["NoseBleeding"]],
        when_diagnosed_map[data_dict["WhenDiagnosed"]],
        systolic_bp_map[data_dict["Systolic"]],
        diastolic_bp_map[data_dict["Diastolic"]],
        yes_no_map[data_dict["ControlledDiet"]]
    ]
    
    input_array = np.array(encoded).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    
    return {
        "stage": stage_map[prediction],
        "confidence": np.max(probability) * 100,
        "probabilities": {stage_map[i]: prob * 100 for i, prob in enumerate(probability)}
    }

# Example: Test your own case
if __name__ == "__main__":
    test_patient = {
        "Gender": "Male",
        "Age": "35-50",
        "History": "Yes",
        "Patient": "No",
        "TakeMedication": "No",
        "Severity": "Mild",
        "BreathShortness": "No",
        "VisualChanges": "No",
        "NoseBleeding": "No",
        "WhenDiagnosed": "1 - 5 Years",
        "Systolic": "121 - 130",
        "Diastolic": "81 - 90",
        "ControlledDiet": "Yes"
    }
    
    result = predict_patient(test_patient)
    print(f"Prediction: {result['stage']} ({result['confidence']:.2f}% confidence)")
    print(f"\nAll Probabilities:")
    for stage, prob in result['probabilities'].items():
        print(f"  {stage}: {prob:.2f}%")
