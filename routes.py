from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = None  # Global variable to store the model

def load_model():
    global model
    # Load your model here
    model = load('final_model.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract and type convert all input fields
        propofol_dosage = float(data['Propofol Dosage'])
        oxygen_flow_rate = float(data['Oxygen Flow Rate'])
        gender = int(data['Gender'])
        age = int(data['Age'])
        bmi = int(data['BMI'])
        neck_circumference = int(data['NC'])
        stop_bang = int(data['STOP-BANG'])
        asa = int(data['ASA'])
        spo2 = int(data['SPO2'])
        systolic_bp = int(data['Systolic Blood Pressure'])
        diastolic_bp = int(data['Diastolic Blood Pressure'])
        heart_rate = int(data['HR'])
        respiratory_rate = int(data['RR'])
        surgery_type_3 = int(data['Surgery Type-3'])
        surgery_type_4 = int(data['Surgery Type-4'])
        surgery_type_2 = int(data['Surgery Type-2'])  # Added this new field
        smoking = int(data['Smoking'])
        drinking = int(data['Drinking'])
        snoring = int(data['Snoring'])
        tired = int(data['Tired'])
        observed_apnea = int(data['Observed'])
        inpatient_status = int(data['Inpatient'])
        height = float(data['Height'])
        years_experience = int(data['Years of Surgical Experience'])
        cardiovascular_disease = int(data['Cardiovascular Disease'])
        other_disease = int(data['Other Diseases'])

        # Combine all features into an array in the order expected by the model
        features = np.array([[
            propofol_dosage,
            oxygen_flow_rate,
            gender,
            age,
            bmi,
            neck_circumference,
            stop_bang,
            asa,
            spo2,
            systolic_bp,
            diastolic_bp,
            heart_rate,
            respiratory_rate,
            surgery_type_2,
            surgery_type_3,
            surgery_type_4,
            smoking,
            drinking,
            snoring,
            tired,
            observed_apnea,
            inpatient_status,
            height,
            years_experience,
            cardiovascular_disease,
            other_disease
        ]])

        # Use the model to make a prediction
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(debug=True)
