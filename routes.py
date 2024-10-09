from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import logging

app = Flask(__name__)
model = None  # Global variable to store the model

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_model():
    global model
    try:
        # Load your model here
        model = load('final_model.joblib')
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        # Extract and type convert all input fields
        propofol_dosage = float(data['Propofol_Dosage'])
        oxygen_flow_rate = float(data['Oxygen_Flow_Rate'])
        gender = int(data['Gender'])
        age = int(data['Age'])
        bmi = int(data['BMI'])
        neck_circumference = int(data['Neck_Circumference'])
        stop_bang = int(data['STOP_BANG'])
        asa = int(data['ASA'])
        spo2 = int(data['SpO2'])
        systolic_bp = int(data['Systolic_BP'])
        diastolic_bp = int(data['Diastolic_BP'])
        heart_rate = int(data['Heart_Rate'])
        respiratory_rate = int(data['Respiratory_Rate'])
        surgery_type_2 = int(data['Surgery_Type_2'])
        surgery_type_3 = int(data['Surgery_Type_3'])
        surgery_type_4 = int(data['Surgery_Type_4'])
        smoking = int(data['Smoking'])
        drinking = int(data['Drinking'])
        snoring = int(data['Snoring'])
        tired = int(data['Tired'])
        observed_apnea = int(data['Observed_Apnea'])
        inpatient_status = int(data['Inpatient_Status'])
        height = float(data['Height'])
        years_experience = int(data['Years_of_Surgical_Experience'])
        cardiovascular_disease = int(data['Cardiovascular_Disease'])
        other_disease = int(data['Other_Diseases'])

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

        logging.debug(f"Features for prediction: {features}")

        # Use the model to make a prediction
        prediction = model.predict(features)
        logging.debug(f"Model prediction: {prediction}")

        # 假设模型返回的是数值类别，您可以根据需要进行调整
        # 例如，将预测结果转换为可读的消息或建议
        prediction_message = f"The predicted risk of hypoxemia is: {prediction[0]}"
        suggestions = "Please consult with your anesthesiologist for further management."

        return jsonify({
            'prediction': prediction_message,
            'suggestions': suggestions
        })
    except KeyError as ke:
        error_msg = f"Missing key in input data: {ke}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except ValueError as ve:
        error_msg = f"Invalid value type: {ve}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"An error occurred during prediction: {e}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(debug=True)
