from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os
import logging
import traceback
import hashlib
import requests

app = Flask(__name__)
model = None

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(model_url, model_path):
    """Download the model file from a remote URL."""
    try:
        logger.info(f"Attempting to download the model file from {model_url}...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            logger.info("Model file downloaded successfully.")
            return True
        else:
            logger.error(f"Failed to download the model file. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error("An exception occurred while downloading the model file.")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return False

def load_model():
    global model
    try:
        model_path = 'final_model.joblib'
        logger.info(f"Model path: {model_path}")
        
        # List all files in the current directory
        files = os.listdir(base_dir)
        logger.debug(f"Files in the current directory: {files}")
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.warning("Model file does not exist. Attempting alternative methods to load the model...")
            # Attempt to download the model file from a remote URL
            model_url = 'https://github.com/wanjiyang/Prediction-Model-for-Hypoxemia-During-Painless-Gastroscopy-and-Colonoscopy/blob/main/final_model.joblib'  # Replace with your actual model file URL
            if download_model(model_url, model_path):
                logger.info("Model file downloaded from the remote URL.")
            else:
                logger.error("Unable to obtain the model file. Model loading failed.")
                model = None
                return
        
        # Print model file size
        file_size = os.path.getsize(model_path)
        logger.debug(f"Model file size: {file_size} bytes")
        
        # Calculate MD5 checksum to verify file integrity
        md5_value = calculate_md5(model_path)
        logger.debug(f"MD5 checksum of the model file: {md5_value}")
        
        # Check file permissions
        permissions = oct(os.stat(model_path).st_mode)[-3:]
        logger.debug(f"Model file permissions: {permissions}")
        
        # Attempt to load the model
        try:
            logger.info("Attempting to load the model...")
            model = load(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load the model. Attempting to load the model in compatibility mode.")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {e}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            # Attempt to load the model in compatibility mode
            try:
                import joblib
                logger.info("Attempting to load the model in compatibility mode...")
                with open(model_path, 'rb') as f:
                    model = joblib.load(f, mmap_mode='r')
                logger.info("Model loaded successfully in compatibility mode.")
            except Exception as e:
                logger.error("Failed to load the model in compatibility mode.")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {e}")
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                model = None
    except Exception as e:
        logger.error("An exception occurred during the model loading process.")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logger.error("The model is not loaded properly. Unable to make predictions.")
            return jsonify({'error': 'The model is not loaded properly. Unable to make predictions.'}), 500

        # Get JSON data from the request
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        # List of required fields
        required_fields = [
            'Propofol Dosage', 'Oxygen Flow Rate', 'Gender', 'Age', 'BMI', 'Neck Circumference', 'STOP-BANG',
            'ASA', 'SpO2', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Heart Rate', 'Respiratory Rate',
            'Surgery Type-3', 'Surgery Type-4', 'Surgery Type-2', 'Smoking', 'Drinking', 'Snoring', 'Tired',
            'Observed Apnea', 'Inpatient Status', 'Height', 'Years of Surgical Experience',
            'Cardiovascular Disease', 'Other Diseases'
        ]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # Extract and convert input data
        try:
            propofol_dosage = float(data['Propofol Dosage'])
            oxygen_flow_rate = float(data['Oxygen Flow Rate'])
            gender = int(data['Gender'])
            age = int(data['Age'])
            bmi = float(data['BMI'])
            neck_circumference = int(data['Neck Circumference'])
            stop_bang = int(data['STOP-BANG'])
            asa = int(data['ASA'])
            spo2 = float(data['SpO2'])
            systolic_bp = float(data['Systolic Blood Pressure'])
            diastolic_bp = float(data['Diastolic Blood Pressure'])
            heart_rate = float(data['Heart Rate'])
            respiratory_rate = float(data['Respiratory Rate'])
            surgery_type_3 = int(data['Surgery Type-3'])
            surgery_type_4 = int(data['Surgery Type-4'])
            surgery_type_2 = int(data['Surgery Type-2'])
            smoking = int(data['Smoking'])
            drinking = int(data['Drinking'])
            snoring = int(data['Snoring'])
            tired = int(data['Tired'])
            observed_apnea = int(data['Observed Apnea'])
            inpatient_status = int(data['Inpatient Status'])
            height = float(data['Height'])
            years_experience = int(data['Years of Surgical Experience'])
            cardiovascular_disease = int(data['Cardiovascular Disease'])
            other_diseases = int(data['Other Diseases'])
            logger.debug("Input data successfully extracted and converted.")
        except ValueError as ve:
            logger.error(f"Data type conversion error: {ve}")
            return jsonify({'error': f'Data type conversion error: {ve}'}), 400

        # Combine features for model input
        features = np.array([[
            propofol_dosage,
            height,
            years_experience,
            stop_bang,
            bmi,
            neck_circumference,
            diastolic_bp,
            spo2,
            systolic_bp,
            age,
            respiratory_rate,
            heart_rate,
            asa,
            snoring,
            surgery_type_3,
            drinking,
            smoking,
            inpatient_status,
            observed_apnea,
            gender,
            cardiovascular_disease,
            tired,
            other_diseases,
            oxygen_flow_rate,
            surgery_type_4,
            surgery_type_2,
        ]])
        logger.debug(f"Model input features: {features}")

        # Make prediction using the model
        logger.info("Making prediction...")
        prediction = model.predict(features)
        logger.info(f"Prediction result: {prediction}")
        result = int(prediction[0])

        # Provide professional suggestions based on the prediction result
        if result == 1:
            message = "Based on the provided information, the patient is at a higher risk of experiencing hypoxemia during painless gastroscopy. Enhanced monitoring and preventive measures are recommended."
            suggestions = (
                "Professional Recommendations:\n"
                "- Ensure continuous and comprehensive SpO2 monitoring.\n"
                "- Prepare emergency oxygen therapy equipment and medications.\n"
                "- Consider adjusting the dosage and administration method of sedatives.\n"
                "- Collaborate closely with anesthesiology and respiratory specialists.\n"
                "- Post-procedure, closely monitor the patient to prevent delayed hypoxemia events."
            )
        else:
            message = "The patient is at a lower risk of experiencing hypoxemia during painless gastroscopy. Standard monitoring protocols are recommended."
            suggestions = (
                "Professional Recommendations:\n"
                "- Continue with standard perioperative monitoring.\n"
                "- Ensure patient comfort and safety during the procedure.\n"
                "- Educate the patient to recognize symptoms of hypoxemia and seek medical attention if necessary."
            )

        return jsonify({'prediction': result, 'message': message, 'suggestions': suggestions})
    except Exception as e:
        logger.error("An error occurred during the prediction process.")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(host='0.0.0.0', port=5000)
