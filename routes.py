from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import logging

app = Flask(__name__)
model =load('final_model.joblib')  # 全局变量存储模型

# 配置日志
logging.basicConfig(level=logging.DEBUG)

def load_model():
    global model
    try:
        # 加载您的模型
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

        # 确保所有需要的键都存在
        required_keys = [
            'Propofol Dosage', 'Height', 'Years of Surgical Experience',
            'STOP-BANG', 'NC', 'BMI', 'Diastolic Blood Pressure', 'SPO2',
            'Systolic Blood Pressure', 'Age', 'HR', 'RR', 'ASA', 'Snoring',
            'Surgery Type-3', 'Drinking', 'Gender', 'Smoking', 'Observed',
            'Inpatient', 'Cardiovascular Disease-1', 'Tired',
            'Other Disease-1.0', 'Oxygen Flow Rate', 'Surgery Type-4',
            'Surgery Type-2'
        ]

        missing_keys = [key for key in required_keys if key not in data or data[key] == ""]
        if missing_keys:
            error_msg = f"Missing keys in input data: {', '.join(missing_keys)}"
            logging.error(error_msg)
            return jsonify({'error': error_msg}), 400

        # 提取并转换所有输入字段
        try:
            propofol_dosage = float(data['Propofol Dosage'])
            height = float(data['Height'])
            years_experience = int(data['Years of Surgical Experience'])
            stop_bang = int(data['STOP-BANG'])
            nc = int(data['NC'])
            bmi = int(data['BMI'])
            diastolic_bp = int(data['Diastolic Blood Pressure'])
            spo2 = int(data['SPO2'])
            systolic_bp = int(data['Systolic Blood Pressure'])
            age = int(data['Age'])
            hr = int(data['HR'])
            rr = int(data['RR'])
            asa = int(data['ASA'])
            snoring = int(data['Snoring'])
            surgery_type_3 = int(data['Surgery Type-3'])
            drinking = int(data['Drinking'])
            gender = int(data['Gender'])
            smoking = int(data['Smoking'])
            observed = int(data['Observed'])
            inpatient = int(data['Inpatient'])
            cardiovascular_disease_1 = int(data['Cardiovascular Disease-1'])
            tired = int(data['Tired'])
            other_disease_1_0 = int(float(data['Other Disease-1.0']))  # 转换为整数
            oxygen_flow_rate = float(data['Oxygen Flow Rate'])
            surgery_type_4 = int(data['Surgery Type-4'])
            surgery_type_2 = int(data['Surgery Type-2'])
        except ValueError as ve:
            error_msg = f"Invalid value type: {ve}"
            logging.error(error_msg)
            return jsonify({'error': error_msg}), 400

        # 组合所有特征到数组中，按模型预期的顺序
        features = np.array([[
            propofol_dosage,
            height,
            years_experience,
            stop_bang,
            nc,
            bmi,
            diastolic_bp,
            spo2,
            systolic_bp,
            age,
            hr,
            rr,
            asa,
            snoring,
            surgery_type_3,
            drinking,
            gender,
            smoking,
            observed,
            inpatient,
            cardiovascular_disease_1,
            tired,
            other_disease_1_0,
            oxygen_flow_rate,
            surgery_type_4,
            surgery_type_2
        ]])

        logging.debug(f"Features for prediction: {features}")

        # 使用模型进行预测
        prediction = model.predict(features)
        logging.debug(f"Model prediction: {prediction}")

        # 根据模型的预测结果生成消息和建议
        # 假设模型返回的是类别编号，如 0 或 1
        if prediction[0] == 1:
            prediction_message = "High risk of hypoxemia."
            suggestions = (
            "1. **Pre-oxygenation**: Before anesthesia induction, ensure sufficient oxygenation to provide the patient with adequate oxygen reserves. If possible, directly use advanced ventilation equipment, such as nasal high-flow oxygen therapy.\n"
            "2. **Adjust Anesthesia Plan**: Choose anesthetic drugs that have minimal respiratory suppression, control the dosage, and carefully select the anesthesia plan.\n"
            "3. **Enhance Monitoring**: Utilize continuous pulse oximetry monitoring while simultaneously monitoring blood pressure, heart rate, and respiration. Ensure that trained and experienced anesthesiologists closely supervise the patient.\n"
            "4. **Prepare for Emergencies**: Equip necessary respiratory support devices, such as airway management equipment, ventilators, and simple resuscitation bags."
        )
        else:
            prediction_message = "Low risk of hypoxemia."
            suggestions = (
            "1. **Pre-oxygenation**: Perform routine oxygenation before induction.\n"
            "2. **Anesthesia Plan**: Execute the standard anesthesia plan.\n"
            "3. **Standard Monitoring**: Continue real-time monitoring of blood oxygen, blood pressure, heart rate, etc.\n"
            "4. **Stay Vigilant**: Experienced anesthesiologists should monitor the patient's respiration and blood oxygen levels.\n"
            "5. **Equipment Preparation**: Prepare standard equipment."
        )

        return jsonify({
            'prediction': prediction_message,
            'suggestions': suggestions
        })
    except Exception as e:
        error_msg = f"An error occurred during prediction: {e}"
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    load_model()  # 应用启动时加载模型
    app.run(debug=True)
