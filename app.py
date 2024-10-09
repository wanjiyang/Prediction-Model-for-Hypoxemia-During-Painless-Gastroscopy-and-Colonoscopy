from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os
import logging
import traceback

app = Flask(__name__)
model = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    global model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'final_model.joblib')
        logger.info(f"Model path: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error("Model file does not exist!")
            model = None
            return
        
        # 打印模型文件大小
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size} bytes")
        
        # 加载模型
        model = load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model")
        logger.error(traceback.format_exc())
        model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'The model is not loaded properly, unable to make predictions'}), 500

        # 从请求中获取JSON数据
        data = request.get_json()

        # 必要字段列表
        required_fields = [
            'Propofol Dosage', 'Oxygen Flow Rate', 'Gender', 'Age', 'BMI', 'Neck Circumference', 'STOP-BANG',
            'ASA', 'SpO2', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Heart Rate', 'Respiratory Rate',
            'Surgery Type-3', 'Surgery Type-4', 'Surgery Type-2', 'Smoking', 'Drinking', 'Snoring', 'Tired',
            'Observed Apnea', 'Inpatient Status', 'Height', 'Years of Surgical Experience',
            'Cardiovascular Disease', 'Other Diseases'
        ]

        # 检查是否有缺失字段
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # 提取并转换输入数据
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
        except ValueError as ve:
            return jsonify({'error': f'Data type conversion error: {ve}'}), 400

        # 组合特征为模型输入
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

        # 使用模型进行预测
        prediction = model.predict(features)
        result = int(prediction[0])

        # 根据预测结果提供专业建议
        if result == 1:
            message = "Based on the provided information, the patient is at a higher risk of experiencing hypoxemia during painless gastroscopy. It is recommended to take enhanced monitoring and precautionary measures."
            suggestions = (
                "Professional Recommendations:\n"
                "- Ensure continuous and comprehensive oxygen saturation monitoring.\n"
                "- Prepare emergency oxygen therapy equipment and medications.\n"
                "- Consider adjusting sedation dosage and techniques.\n"
                "- Collaborate closely with anesthesiology and respiratory specialists.\n"
                "- Post-procedure, monitor the patient closely for any delayed hypoxemia events."
            )
        else:
            message = "The patient has a lower risk of hypoxemia during painless gastroscopy. Standard monitoring protocols are recommended."
            suggestions = (
                "Professional Recommendations:\n"
                "- Continue with standard perioperative monitoring.\n"
                "- Ensure patient comfort and safety during the procedure.\n"
                "- Educate the patient on signs of hypoxemia and when to seek medical attention."
            )

        return jsonify({'prediction': result, 'message': message, 'suggestions': suggestions})
    except Exception as e:
        logger.error("An error occurred during prediction")
        logger.error(traceback.format_exc())
        # 返回服务器内部错误信息
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # 在应用启动时加载模型
    # 注意：在生产环境中，不应使用 debug=True
    app.run(host='0.0.0.0', port=5000)
