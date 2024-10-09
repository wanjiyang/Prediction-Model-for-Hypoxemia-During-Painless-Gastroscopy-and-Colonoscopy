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
        logger.info(f"模型路径：{model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error("模型文件不存在！")
            model = None
            return
        
        # 打印模型文件大小
        file_size = os.path.getsize(model_path)
        logger.info(f"模型文件大小：{file_size} bytes")
        
        # 加载模型
        model = load(model_path)
        logger.info("模型已成功加载")
    except Exception as e:
        logger.error("模型加载失败")
        logger.error(traceback.format_exc())
        model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': '模型未正确加载，无法进行预测'}), 500

        # 从请求中获取JSON数据
        data = request.get_json()

        # 必要字段列表
        required_fields = [
            'Propofol Dosage', 'Oxygen Flow Rate', 'Gender', 'Age', 'BMI', 'NC', 'STOP-BANG',
            'ASA', 'SPO2', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'HR', 'RR',
            'Surgery Type-3', 'Surgery Type-4', 'Smoking', 'Drinking', 'Snoring', 'Tired',
            'Observed', 'Inpatient', 'Height', 'Years of Surgical Experience',
            'Cardiovascular Disease-1', 'Other Disease-1.0', 'Surgery Type-2'
        ]

        # 检查是否有缺失字段
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'缺少必要的字段: {missing_fields}'}), 400

        # 提取并转换输入数据
        try:
            propofol_dosage = float(data['Propofol Dosage'])
            oxygen_flow_rate = int(data['Oxygen Flow Rate'])
            gender = int(data['Gender'])
            age = int(data['Age'])
            bmi = float(data['BMI'])
            nc = int(data['NC'])
            stop_bang = int(data['STOP-BANG'])
            asa = int(data['ASA'])
            spo2 = float(data['SPO2'])
            systolic_bp = float(data['Systolic Blood Pressure'])
            diastolic_bp = float(data['Diastolic Blood Pressure'])
            hr = float(data['HR'])
            rr = float(data['RR'])
            surgery_type_3 = int(data['Surgery Type-3'])
            surgery_type_4 = int(data['Surgery Type-4'])
            surgery_type_2 = int(data['Surgery Type-2'])
            smoking = int(data['Smoking'])
            drinking = int(data['Drinking'])
            snoring = int(data['Snoring'])
            tired = int(data['Tired'])
            observed = int(data['Observed'])
            inpatient = int(data['Inpatient'])
            # 新增字段
            height = float(data['Height'])
            years_experience = int(data['Years of Surgical Experience'])
            cardiovascular_disease = int(data['Cardiovascular Disease-1'])
            other_disease = float(data['Other Disease-1.0'])
        except ValueError as ve:
            return jsonify({'error': f'数据类型转换错误: {ve}'}), 400

        # 组合特征为模型输入
        features = np.array([[
            propofol_dosage,
            height,
            years_experience,
            stop_bang,
            bmi,
            nc,
            diastolic_bp,
            spo2,
            systolic_bp,
            age,
            rr,
            hr,
            asa,
            snoring,
            surgery_type_3,
            drinking,
            smoking,
            inpatient,
            observed,
            gender,
            cardiovascular_disease,
            tired,
            other_disease,
            oxygen_flow_rate,
            surgery_type_4,
            surgery_type_2,
        ]])

        # 使用模型进行预测
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error("预测过程中发生错误")
        logger.error(traceback.format_exc())
        # 返回服务器内部错误信息
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # 在应用启动时加载模型
    # 注意：在生产环境中，不应使用 debug=True
    app.run(host='0.0.0.0', port=5000)
