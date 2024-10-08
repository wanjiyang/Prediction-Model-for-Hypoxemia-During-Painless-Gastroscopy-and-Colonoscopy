from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os

app = Flask(__name__)
model = None

def load_model():
    global model
    try:
        # 定义模型路径，请将 'model.joblib' 替换为您的实际模型文件名
        model_path = os.path.join(os.path.dirname(__file__), 'final_model.joblib')
        # 加载模型
        model = load(model_path)
        print("模型已成功加载")
    except Exception as e:
        print(f"模型加载失败: {e}")
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
            'Surgery Type-3', 'Surgery Type-4', 'Smoking', 'Drinking', 'BP', 'Snoring', 'Tired',
            'Observed', 'Inpatient', 'Height', 'Years of Surgical Experience',
            'Cardiovascular Disease-1', 'Other Disease-1.0'
        ]

        # 检查是否有缺失字段
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'缺少必要的字段: {missing_fields}'}), 400

        # 提取并转换输入数据
        try:
            propofol_dosage = float(data['Propofol Dosage'])
            oxygen_flow_rate = float(data['Oxygen Flow Rate'])
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
            smoking = int(data['Smoking'])
            drinking = int(data['Drinking'])
            bp = int(data['BP'])
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
            bp,
            smoking,
            inpatient,
            observed,
            gender,
            cardiovascular_disease,
            tired,
            other_disease,
            oxygen_flow_rate,
            surgery_type_4
        ]])

        # 使用模型进行预测
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # 返回服务器内部错误信息
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # 在应用启动时加载模型
    # 注意：在生产环境中，不应使用 debug=True
    app.run(host='0.0.0.0', port=5000)
