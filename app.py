from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os
app = Flask(__name__)
model = None
# 全局加载模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'final_model.joblib')

def load_model():
    global model
    # 在此处加载您的模型
    model = load(model_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从请求中获取JSON数据
        data = request.get_json()

        # 提取所有输入字段，并进行类型转换
        propofol_dosage = float(data['Propofol Dosage'])
        oxygen_flow_rate = float(data['Oxygen Flow Rate'])
        gender = int(data['Gender'])
        age = int(data['Age'])
        bmi = int(data['BMI'])
        nc = int(data['NC'])
        stop_bang = int(data['STOP-BANG'])
        asa = int(data['ASA'])
        spo2 = int(data['SPO2'])
        systolic_bp = int(data['Systolic Blood Pressure'])
        diastolic_bp = int(data['Diastolic Blood Pressure'])
        hr = int(data['HR'])
        rr = int(data['RR'])
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
        other_disease = int(data['Other Disease-1.0'])

        # 按照模型期望的特征顺序，将所有特征组合成一个数组
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # 在应用启动时加载模型
    app.run(debug=True)
