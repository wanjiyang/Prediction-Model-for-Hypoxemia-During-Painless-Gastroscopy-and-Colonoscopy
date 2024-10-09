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

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_md5(file_path):
    """计算文件的 MD5 值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(model_url, model_path):
    """从远程下载模型文件"""
    try:
        logger.info(f"尝试从 {model_url} 下载模型文件...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            logger.info("模型文件已成功下载")
            return True
        else:
            logger.error(f"无法从远程下载模型文件，状态码：{response.status_code}")
            return False
    except Exception as e:
        logger.error("下载模型文件时发生异常")
        logger.error(f"错误类型：{type(e).__name__}")
        logger.error(f"错误详细信息：{e}")
        logger.error("完整的异常堆栈信息如下：")
        logger.error(traceback.format_exc())
        return False

@app.route('/list-files', methods=['GET'])
def list_files():
    """测试路由：列出当前目录的文件"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(base_dir)
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"列出文件时发生错误：{e}")
        return jsonify({'error': '无法列出文件'}), 500

def load_model():
    global model
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = 'final_model.joblib'
        model_path = os.path.join(base_dir, model_filename)
        logger.info(f"模型路径：{model_path}")

        # 列出当前目录的所有文件
        files = os.listdir(base_dir)
        logger.info(f"当前目录下的文件：{files}")

        # 检查模型文件是否存在
        if model_filename in files:
            logger.info("模型文件存在，开始加载模型...")
            # 打印模型文件大小
            file_size = os.path.getsize(model_path)
            logger.info(f"模型文件大小：{file_size} bytes")

            # 计算模型文件的 MD5 值，验证文件完整性
            md5_value = calculate_md5(model_path)
            logger.info(f"模型文件的 MD5 值：{md5_value}")

            # 检查文件权限
            permissions = oct(os.stat(model_path).st_mode)[-3:]
            logger.info(f"模型文件权限：{permissions}")

            # 尝试加载模型
            try:
                model = load(model_path)
                logger.info("模型已成功加载")
            except Exception as e:
                logger.error("模型加载失败，尝试使用兼容模式加载模型")
                logger.error(f"错误类型：{type(e).__name__}")
                logger.error(f"错误详细信息：{e}")
                logger.error("完整的异常堆栈信息如下：")
                logger.error(traceback.format_exc())
                # 尝试使用兼容模式加载模型
                try:
                    import joblib
                    logger.info("尝试使用兼容模式加载模型...")
                    with open(model_path, 'rb') as f:
                        model = joblib.load(f, mmap_mode='r')
                    logger.info("模型已成功加载（兼容模式）")
                except Exception as e:
                    logger.error("兼容模式加载模型失败")
                    logger.error(f"错误类型：{type(e).__name__}")
                    logger.error(f"错误详细信息：{e}")
                    logger.error("完整的异常堆栈信息如下：")
                    logger.error(traceback.format_exc())
                    model = None
        else:
            logger.error("模型文件不存在")
            # 如果模型文件不存在，尝试从远程下载
            model_url = 'https://your-model-url.com/final_model.joblib'  # 请替换为您的模型文件下载链接
            if download_model(model_url, model_path):
                logger.info("模型文件已从远程下载，重新尝试加载模型...")
                load_model()  # 递归调用以加载模型
            else:
                logger.error("无法获取模型文件，模型加载失败")
                model = None
    except Exception as e:
        logger.error("加载模型过程中发生异常")
        logger.error(f"错误类型：{type(e).__name__}")
        logger.error(f"错误详细信息：{e}")
        logger.error("完整的异常堆栈信息如下：")
        logger.error(traceback.format_exc())
        model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logger.error("模型未正确加载，无法进行预测")
            return jsonify({'error': '模型未正确加载，无法进行预测'}), 500

        # 从请求中获取JSON数据
        data = request.get_json()
        logger.debug(f"接收到的数据：{data}")

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
            logger.error(f"缺少必要的字段: {missing_fields}")
            return jsonify({'error': f'缺少必要的字段: {missing_fields}'}), 400

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
            logger.debug("输入数据已成功提取和转换")
        except ValueError as ve:
            logger.error(f"数据类型转换错误: {ve}")
            return jsonify({'error': f'数据类型转换错误: {ve}'}), 400

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
        logger.debug(f"模型输入特征：{features}")

        # 使用模型进行预测
        logger.info("开始进行预测...")
        prediction = model.predict(features)
        logger.info(f"预测结果：{prediction}")
        result = int(prediction[0])

        # 根据预测结果提供专业建议
        if result == 1:
            message = "根据提供的信息，患者在无痛胃镜过程中发生低氧血症的风险较高。建议采取增强的监测和预防措施。"
            suggestions = (
                "专业建议：\n"
                "- 确保持续和全面的血氧饱和度监测。\n"
                "- 准备紧急氧疗设备和药物。\n"
                "- 考虑调整镇静剂的剂量和使用方法。\n"
                "- 与麻醉科和呼吸科专家密切合作。\n"
                "- 术后密切观察患者，防止延迟性低氧事件的发生。"
            )
        else:
            message = "患者在无痛胃镜过程中发生低氧血症的风险较低。建议遵循标准的监测协议。"
            suggestions = (
                "专业建议：\n"
                "- 继续执行标准的围术期监测。\n"
                "- 确保患者在手术过程中的舒适和安全。\n"
                "- 教育患者识别低氧血症的症状，并在需要时寻求医疗帮助。"
            )

        return jsonify({'prediction': result, 'message': message, 'suggestions': suggestions})
    except Exception as e:
        logger.error("预测过程中发生错误")
        logger.error(f"错误类型：{type(e).__name__}")
        logger.error(f"错误详细信息：{e}")
        logger.error("完整的异常堆栈信息如下：")
        logger.error(traceback.format_exc())
        return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    load_model()  # 在应用启动时加载模型
    app.run(host='0.0.0.0', port=5000)
