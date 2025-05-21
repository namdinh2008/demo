import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = "dubaomucnuocrnn"

# Cấu hình đường dẫn
AVAILABLE_DAYS = [5, 15, 30]  # Các tùy chọn ngày dự báo
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_rnn_model(model_path):
    """
    Tải mô hình RNN và các tham số cần thiết
    """
    model = load_model(f'{model_path}.h5', compile=False)
    scaler = joblib.load(f'{model_path}_scaler.pkl')
    config = joblib.load(f'{model_path}_config.pkl')
    return model, scaler, config

def predict_with_rnn(file_path, model_path):
    """
    Dự báo sử dụng mô hình RNN đã huấn luyện
    """
    # Tải mô hình và tham số
    model, scaler, config = load_rnn_model(model_path)
    
    # Đọc dữ liệu đầu vào
    data = pd.read_csv(file_path, parse_dates=['datetime'])
    input_series = data['q64'].values
    
    # Chuẩn bị dữ liệu đầu vào
    input_reshaped = input_series.reshape(-1, 1)
    input_scaled = scaler.transform(input_reshaped)
    
    # Reshape theo định dạng RNN [samples, time_steps, features]
    past_window = config['past_window']
    num_features = len(config['features'])
    
    # Lấy các điểm dữ liệu gần nhất theo cửa sổ quá khứ
    if len(input_scaled) >= past_window:
        input_rnn = input_scaled[-past_window:].reshape(1, past_window, num_features)
    else:
        # Xử lý trường hợp không đủ dữ liệu
        padding = np.zeros((past_window - len(input_scaled), num_features))
        padded_input = np.vstack([padding, input_scaled])
        input_rnn = padded_input.reshape(1, past_window, num_features)
    
    # Dự báo
    prediction_scaled = model.predict(input_rnn)
    
    # Chuyển đổi kết quả về định dạng gốc
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    return prediction.flatten()

def create_forecast_chart(historical_data, prediction, forecast_days=5):
    """
    Tạo biểu đồ dự báo
    """
    plt.figure(figsize=(12, 6))
    
    # Lấy dữ liệu lịch sử
    hist_values = historical_data['q64'].values  # Thay đổi từ 'luuluongden' sang 'q64'
    
    # Tạo trục thời gian 
    last_date = historical_data['datetime'].iloc[-1]
    if isinstance(last_date, str):
        last_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
        
    # Trục thời gian cho dữ liệu lịch sử (số ngày hiển thị lịch sử bằng số ngày dự báo)
    history_days_to_show = min(forecast_days, len(hist_values))
    hist_dates = pd.date_range(end=last_date, periods=history_days_to_show, freq='D')
    hist_values_to_show = hist_values[-history_days_to_show:]

    # Trục thời gian cho dự báo
    forecast_dates = pd.date_range(start=last_date + timedelta(forecast_days), periods=len(prediction), freq='D')
    print("forecast_dates: ",forecast_dates)
    
    # Vẽ biểu đồ
    plt.plot(hist_dates, hist_values_to_show, 'b-', label='Dữ liệu lịch sử')
    plt.plot(forecast_dates, prediction, 'r-', label=f'Dự báo {forecast_days} ngày')
    
    # Định dạng trục ngày tháng
    plt.gcf().autofmt_xdate()
    
    plt.title(f'Dự báo mực nước {forecast_days} ngày')
    plt.xlabel('Thời gian')
    plt.ylabel('Mực nước (m³/s)')
    plt.legend()
    plt.grid(True)
    
    # Lưu biểu đồ vào buffer để hiển thị trong HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_image = None
    predictions = None
    uploaded_file = None
    selected_days = 5  # Mặc định là 5 ngày
    
    if request.method == 'POST':
        # Kiểm tra nếu có tham số days được truyền vào
        selected_days = int(request.form.get('days', 5))
        
        # Kiểm tra nếu không có file tải lên mới nhưng có file đã tải lên trước đó
        if 'file' not in request.files or request.files['file'].filename == '':
            # Sử dụng file đã tải lên trước đó
            if os.path.exists(os.path.join(UPLOAD_FOLDER, 'data_input.csv')):
                file_path = os.path.join(UPLOAD_FOLDER, 'data_input.csv')
                uploaded_file = "file đã tải lên trước đó"
            else:
                flash('Chưa có file dữ liệu nào được tải lên')
                return render_template('index.html', selected_days=selected_days)
        else:
            # Xử lý file mới tải lên
            file = request.files['file']
            file_path = os.path.join(UPLOAD_FOLDER, 'data_input.csv')
            file.save(file_path)
            uploaded_file = file.filename
        
        try:
            # Đọc dữ liệu
            data = pd.read_csv(file_path, parse_dates=['datetime'])
            
            # Dự báo theo số ngày đã chọn
            model_path = f'RNN_{selected_days}'  # Ví dụ: model_5d, model_15d
            predictions = predict_with_rnn(file_path, model_path)
            
            # Tạo biểu đồ
            chart_image = create_forecast_chart(data, predictions, forecast_days=selected_days)
            
            # Chuyển đổi dự báo thành danh sách có thể hiển thị
            predictions = predictions.tolist()
            
        except Exception as e:
            flash(f'Lỗi khi xử lý dữ liệu: {str(e)}')
            return render_template('index.html', selected_days=selected_days)
    
    return render_template('index2.html',
                        chart_image=chart_image,
                        predictions=predictions,
                        uploaded_file=uploaded_file,
                        selected_days=selected_days)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API để dự báo từ các ứng dụng khác
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
            
    file = request.files['file']
    days = int(request.form.get('days', 5))  # Mặc định là 5 ngày
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400
        
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, 'data_input.csv')
        file.save(file_path)
        
        try:
            model_path = f'model_{days}d'
            predictions = predict_with_rnn(file_path, model_path)
            
            # Đọc dữ liệu để tạo ngày dự báo
            data = pd.read_csv(file_path, parse_dates=['datetime'])
            last_date = data['date'].iloc[-1]
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
                
            # Tạo ngày dự báo
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions), freq='D')
            forecast_dates_str = [date.strftime('%Y-%m-%d') for date in forecast_dates]
            
            return jsonify({
                'forecast_days': days,
                'forecast_dates': forecast_dates_str,
                'predictions': predictions.tolist()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)