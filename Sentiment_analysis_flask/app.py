from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np # Cần numpy

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- Tải mô hình và vectorizer đã lưu ---
try:
    # Đảm bảo tên file khớp với file bạn đã lưu
    model = joblib.load('sentiment_logreg_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Tải mô hình và vectorizer thành công!")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model hoặc vectorizer. Hãy đảm bảo chúng ở cùng thư mục với app.py")
    model = None
    vectorizer = None
except Exception as e:
    print(f"Lỗi khi tải model/vectorizer: {e}")
    model = None
    vectorizer = None

# --- Tải tài nguyên NLTK ---
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("Lỗi: Chưa tải dữ liệu NLTK (stopwords, wordnet).")
    stop_words = set()
    lemmatizer = None

# --- Hàm tiền xử lý văn bản (Phải giống hệt hàm đã dùng khi huấn luyện) ---
def preprocess_text(text):
    if lemmatizer is None:
        print("Cảnh báo: Lemmatizer chưa được tải.")
        # Xử lý nếu không có lemmatizer
        return text # Hoặc trả về lỗi

    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Route cho trang chủ ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route xử lý dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem model và vectorizer đã được tải thành công chưa
    if model is None or vectorizer is None or lemmatizer is None:
        return render_template('index.html', prediction_text='Lỗi: Hệ thống chưa sẵn sàng để dự đoán.', error=True)

    if request.method == 'POST':
        review_text = request.form['review']

        if not review_text.strip():
             return render_template('index.html', prediction_text='Vui lòng nhập đánh giá phim.', error=True, original_review=review_text)

        # 1. Tiền xử lý
        processed_text = preprocess_text(review_text)

        # 2. Vector hóa (Dùng transform, không fit_transform)
        try:
            # Quan trọng: Vectorizer mong đợi một list/iterable chứa các documents
            vectorized_text = vectorizer.transform([processed_text]) # Truyền vào list chứa 1 document
        except Exception as e:
             print(f"Lỗi khi vector hóa: {e}")
             return render_template('index.html', prediction_text='Lỗi xử lý văn bản.', error=True, original_review=review_text)

        # 3. Dự đoán
        try:
            prediction = model.predict(vectorized_text)
            # predict_proba để lấy xác suất (nếu muốn hiển thị độ tin cậy)
            # probability = model.predict_proba(vectorized_text)

            sentiment = "Tích cực 😊" if prediction[0] == 1 else "Tiêu cực 😞"

            # Lấy xác suất của lớp dự đoán (ví dụ)
            # predicted_proba = probability[0][prediction[0]] * 100 # Lấy xác suất của lớp được dự đoán

            return render_template('index.html',
                                   prediction_text=f'Dự đoán cảm xúc: {sentiment}',
                                   # prediction_confidence=f'Độ tin cậy: {predicted_proba:.2f}%', # (Tùy chọn)
                                   original_review=review_text)
        except Exception as e:
             print(f"Lỗi khi dự đoán: {e}")
             return render_template('index.html', prediction_text='Lỗi dự đoán.', error=True, original_review=review_text)

    return render_template('index.html')

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    # Tắt debug khi không cần thiết nữa
    app.run(debug=False, host='0.0.0.0') # host='0.0.0.0' để có thể truy cập từ máy khác trong cùng mạng
    # Hoặc chỉ cần app.run() nếu chạy trên máy cục bộ