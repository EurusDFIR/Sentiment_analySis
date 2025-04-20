from flask import Flask, request, render_template, jsonify, session, redirect,url_for
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os 

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

app.secret_key = os.urandom(24)
print("--- Khởi tạo ứng dụng Flask ---") # Print 1

# --- Lấy đường dẫn thư mục ---
basedir = os.path.abspath(os.path.dirname(__file__))
print(f"Thư mục cơ sở (basedir): {basedir}") # Print 2

# --- Tải tài nguyên NLTK ---
print("--- Bắt đầu tải NLTK ---") # Print 3
print("NLTK data path:", nltk.data.path) # Print 4
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("Tải stopwords và lemmatizer thành công!") # Print 5
except LookupError as e:
    print(f"Lỗi LookupError khi tải NLTK: {e}") # Print 6 - Lỗi cụ thể
    print("Hãy chạy lại nltk.download trong console.")
    stop_words = set()
    lemmatizer = None
except Exception as e:
    print(f"Lỗi không xác định khi tải NLTK: {e}") # Print 7 - Lỗi khác
    stop_words = set()
    lemmatizer = None


# --- Hàm tiền xử lý văn bản  ---
def preprocess_text(text):
    if lemmatizer is None:
        print("Cảnh báo: Lemmatizer chưa được tải.")
       
        return text # Trả về text gốc nếu không có lemmatizer

    text = text.lower() # Chuyển về chữ thường
    text = re.sub(r'<[^>]*>', '', text) # Loại bỏ HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Loại bỏ ký tự đặc biệt
    tokens = text.split() # Tách từ
    # Lemmatization và loại bỏ stopword
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens) # Ghép lại thành chuỗi


# --- Tải mô hình và vectorizer đã lưu ---
print("--- Bắt đầu tải Model và Vectorizer ---") # Print 8
model = None # Khởi tạo là None
vectorizer = None # Khởi tạo là None
model_path = os.path.join(basedir, 'sentiment_logreg_model.joblib')
vectorizer_path = os.path.join(basedir, 'tfidf_vectorizer.joblib')
print(f"Đường dẫn model dự kiến: {model_path}") # Print 9
print(f"Đường dẫn vectorizer dự kiến: {vectorizer_path}") # Print 10

try:
    print("Đang thử tải model...") # Print 11
    model = joblib.load(model_path)
    print("Tải model thành công!") # Print 12
    print("Đang thử tải vectorizer...") # Print 13
    vectorizer = joblib.load(vectorizer_path)
    print("Tải vectorizer thành công!") # Print 14
except FileNotFoundError:
    print(f"Lỗi FileNotFoundError: Không tìm thấy file model hoặc vectorizer.") # Print 15
    print(f"Đã kiểm tra: {model_path} và {vectorizer_path}")
    # model và vectorizer vẫn là None
except Exception as e:
    print(f"Lỗi không xác định khi tải model/vectorizer: {e}") # Print 16 - Lỗi khác
    # model và vectorizer vẫn là None

print("--- Hoàn tất tải tài nguyên ---") # Print 17
print(f"Trạng thái model: {'Đã tải' if model is not None else 'Chưa tải (None)'}") # Print 18
print(f"Trạng thái vectorizer: {'Đã tải' if vectorizer is not None else 'Chưa tải (None)'}") # Print 19
print(f"Trạng thái lemmatizer: {'Đã tải' if lemmatizer is not None else 'Chưa tải (None)'}") # Print 20

# --- Route cho trang chủ ---
@app.route('/')
def home():
    print("Truy cập trang chủ ('/')") # Print 21
    prediction_text = session.pop('prediction_text', None) # Lấy và xóa khỏi session
    original_review = session.pop('original_review', None)
    is_error = session.pop('is_error', False)
    print("Truy cập trang chủ ('/')")
    return render_template('index.html',
                           prediction_text=prediction_text,
                           original_review=original_review,
                           error=is_error)

# --- Route xử lý dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    print("Nhận yêu cầu dự đoán ('/predict') [POST]")
    if model is None or vectorizer is None or lemmatizer is None:
         print("Lỗi trong /predict: Model, Vectorizer hoặc Lemmatizer là None.")
         session['prediction_text'] = 'Lỗi: Hệ thống chưa sẵn sàng để dự đoán.'
         session['is_error'] = True
         return redirect(url_for('home'))

    if request.method == 'POST':
        review_text = request.form['review']
        print(f"Nhận được đánh giá: '{review_text[:50]}...'")

        if not review_text.strip():
             print("Lỗi trong /predict: Đánh giá rỗng.")
             session['prediction_text'] = 'Vui lòng nhập đánh giá phim.'
             session['original_review'] = review_text # Vẫn lưu lại để hiển thị
             session['is_error'] = True
             return redirect(url_for('home'))

        try:
            processed_text = preprocess_text(review_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)
            sentiment = "Tích cực 😊" if prediction[0] == 1 else "Tiêu cực 😞"
            print(f"Kết quả dự đoán: {prediction[0]} ({sentiment})")

            # Lưu kết quả và đánh giá gốc vào session
            session['prediction_text'] = f'Dự đoán cảm xúc: {sentiment}'
            session['original_review'] = review_text
            session['is_error'] = False # Không có lỗi

        except Exception as e:
             print(f"Lỗi trong /predict khi xử lý/dự đoán: {e}")
             session['prediction_text'] = 'Lỗi trong quá trình xử lý hoặc dự đoán.'
             session['original_review'] = review_text
             session['is_error'] = True

        # Luôn chuyển hướng về trang chủ sau khi xử lý POST
        return redirect(url_for('home'))

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    pass
    print("app.py được chạy trực tiếp (không phải qua WSGI)")