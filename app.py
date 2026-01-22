from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# --- 1. Load Model & Data saat aplikasi mulai ---
print("Memuat model...")
try:
    svm_model = joblib.load('model_svm_ikn.pkl')
    tfidf_vec = joblib.load('tfidf_vectorizer.pkl')
    df = pd.read_csv('ikn.csv')
except Exception as e:
    print(f"Error memuat file: {e}")
    print("Pastikan file .pkl dan .csv ada di folder yang sama!")

# --- 2. Fungsi Preprocessing (Harus sama persis dengan di Colab) ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_process(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return stemmer.stem(text)

# --- 3. Routes (Halaman Web) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""
    
    # Jika tombol "Cek Sentimen" ditekan (POST request)
    if request.method == 'POST':
        input_text = request.form['comment']
        if input_text:
            # Lakukan Preprocessing & Prediksi
            cleaned_text = clean_process(input_text)
            vectorized_text = tfidf_vec.transform([cleaned_text]).toarray()
            prediction = svm_model.predict(vectorized_text)[0]
            
            # Mapping hasil agar lebih enak dibaca
            prediction_result = "Positif" if prediction == "positive" else "Negatif"

    # Ambil 100 data pertama untuk ditampilkan di tabel
    # Kita convert ke dictionary agar mudah dibaca HTML
    dataset_sample = df.head(50).to_dict(orient='records')
    
    # Hitung Statistik Sederhana untuk ditampilkan
    total_pos = df[df['sentiment'] == 'positive'].shape[0]
    total_neg = df[df['sentiment'] == 'negative'].shape[0]

    return render_template('index.html', 
                           prediction=prediction_result, 
                           original_input=input_text,
                           data=dataset_sample,
                           count_pos=total_pos,
                           count_neg=total_neg)

if __name__ == '__main__':
    app.run(debug=True)