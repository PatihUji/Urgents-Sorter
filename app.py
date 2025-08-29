from flask import Flask, render_template, request
import joblib
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# ========== LOAD MODEL DAN VECTORIZER ==========
model = joblib.load("model_rf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ========== PREPROCESSING ==========
stop_words = set(stopwords.words("indonesian"))
stop_words -= {"tidak", "bukan", "belum", "kurang"}
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# ========== RULE-BASED PRIORITAS ==========
def tentukan_prioritas(teks):
    teks_bersih = clean_text(teks)

    very_high_keywords = [
        "cabul", "leceh", "lecehan", "leceh verbal", "kekerasan", "keras", "penganiayaan",
        "bunuh diri", "ancam", "ancaman", "intimidasi", "pukul", "baku hantam",
        "tidak ada satpam", "tanpa keamanan", "curi", "pencurian", "hilang barang",
        "perampokan", "celaka", "kecelakaan", "darurat", "berbahaya", "mengancam jiwa",
        "perkelahian", "senjata", "dibacok", "dibunuh", "diancam", "dilecehkan"
    ]

    high_keywords = [
        "racun", "keracunan", "gas bocor", "bau gas", "jatuh", "terjatuh", "terpeleset", "tergelincir",
        "lampu jatuh", "lampu pecah", "kaca pecah", "meledak", "ledakan",
        "cedera", "terluka", "terkena luka", "tergores", "benda tajam",
        "mati total", "tidak bisa digunakan", "tidak bisa dipakai",
        "sering error", "hang", "kerusakan sistem", "gagal login", "tidak bisa masuk",
        "server down", "jaringan putus", "login gagal", "akses ditolak", "komputer mati"
    ]

    medium_keywords = [
        "roda rusak", "ban rusak", "kelas rusak", "meja rusak", "kursi rusak", "alat rusak",
        "rusak ringan", "kerusakan kecil", "ventilasi buruk", "kipas tidak nyala",
        "ac tidak dingin", "wifi lambat", "lampu redup", "toilet bau", "toilet kotor",
        "kelas panas", "kelas kotor", "suara berisik", "tidak nyaman",
        "tidak ramah", "sikap buruk", "tidak sopan", "pelayanan buruk", "tidak hadir",
        "jarang hadir", "bolos", "tidak merespon", "respon lambat", "cuek", "diabaikan",
        "input salah", "data salah", "kesalahan input", "kebijakan", "prosedur membingungkan",
        "antri", "antrian panjang", "tidak tertib", "tidak update", "data lama",
        "informasi tidak jelas"
    ]

    if any(k in teks_bersih for k in very_high_keywords):
        return "Very High"
    elif any(k in teks_bersih for k in high_keywords):
        return "High"
    elif any(k in teks_bersih for k in medium_keywords):
        return "Medium"
    else:
        return "Low"

# ========== DATA AWAL ==========
pengaduan_list = [
    {'tanggal': '24-07-2025', 'pengaduan': 'Terjadi Kecelakaan saat praktik.', 'prioritas': ''},
    {'tanggal': '23-07-2025', 'pengaduan': 'Proyektor di ruang kelas C-203 mati total.', 'prioritas': ''},
    {'tanggal': '22-07-2025', 'pengaduan': 'Lampu lorong mati di gedung A lantai 3.', 'prioritas': ''},
    {'tanggal': '21-07-2025', 'pengaduan': 'Lantai licin, banyak mahasiswa terjatuh di tangga gedung B.', 'prioritas': ''},
    {'tanggal': '20-07-2025', 'pengaduan': 'Ada bau gas menyengat di ruang praktikum Fisika.', 'prioritas': ''}
]

def prioritas_sort_key(item):
    urutan = {'Very High': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Unclassified': 4, '': 5}
    return urutan.get(item['prioritas'], 6)

@app.route('/', methods=['GET', 'POST'])
def index():
    global pengaduan_list

    if request.method == 'POST':
        if 'submit_pengaduan' in request.form:
            pengaduan = request.form['pengaduan']
            tanggal_input = request.form['tanggal']
            try:
                tanggal_obj = datetime.strptime(tanggal_input, "%Y-%m-%d")
                tanggal = tanggal_obj.strftime("%d-%m-%Y")
            except ValueError:
                tanggal = datetime.now().strftime("%d-%m-%Y")

            pengaduan_list.append({
                'pengaduan': pengaduan,
                'tanggal': tanggal,
                'prioritas': 'Unclassified'
            })

        elif 'cek_prioritas' in request.form:
            for item in pengaduan_list:
                teks_awal = item['pengaduan']
                teks_bersih = clean_text(teks_awal)
                hasil_ml = model.predict(vectorizer.transform([teks_bersih]))[0]
                hasil_rule = tentukan_prioritas(teks_awal)

                if hasil_rule == hasil_ml:
                    final = hasil_rule
                else:
                    final = hasil_rule if hasil_rule in ["Very High", "High", "Medium"] else hasil_ml

                item['prioritas'] = final

            pengaduan_list.sort(key=prioritas_sort_key)

    def sort_by_tanggal(item):
        try:
            return datetime.strptime(item['tanggal'], "%d-%m-%Y")
        except ValueError:
            return datetime.min

    pengaduan_list_asli = sorted(pengaduan_list, key=sort_by_tanggal)

    return render_template(
        'index.html',
        pengaduan_list_asli=pengaduan_list_asli,
        pengaduan_list=pengaduan_list
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
