import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import speech_recognition as sr
from speech_recognition import Recognizer, AudioFile
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io import wavfile
import time
import queue
import threading

# Sayfa yapılandırması
st.set_page_config(page_title="Ses Analiz Sistemi", page_icon="🎤", layout="wide")

# CSS stilleri
st.markdown("""
    <style>
    /* Ana container stili */
    .main {
        padding: 0rem 1rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }

    /* Buton stilleri */
    .stButton>button {
        width: 100%;
        padding: 0.8rem;
        border-radius: 10px;
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(45deg, #27ae60, #2ecc71);
    }

    /* Kart stilleri */
    .metric-card {
        background: linear-gradient(145deg, #2d2d2d, #1f1f1f);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #333;
        margin-bottom: 1rem;
    }

    /* Grafik container */
    .chart-container {
        background: linear-gradient(145deg, #2d2d2d, #1f1f1f);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #333;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }

    .chart-container:hover {
        transform: translateY(-2px);
    }

    /* Başlık stilleri */
    h1 {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }

    h3 {
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333;
    }

    /* Progress bar stilleri */
    .stProgress > div > div {
        background-color: #2ecc71;
        height: 10px;
        border-radius: 5px;
    }

    .stProgress > div {
        background-color: #333;
        border-radius: 5px;
    }

    /* Text area stili */
    .stTextArea textarea {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
    }

    /* Metrik değerleri */
    .css-1wivap2 {
        color: #2ecc71;
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* Metrik etiketleri */
    .css-1wivap2 + div {
        color: #ffffff;
        font-size: 1rem;
    }

    /* Emoji boyutları */
    .emotion-emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }

    /* Duygu metni */
    .emotion-text {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }

    /* Expander stilleri */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        border-radius: 10px;
        color: #ffffff;
    }

    /* Scrollbar stilleri */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #444;
    }
    </style>
""", unsafe_allow_html=True)
class EmotionAnalyzer:
    def __init__(self):
        # Duygu kategorileri ve ilgili kelimeler
        self.emotion_keywords = {
            "Mutlu": [
                "mutlu", "sevinçli", "neşeli", "harika", "güzel", "muhteşem", "süper", 
                "keyifli", "memnun", "pozitif", "coşkulu", "şen", "güleryüzlü", "huzurlu", 
                "umut dolu", "heyecanlı", "sevindirici", "tatmin olmuş", "hayran", "minnettar", 
                "şükür dolu", "iyi", "sevecen", "hoşnut", "mutluluk dolu", "canlı", "aydınlık"
            ],
            "Üzgün": [
                "üzgün", "mutsuz", "kötü", "kederli", "acı", "ağlamak", "hüzünlü", "çaresiz", 
                "umutsuz", "yalnız", "kırgın", "içli", "melankolik", "acı verici", "yıkılmış", 
                "hayal kırıklığına uğramış", "pişman", "kasvetli", "bunalımlı", "sıkıntılı"
            ],
            "Kızgın": [
                "kızgın", "sinirli", "öfkeli", "rahatsız", "bıkmış", "sıkılmış", "gergin", 
                "agresif", "hırçın", "öfke dolu", "sabırsız", "tepkili", "hiddetli", "çileden çıkmış", 
                "düşmanlık", "sert", "gücenmiş", "kavgacı", "bağıran", "öfkesini dışa vuran"
            ],
            "Heyecanlı": [
                "heyecanlı", "heyecan", "coşkulu", "enerjik", "istekli", "hevesli", "meraklı", 
                "sabırsız", "umutlu", "beklenti dolu", "hareketli", "heyecan verici", "dinamik", 
                "mutlu heyecanlı", "canlı", "çevik", "adrenalin dolu", "coşkulanmış", "şevkli"
            ],
            "Endişeli": [
                "endişeli", "kaygılı", "tedirgin", "korkmuş", "stresli", "gergin", "şüpheli", 
                "huzursuz", "panik", "ürkek", "sıkıntılı", "güvensiz", "korku dolu", "şüphe içinde", 
                "kötü beklenti", "ürkeklik", "tereddütlü", "huzursuzluk", "çaresizlik hissi"
            ],
            "Sakin": [
                "sakin", "rahat", "huzurlu", "dingin", "sessiz", "barışçıl", "dengeli", 
                "soğukkanlı", "tatmin olmuş", "gevşemiş", "dinlenmiş", "huzur dolu", "dinginlik", 
                "yavaş", "kontrollü", "uyumlu", "nazik", "su serpilmiş gibi", "yumuşak"
            ],
            "Şaşkın": [
                "şaşkın", "şaşırmış", "hayret", "beklenmedik", "inanılmaz", "garip", "tuhaf", 
                "şaşırtıcı", "afallamış", "şok", "düşünceli", "şaşkınlık içinde", "hayal edilemez", 
                "şaşırtıcı derecede", "şaşkın hissetmek", "şaşırmış bakmak", "şüphe içinde", 
                "şaşkın duygular", "inanılmaz derecede"
            ],
            "Nötr": [
                "tarafsız", "belirsiz", "ortalama", "düz", "sıradan", "nötr", "ilgisiz", 
                "kararsız", "tepkisiz", "duygusuz", "yansız", "durağan", "renksiz", 
                "hissiz", "düz bir şekilde", "duygu katılmamış", "isteksiz", "huzurlu ama etkisiz"
            ]
        }



    def analyze_emotion(self, text):
        if not text:
            return {"duygular": {"Nötr": 100}}

        words = text.lower().split()
        emotion_scores = {emotion: 0 for emotion in self.emotion_keywords}
        total_matches = 0

        # Her kelime için duygu skorlarını hesapla
        for word in words:
            for emotion, keywords in self.emotion_keywords.items():
                if emotion != "Nötr" and any(keyword in word for keyword in keywords):
                    emotion_scores[emotion] += 1
                    total_matches += 1

        # Hiç eşleşme yoksa Nötr
        if total_matches == 0:
            return {"duygular": {"Nötr": 100}}

        # Yüzdelik skorlara dönüştür
        for emotion in emotion_scores:
            if emotion != "Nötr":
                emotion_scores[emotion] = (emotion_scores[emotion] / total_matches) * 100 if total_matches > 0 else 0

        # Eğer hiçbir duygu belirgin değilse (düşük skorlar) Nötr olarak işaretle
        max_score = max(emotion_scores.values())
        if max_score < 10:  # Eşik değeri
            return {"duygular": {"Nötr": 100}}

        # Nötr'ü sıfırla (başka duygular varsa)
        emotion_scores["Nötr"] = 0

        return {"duygular": emotion_scores}

class TopicAnalyzer:
    def __init__(self):
        self.topics = {
            "Eğitim 📚": [
                "okul", "ders", "öğrenci", "öğretmen", "sınav", "ödev", "kitap", "not", "başarı",
                "eğitim", "öğrenmek", "üniversite", "sınıf", "akademik", "araştırma", "bilim",
                "matematik", "fizik", "kimya", "biyoloji", "tarih", "edebiyat", "tez", "makale",
                "çalıştay", "konferans", "okuma", "kütüphane", "dershane", "çalışma"
            ],
            "Sağlık 🏥": [
                "hastane", "doktor", "ilaç", "sağlık", "hastalık", "tedavi", "muayene", "ağrı",
                "kontrol", "diş", "grip", "ateş", "vitamin", "spor", "beslenme", "diyet",
                "egzersiz", "sağlıklı", "hasta", "iyi", "kötü", "psikoloji", "terapi",
                "şifa", "rehabilitasyon", "ameliyat", "check-up", "fitness", "meditasyon", "fizyoterapi"
            ],
            "Teknoloji 💻": [
                "bilgisayar", "telefon", "internet", "uygulama", "yazılım", "teknoloji", "sistem",
                "program", "web", "site", "sosyal medya", "oyun", "veri", "kod", "yapay zeka",
                "robot", "dijital", "online", "elektronik", "donanım", "algoritma", "geliştirici",
                "veritabanı", "siber güvenlik", "sunucu", "blockchain", "bulut", "drone",
                "akıllı saat", "otomasyon", "kripto", "nesnelerin interneti", "teknolojik ürünler"
            ],
            "İş ve Kariyer 💼": [
                "iş", "çalışma", "toplantı", "proje", "müşteri", "şirket", "ofis", "yönetici",
                "maaş", "kariyer", "meslek", "başvuru", "görüşme", "deneyim", "uzman", "personel",
                "ekip", "takım", "lider", "performans", "strateji", "networking", "fırsat",
                "freelance", "girişimcilik", "pazarlama", "satış", "iş planı", "hedefler", "bütçe"
            ],
            "Günlük Yaşam 🏠": [
                "ev", "yemek", "uyku", "alışveriş", "market", "temizlik", "giyim", "aile",
                "arkadaş", "komşu", "tatil", "seyahat", "hobi", "eğlence", "sinema", "müzik",
                "spor", "park", "bahçe", "cafe", "restoran", "alışveriş merkezi", "trafik",
                "evcil hayvan", "bahçe işleri", "tarif", "dizi", "kitap", "kahve", "piknik",
                "kış aktiviteleri", "yaz tatili", "etkinlik", "organize"
            ],
            "Duygusal 💭": [
                "mutlu", "üzgün", "kızgın", "sevinçli", "heyecanlı", "endişeli", "stresli",
                "rahat", "huzurlu", "sevgi", "aşk", "özlem", "umut", "korku", "kaygı",
                "merak", "şaşkınlık", "gurur", "hüzün", "pişmanlık", "şükran", "hayal kırıklığı",
                "keyif", "tatmin", "özveri", "empati", "sevinç", "melankoli", "mutsuzluk"
            ],
            "Sosyal İlişkiler 👥": [
                "arkadaş", "aile", "anne", "baba", "kardeş", "akraba", "dost", "sevgili",
                "komşu", "tanıdık", "ilişki", "sohbet", "buluşma", "davet", "parti",
                "kutlama", "hediye", "misafir", "iletişim", "konuşma", "destek", "dayanışma",
                "paylaşım", "topluluk", "bağlantı", "anlayış", "dostluk", "yakınlık", "tanışma"
            ],
            "Finans 💰": [
                "para", "yatırım", "banka", "kredi", "borç", "bütçe", "tasarruf", "faiz",
                "borsa", "kripto", "bitcoin", "ödeme", "maaş", "kazanç", "gelir", "harcama",
                "hesap", "vergi", "finansman", "döviz", "ekonomi", "işletme", "kâr",
                "sigorta", "emeklilik", "sermaye", "borçlanma", "portföy", "varlık", "gelir"
            ],
            "Sanat 🎨": [
                "resim", "müzik", "tiyatro", "film", "sinema", "heykel", "şiir", "roman",
                "sanat", "yaratıcı", "sergi", "galeri", "yazı", "şarkı", "melodi", "dans",
                "kültür", "gösteri", "kitap", "yazar", "besteci", "aktör", "aktris",
                "eleştiri", "sanatçı", "performans", "festival", "klasik", "modern",
                "müzisyen", "drama", "opera"
            ],
            "Spor ⚽": [
                "futbol", "basketbol", "voleybol", "tenis", "yüzme", "koşu", "dağcılık",
                "bisiklet", "yoga", "fitness", "antrenman", "spor salonu", "maç", "turnuva",
                "kupa", "şampiyona", "skor", "gol", "kadro", "hakem", "taraftar", "takım",
                "oyuncu", "sporcu", "lig", "rekabet", "hentbol", "okçuluk", "kayak", "boks",
                "motorsporları", "kriket", "beach voleybol", "masa tenisi"
            ]
        }


            
        

    def analyze_topics(self, text):
        if not text:
            return []

        words = text.lower().split()
        topic_matches = []
        total_matches = 0
        
        # Her konu için eşleşmeleri bul
        topic_scores = {}
        matched_words_dict = {}
        
        for topic, keywords in self.topics.items():
            matched_words = [word for word in words if word in keywords]
            if matched_words:
                score = (len(matched_words) / len(words)) * 100
                topic_scores[topic] = score
                matched_words_dict[topic] = matched_words
                total_matches += len(matched_words)
        
        # Skorları normalize et
        if total_matches > 0:
            for topic in topic_scores:
                normalized_score = (topic_scores[topic] / sum(topic_scores.values())) * 100
                topic_matches.append({
                    'konu': topic,
                    'skor': round(normalized_score, 1),
                    'eşleşen_kelimeler': matched_words_dict[topic]
                })
        
        # Skorlarına göre sırala
        topic_matches.sort(key=lambda x: x['skor'], reverse=True)
        
        return topic_matches

class StreamlitAudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 44100
        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        self.visualization_queue = queue.Queue()
        self.lock = threading.Lock()
        
        try:
            model_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\model.pkl'
            scaler_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\scaler.pkl'
            labels_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\labels.pkl'
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.speaker_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(labels_path)
                
        except Exception as e:
            st.error(f"Model yükleme hatası: {str(e)}")
            self.speaker_model = None
            self.scaler = None
            self.label_encoder = None

    def start_recording(self):
        try:
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                if self.is_recording:
                    with self.lock:
                        self.frames.append(indata.copy())
                        # Görselleştirme verilerini kuyruğa ekle
                        if len(self.frames) > 10:  # Her 10 frame'de bir güncelle
                            self.visualization_queue.put(np.concatenate(self.frames[-10:]))

            self.frames = []
            self.is_recording = True
            
            # Ses akışını başlat
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=callback,
                dtype=np.float32
            )
            self.stream.start()
            
            return True
            
        except Exception as e:
            st.error(f"Kayıt başlatma hatası: {str(e)}")
            return False

    def stop_recording(self):
        try:
            self.is_recording = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            return True
        except Exception as e:
            st.error(f"Kayıt durdurma hatası: {str(e)}")
            return False

    def save_recording(self):
        try:
            if not self.frames:
                st.warning("Kayıt bulunamadı!")
                return False
                
            # Ses verisini düzgün formatta kaydet
            audio_data = np.concatenate(self.frames, axis=0)
            audio_data = audio_data.flatten()  # Tek boyutlu diziye dönüştür
            
            # Normalize et ve 16-bit PCM formatına dönüştür
            audio_data = np.int16(audio_data * 32767)
            
            # WAV dosyası olarak kaydet
            wavfile.write(
                "kayit1_pcm.wav",
                self.sample_rate,
                audio_data
            )
            
            st.success("Kayıt başarıyla kaydedildi!")
            return True
            
        except Exception as e:
            st.error(f"Kayıt kaydetme hatası: {str(e)}")
            return False

    def extract_mfcc_features(self, audio_data):
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate,
                n_mfcc=128
            )
            mfcc_scaled = np.mean(mfccs, axis=1)
            return mfcc_scaled
        except Exception as e:
            st.error(f"MFCC özellik çıkarma hatası: {str(e)}")
            return None

    def predict_speaker(self, audio_data):
        try:
            mfcc_features = self.extract_mfcc_features(audio_data)
            if mfcc_features is None:
                return "Belirsiz", {"Belirsiz": 1.0}

            mfcc_features = mfcc_features.reshape(1, -1)
            probabilities = self.speaker_model.predict_proba(mfcc_features)[0]
            
            speakers = ["Nursena", "Sıla", "Zeynep"]
            speaker_probs = {speaker: float(prob) for speaker, prob in zip(speakers, probabilities)}
            predicted_speaker = speakers[np.argmax(probabilities)]
            
            return predicted_speaker, speaker_probs
            
        except Exception as e:
            st.error(f"Konuşmacı tahmini hatası: {str(e)}")
            return "Belirsiz", {"Belirsiz": 1.0}

    def process_recording(self):
        try:
            if not self.save_recording():
                return {"status": "error", "message": "Kayıt bulunamadı"}

            # Ses dosyasını yükle
            try:
                audio_data, sr_rate = librosa.load(
                    "kayit1_pcm.wav",
                    sr=self.sample_rate,
                    mono=True
                )
            except Exception as e:
                st.error(f"Ses dosyası yükleme hatası: {str(e)}")
                return {"status": "error", "message": "Ses dosyası yüklenemedi"}

            # Debug bilgisi
            st.write("Ses dosyası yüklendi:")
            st.write(f"- Örnek sayısı: {len(audio_data)}")
            st.write(f"- Örnekleme hızı: {sr_rate} Hz")
            
            # Konuşmacı tahmini
            predicted_speaker, speaker_probabilities = self.predict_speaker(audio_data)

            # Speech recognition
            try:
                recognizer = Recognizer()
                with AudioFile("kayit1_pcm.wav") as source:
                    audio = recognizer.record(source)
                
                try:
                    transcript = recognizer.recognize_google(audio, language="tr-TR")
                except Exception as e:
                    st.warning("Ses metne çevrilemedi. Google API hatası olabilir.")
                    transcript = "Ses anlaşılamadı"
            except Exception as e:
                st.error(f"Speech recognition hatası: {str(e)}")
                transcript = "Ses işleme hatası"

            word_count = len(transcript.split()) if transcript else 0
            emotion_results = self.emotion_analyzer.analyze_emotion(transcript)
            topic_results = self.topic_analyzer.analyze_topics(transcript)

            return {
                "status": "success",
                "transcript": transcript,
                "wordCount": word_count,
                "emotions": emotion_results["duygular"],
                "topics": topic_results,
                "speaker": predicted_speaker,
                "speaker_probabilities": speaker_probabilities
            }

        except Exception as e:
            st.error(f"İşleme hatası: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_visualization_data(self):
        try:
            return self.visualization_queue.get_nowait()
        except queue.Empty:
            return None

    def register_user(self, user_data):
        """Kullanıcı kayıt fonksiyonu"""
        try:
            # Giriş verilerini kontrol et
            if not all([user_data["email"], user_data["username"], 
                       user_data["password"], user_data["repassword"]]):
                return None

            # Şifrelerin eşleştiğini kontrol et
            if user_data["password"] != user_data["repassword"]:
                return None

            # Başarılı kayıt sonucunu döndür
            return {
                "title": "Kayıt Başarılı",
                "redirectUrl": "/Home/Index",
                "message": "Lütfen e-posta adresinize gönderdiğimiz aktivasyon link'ine tıklayarak hesabınızı aktive ediniz. "
                          "Hesabını aktive etmeden gönderi ekleyemez ve beğeni yapamazsınız"
            }

        except Exception as e:
            st.error(f"Kayıt hatası: {str(e)}")
            return None

# Session state başlatma
if 'audio_recorder' not in st.session_state:
    st.session_state.audio_recorder = StreamlitAudioRecorder()

# Ana başlık
st.markdown("<h1>🎤 Ses Analiz Sistemi</h1>", unsafe_allow_html=True)

# Görselleştirme için placeholder
viz_placeholder = st.empty()

# Kontrol butonları
col1, col2, col3 = st.columns(3)

# Yeni Ses Kaydı ve Model Eğitimi
# Kullanıcıdan konuşmacı adı alın
new_speaker_name = st.text_input("Yeni konuşmacının adı", value="")

if st.sidebar.button("Yeni Ses Kaydı ve Eğit"):
    if not new_speaker_name.strip():
        st.error("Lütfen konuşmacı için bir isim girin!")
    else:
        with st.spinner("Yeni ses kaydediliyor ve model eğitiliyor..."):
            try:
                # Ses kaydını başlat
                if st.session_state.audio_recorder.start_recording():
                    st.success("✅ Ses kaydı başladı. Lütfen birkaç saniye konuşun.")
                    time.sleep(5)  # Ses kaydı süresi (5 saniye)
                    st.session_state.audio_recorder.stop_recording()
                    st.success("✅ Ses kaydı tamamlandı.")
                    
                    # Kaydedilen sesi WAV olarak kaydet
                    if st.session_state.audio_recorder.save_recording():
                        file_path = "new_audio.wav"

                        # MFCC özelliklerini çıkar ve isimle etiketle
                        from model import create_mfcc_features, train_model
                        create_mfcc_features(file_path, "MFCC", new_speaker_name)
                        
                        # Modeli yeniden eğit ve dosyalara kaydet
                        model, scaler, labels = train_model()
                        
                        # Dosyaları kaydet
                        import joblib
                        joblib.dump(model, "model.pkl")  # Model dosyası
                        joblib.dump(scaler, "scaler.pkl")  # Scaler dosyası
                        joblib.dump(labels, "labels.pkl")  # Etiketler dosyası
                        
                        st.success(f"Model başarıyla yeniden eğitildi ve dosyalara kaydedildi! Yeni konuşmacı: {new_speaker_name}")
                    else:
                        st.error("Ses kaydı kaydedilemedi.")
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")






with col1:
    if st.button("🎙 Kaydet"):
        if st.session_state.audio_recorder.start_recording():
            st.success("✅ Kayıt başladı")

with col2:
    if st.button("⏹ Durdur"):
        if st.session_state.audio_recorder.stop_recording():
            st.success("✅ Kayıt bitti")

with col3:
    if st.button("⚙ Analiz Et"):
        with st.spinner('🔄 Analiz ediliyor...'):
            results = st.session_state.audio_recorder.process_recording()
            if results["status"] == "success":
                st.session_state.results = results
                st.success("✅ Analiz tamamlandı!")
                st.rerun()

# Görselleştirme güncelleme döngüsü
if st.session_state.audio_recorder.is_recording:
    while True:
        audio_data = st.session_state.audio_recorder.get_visualization_data()
        if audio_data is not None:
            with viz_placeholder.container():
                # Container'ı daralt
                st.markdown("""
                    <style>
                    .small-chart {
                        margin: 0;
                        padding: 0;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Sinyal grafiği boyutunu küçült
                    fig_signal, ax_signal = plt.subplots(figsize=(3, 1.5))
                    ax_signal.plot(audio_data[-500:], color='#1f77b4', linewidth=0.8)
                    ax_signal.set_title("Ses Sinyali", fontsize=8, pad=2)
                    ax_signal.tick_params(labelsize=6)
                    ax_signal.grid(True, alpha=0.2, linewidth=0.5)
                    # Kenar boşluklarını azalt
                    plt.tight_layout()
                    st.pyplot(fig_signal, use_container_width=False)
                    plt.close(fig_signal)
                
                with col2:
                    # Histogram boyutunu küçült
                    fig_hist, ax_hist = plt.subplots(figsize=(3, 1.5))
                    ax_hist.hist(audio_data, bins=30, color='#2ecc71', alpha=0.7)
                    ax_hist.set_title("Ses Dağılımı", fontsize=8, pad=2)
                    ax_hist.tick_params(labelsize=6)
                    ax_hist.grid(True, alpha=0.2, linewidth=0.5)
                    # Kenar boşluklarını azalt
                    plt.tight_layout()
                    st.pyplot(fig_hist, use_container_width=False)
                    plt.close(fig_hist)
        
        time.sleep(0.1)  # 100ms bekle

# Sonuçları göster
if 'results' in st.session_state:
    results = st.session_state.results
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👥 Konuşmacı", results["speaker"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Kelime Sayısı", results["wordCount"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_emotion = max(results["emotions"].items(), key=lambda x: x[1])[0]
        st.metric("😊 Baskın Duygu", max_emotion)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 👥 Konuşmacı Analizi")
        
        # Pasta grafiğinin boyutunu küçülttüm (6,4 yerine 4,3)
        fig, ax = plt.subplots(figsize=(4, 3))
        speaker_probs = results.get("speaker_probabilities", {})

        # Etiketleri ve değerleri hazırla
        labels = list(speaker_probs.keys())
        values = list(speaker_probs.values())

        # Pasta grafiğini çiz
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=['#2ecc71', '#3498db', '#e74c3c'],
            startangle=90,
            labeldistance=1.1,
            pctdistance=0.8,
            textprops={'fontsize': 8}  # Yazı boyutunu da küçülttüm
        )

        # Etiket stillerini ayarla
        plt.setp(texts, size=8)  # Etiket boyutunu küçülttüm
        plt.setp(autotexts, size=7, weight='bold', color='white')  # Yüzde yazı boyutunu küçülttüm

        # Lejandı grafiğin sağına ekle
        ax.legend(
            wedges,
            [f"{name}: {value*100:.1f}%" for name, value in speaker_probs.items()],
            title="Konuşmacılar",
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            fontsize=7  # Lejand yazı boyutunu küçülttüm
        )

        plt.axis('equal')
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Duygu analizi görselleştirmesi
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 😊 Duygu Analizi")
        
        # Duygu emojileri
        emoji_map = {
            "Mutlu": "😊",
            "Üzgün": "😢",
            "Kızgın": "😠",
            "Heyecanlı": "🤩",
            "Endişeli": "😰",
            "Sakin": "😌",
            "Şaşkın": "😲",
            "Nötr": "😐"
        }
        
        # Tüm duyguları sabit sırayla göster
        emotion_order = ["Mutlu", "Üzgün", "Kızgın", "Heyecanlı", "Endişeli", "Sakin", "Şaşkın", "Nötr"]

        # Her duygu için progress bar
        for emotion in emotion_order:
            score = results["emotions"].get(emotion, 0)  # Eğer duygu yoksa 0 değerini kullan
            emoji = emoji_map.get(emotion, "")
            st.markdown(f"{emoji} {emotion}")
            st.progress(score/100, text=f"{score:.1f}%")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📝 Konu Analizi")
        
        # Renk paleti
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c', '#34495e']
        
        for idx, topic in enumerate(results["topics"]):
            if topic['skor'] > 0:  # Sadece skoru 0'dan büyük konuları göster
                # Konu başlığı ve skor
                st.markdown(f"{topic['konu']} ({topic['skor']}%)")
                
                # Progress bar
                progress_color = colors[idx % len(colors)]
                st.markdown(
                    f"""
                    <div style="border-radius:10px; padding:0px; margin:5px 0;">
                        <div style="width:{topic['skor']}%; height:20px; 
                                  background-color:{progress_color}; border-radius:10px;
                                  transition: width 0.3s ease-in-out;">
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Eşleşen kelimeler
                if topic['eşleşen_kelimeler']:
                    with st.expander("Eşleşen Kelimeler"):
                        st.write(", ".join(topic['eşleşen_kelimeler']))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Konuşma Metni")
        st.text_area(
            label="Konuşma Metni",
            value=results.get("transcript", ""),
            height=150,
            key="transcript",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)