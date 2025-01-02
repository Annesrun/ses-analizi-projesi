from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import os
import numpy as np
import collections
import librosa
import soundfile as sf
import sounddevice as sd
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Sabit değişkenler
HIDDEN_LAYERS = (128, 64)
MAX_ITERATIONS = 1000
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 90
TEST_SIZE = 0.2
SAMPLE_RATE = 44100
DURATION = 5  # saniye
CHANNELS = 1

def create_mfcc_features(wav_path, mfcc_path, kisi_adi):
    """Ses dosyasından MFCC özelliklerini çıkarır ve kaydeder."""
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(wav_path)
        
        # MFCC özelliklerini çıkar
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        
        # Kişiye özel klasör oluştur
        kisi_klasoru = os.path.join(mfcc_path, kisi_adi)
        os.makedirs(kisi_klasoru, exist_ok=True)
        
        # Dosya adını al ve .npy uzantılı olarak kaydet
        dosya_adi = os.path.splitext(os.path.basename(wav_path))[0]
        kayit_yolu = os.path.join(kisi_klasoru, f"{dosya_adi}.npy")
        
        # MFCC'yi kaydet
        np.save(kayit_yolu, mfcc)
        print(f"MFCC kaydedildi: {kayit_yolu}")
        
        return True
    except Exception as e:
        print(f"Hata oluştu ({wav_path}): {str(e)}")
        return False

def process_all_wav_files():
    """Tüm WAV dosyalarını işler ve MFCC özelliklerini çıkarır."""
    ses_dizini = r"C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\bolunmus_wav"
    mfcc_dizini = r"C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\MFCC"
    
    os.makedirs(mfcc_dizini, exist_ok=True)
    
    basari_sayisi = 0
    hata_sayisi = 0
    
    for kisi in os.listdir(ses_dizini):
        kisi_yolu = os.path.join(ses_dizini, kisi)
        if os.path.isdir(kisi_yolu):
            print(f"\nKişi işleniyor: {kisi}")
            
            for dosya in os.listdir(kisi_yolu):
                if dosya.endswith('.wav'):
                    wav_yolu = os.path.join(kisi_yolu, dosya)
                    if create_mfcc_features(wav_yolu, mfcc_dizini, kisi):
                        basari_sayisi += 1
                    else:
                        hata_sayisi += 1
    
    print(f"\nİşlem tamamlandı!")
    print(f"Başarılı: {basari_sayisi}")
    print(f"Hatalı: {hata_sayisi}")

def record_audio(output_path):
    """Mikrofondan ses kaydeder."""
    print("Kayıt başlıyor...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                    samplerate=SAMPLE_RATE, 
                    channels=CHANNELS)
    sd.wait()
    print("Kayıt tamamlandı!")
    
    sf.write(output_path, recording, SAMPLE_RATE)
    return output_path

def predict_speaker(model, scaler, wav_path, sinif_isimleri):
    """Ses kaydından konuşmacıyı tahmin eder."""
    try:
        # Ses dosyasını yükle ve MFCC çıkar
        y, sr = librosa.load(wav_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Özellikleri ölçeklendir
        mfcc_scaled = scaler.transform(mfcc_mean.reshape(1, -1))
        
        # Tahmin yap
        tahmin = model.predict(mfcc_scaled)[0]
        olasiliklar = model.predict_proba(mfcc_scaled)[0]
        guven = olasiliklar[tahmin]
        
        return sinif_isimleri[tahmin], guven
        
    except Exception as e:
        print(f"Tahmin sırasında hata: {str(e)}")
        return None, 0

def pad_or_truncate(mfcc_data, target_length=1):
    """MFCC verilerini hedef uzunluğa getirir"""
    if len(mfcc_data.shape) == 1:
        mfcc_data = mfcc_data.reshape(-1, 128)
    
    # Sadece ilk zaman adımını al
    return mfcc_data[:, :target_length]

def process_mfcc(mfcc_data):
    """MFCC verilerini model için hazırlar"""
    try:
        # Boyutu standardize et
        mfcc_padded = pad_or_truncate(mfcc_data)
        # Veriyi düzleştir (128, 1) -> (128,)
        return mfcc_padded.flatten()
    except Exception as e:
        print(f"MFCC işleme hatası: {str(e)}")
        raise

def predict(mfcc_data):
    """Yeni kayıt için tahmin yapar"""
    try:
        # MFCC verilerini işle
        mfcc_processed = process_mfcc(mfcc_data)
        
        # Boyut kontrolü yap
        expected_features = 55168  # 128 * 431
        if len(mfcc_processed) != expected_features:
            print(f"Boyut ayarlanıyor: {len(mfcc_processed)} -> {expected_features}")
            mfcc_processed = np.resize(mfcc_processed, expected_features)
        
        # Veriyi yeniden şekillendir
        mfcc_reshaped = mfcc_processed.reshape(1, -1)
        
        # Scaler'ı yükle ve uygula
        scaler = np.load('scaler.npy', allow_pickle=True)
        mfcc_scaled = scaler.transform(mfcc_reshaped)
        
        return mfcc_scaled
        
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        raise

def train_model():
    """Modeli eğitir ve kaydeder."""
    # MFCC dizinini kontrol et
    mfcc_dizin = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\Egitim'
    if not os.path.exists(mfcc_dizin):
        print(f"HATA: MFCC dizini bulunamadı: {mfcc_dizin}")
        return None, None, None

    X = []
    y = []

    # MFCC dizinini kontrol et
    mfcc_dizin = 'MFCC'
    print(f"MFCC dizini: {mfcc_dizin}")

    # MFCC klasörlerini listele
    mfcc_folders = os.listdir(mfcc_dizin)
    print(f"Bulunan MFCC klasör sayısı: {len(mfcc_folders)}")

    # Her bir konuşmacı klasörü için
    for speaker_folder in mfcc_folders:
        speaker_path = os.path.join(mfcc_dizin, speaker_folder)
        
        # Klasör içeriğini kontrol et
        if os.path.isdir(speaker_path):
            print(f"\nKonuşmacı klasörü işleniyor: {speaker_folder}")
            
            # Klasör içindeki MFCC dosyalarını listele
            mfcc_files = [f for f in os.listdir(speaker_path) if f.endswith('.npy')]
            print(f"Bulunan MFCC dosya sayısı: {len(mfcc_files)}")
            
            # Her bir MFCC dosyası için
            for mfcc_file in mfcc_files:
                try:
                    # MFCC dosyasını yükle
                    mfcc_path = os.path.join(speaker_path, mfcc_file)
                    mfcc_data = np.load(mfcc_path)
                    print(f"Yüklenen dosya: {mfcc_file}, Boyut: {mfcc_data.shape}")
                    
                    # Tüm MFCC verilerini aynı boyuta getir
                    mfcc_data = pad_or_truncate(mfcc_data)
                    # Veriyi düzleştir
                    mfcc_flat = mfcc_data.flatten()
                    X.append(mfcc_flat)
                    y.append(speaker_folder)
                    
                except Exception as e:
                    print(f"Hata: {mfcc_file} dosyası işlenirken hata oluştu - {str(e)}")

    print(f"\nOluşturulan veri seti boyutu: X={len(X)}, y={len(y)}")

    if len(X) == 0:
        print("HATA: Veri seti boş!")
        print("Model eğitimi başarısız!")
        return None, None, None

    X = np.array(X)
    y = np.array(y)

    # Etiketleri dönüştür
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sınıf dengesizliğini gider
    X_train_balanced = []
    y_train_balanced = []

    for label in np.unique(y_train):
        indices = np.where(y_train == label)[0]
        X_class = X_train_scaled[indices]
        y_class = y_train[indices]
        
        if len(X_class) < MIN_SAMPLES_PER_CLASS:
            X_resampled, y_resampled = resample(
                X_class, y_class, 
                n_samples=MIN_SAMPLES_PER_CLASS, 
                random_state=RANDOM_STATE
            )
        else:
            X_resampled, y_resampled = X_class, y_class
            
        X_train_balanced.extend(X_resampled)
        y_train_balanced.extend(y_resampled)

    X_train_balanced = np.array(X_train_balanced)
    y_train_balanced = np.array(y_train_balanced)

    # Farklı modeller dene
    models = {
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier()
    }
    
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        score = model.score(X_test_scaled, y_test)
        print(f"{name} Test Doğruluğu: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model

    # Modeli kaydet
    model_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\model.pkl'
    scaler_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\scaler.pkl'
    labels_path = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\labels.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le.classes_, labels_path)
    
    print("Model ve gerekli dosyalar kaydedildi!")
    
    return best_model, scaler, le.classes_

def augment_audio(mfcc_data):
    """MFCC verilerini çeşitlendirir"""
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5)
    ])
    
    augmented_data = []
    augmented_data.append(mfcc_data)  # Orijinal veri
    
    # Her örnek için 3 farklı augmentasyon uygula
    for _ in range(3):
        augmented = augmenter(samples=mfcc_data)
        augmented_data.append(augmented)
    
    return augmented_data

def evaluate_model(model, X, y):
    """Model performansını değerlendirir"""
    # 5-fold cross validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation skorları: {cv_scores}")
    print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def main():
    """Ana program döngüsü"""
    # Önce MFCC özelliklerini oluştur
    print("MFCC özellikleri oluşturuluyor...")
    process_all_wav_files()
    
    # Modeli eğit
    print("\nModel eğitiliyor...")
    model, scaler, sinif_isimleri = train_model()
    
    if model is None:
        print("Model eğitimi başarısız!")
        return
    
    # Tahmin döngüsü
    while True:
        print("\n1: Yeni kayıt al ve tahmin et")
        print("2: Çıkış")
        secim = input("Seçiminiz: ")
        
        if secim == "1":
            # Ses kaydı al
            kayit_yolu = r'C:\Users\Excalibur\Desktop\sesanalizprojesi\sesanalizprojesi\123\kayit1_pcm.wav'
            record_audio(kayit_yolu)
            
            # Tahmin yap
            tahmin, guven = predict_speaker(model, scaler, kayit_yolu, sinif_isimleri)
            if tahmin:
                print(f"\nTahmin edilen kişi: {tahmin}")
                print(f"Güven skoru: {guven:.2f}")
            
            # Geçici dosyayı sil
            if os.path.exists(kayit_yolu):
                os.remove(kayit_yolu)
                
        elif secim == "2":
            print("Program sonlandırılıyor...")
            break
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main()