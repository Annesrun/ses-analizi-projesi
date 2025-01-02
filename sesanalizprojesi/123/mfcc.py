import os
import librosa
import numpy as np

# Eğitim ve Test klasörlerinin yolları
egitim_klasoru = r'C:\Users\Excalibur\Desktop\kerememirproje\123\Egitim'
test_klasoru = r'C:\Users\Excalibur\Desktop\kerememirproje\123\Test'

# MFCC'leri kaydedeceğiniz klasörlerin yolları
mfcc_egitim_klasoru = r'C:\Users\Excalibur\Desktop\kerememirproje\123\MFCC_Egitim'
mfcc_test_klasoru = r'C:\Users\Excalibur\Desktop\kerememirproje\123\MFCC_Test'

# MFCC klasörlerini oluştur
os.makedirs(mfcc_egitim_klasoru, exist_ok=True)
os.makedirs(mfcc_test_klasoru, exist_ok=True)

def mfcc_cikar_ve_kaydet(kaynak_klasor, hedef_klasor):
    for kisi_klasoru in os.listdir(kaynak_klasor):
        kisi_yolu = os.path.join(kaynak_klasor, kisi_klasoru)
        if os.path.isdir(kisi_yolu):
            kisi_hedef_klasoru = os.path.join(hedef_klasor, kisi_klasoru)
            os.makedirs(kisi_hedef_klasoru, exist_ok=True)
            
            for dosya in os.listdir(kisi_yolu):
                if dosya.endswith('.wav'):
                    dosya_yolu = os.path.join(kisi_yolu, dosya)
                    try:
                        y, sr = librosa.load(dosya_yolu, sr=None)  # Orijinal örnekleme hızında yükle
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
                        mfcc_dosya_adi = os.path.splitext(dosya)[0] + '.npy'
                        mfcc_dosya_yolu = os.path.join(kisi_hedef_klasoru, mfcc_dosya_adi)
                        np.save(mfcc_dosya_yolu, mfcc)
                        print(f"MFCC kaydedildi: {mfcc_dosya_yolu}")
                    except Exception as e:
                        print(f"Hata: {dosya_yolu} dosyasından MFCC çıkarılamadı. Hata: {e}")

# Eğitim seti için MFCC çıkarma
print("Eğitim seti için MFCC özellikleri çıkarılıyor...")
mfcc_cikar_ve_kaydet(egitim_klasoru, mfcc_egitim_klasoru)

# Test seti için MFCC çıkarma
print("Test seti için MFCC özellikleri çıkarılıyor...")
mfcc_cikar_ve_kaydet(test_klasoru, mfcc_test_klasoru)

print("MFCC çıkarma işlemi tamamlandı!")

