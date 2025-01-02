import os
import shutil

# Kaynak MFCC dizinini ve hedef dizinleri belirtin
mfcc_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC'
egitim_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Egitim'
test_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC_Test'

# Eğitim ve test dizinlerini oluştur
os.makedirs(egitim_dizin, exist_ok=True)
os.makedirs(test_dizin, exist_ok=True)

# Klasörleri kontrol ederek dosyaları sırala ve ayır
for kisi_klasoru in os.listdir(mfcc_dizin):
    kisi_kaynak_dizin = os.path.join(mfcc_dizin, kisi_klasoru)
    kisi_egitim_dizin = os.path.join(egitim_dizin, kisi_klasoru)
    kisi_test_dizin = os.path.join(test_dizin, kisi_klasoru)

    if os.path.isdir(kisi_kaynak_dizin):
        # Eğitim ve test klasörlerini oluştur
        os.makedirs(kisi_egitim_dizin, exist_ok=True)
        os.makedirs(kisi_test_dizin, exist_ok=True)

        # Dosyaları alfabetik olarak sırala
        dosyalar = sorted([f for f in os.listdir(kisi_kaynak_dizin) if f.endswith('.npy')])

        # Dosyaları eğitim ve test setlerine ayır
        for i, dosya in enumerate(dosyalar):
            kaynak_dosya = os.path.join(kisi_kaynak_dizin, dosya)
            if i < 96:  # İlk 96 dosya eğitim setine
                hedef_dosya = os.path.join(kisi_egitim_dizin, dosya)
            else:  # Son 24 dosya test setine
                hedef_dosya = os.path.join(kisi_test_dizin, dosya)
            
            # Dosyayı kopyala
            shutil.copy(kaynak_dosya, hedef_dosya)

print("Veriler başarıyla 96/24 oranında eğitim ve test setlerine ayrıldı!")