import os

# Ana dizin
ana_dizin = r'C:\Users\Lenovo\Desktop\123\MFCC'

# Kişi klasörlerini listele
kisi_klasorleri = [f for f in os.listdir(ana_dizin) if os.path.isdir(os.path.join(ana_dizin, f))]

if not kisi_klasorleri:
    print("Ana dizinde kişi klasörleri bulunamadı.")
    exit()

print("Kişi klasörleri:", kisi_klasorleri)

print("Dosya isimlendirme başlatıldı...")

for kisi in kisi_klasorleri:
    kisi_dizin = os.path.join(ana_dizin, kisi)
    if not os.path.exists(kisi_dizin):
        print(f"Klasör mevcut değil: {kisi_dizin}")
        continue
    
    dosyalar = sorted([f for f in os.listdir(kisi_dizin) if f.endswith('.npy')])
    print(f"{kisi} klasöründe {len(dosyalar)} dosya bulundu.")
    
    if not dosyalar:
        print(f"{kisi} klasöründe hiçbir .npy dosyası bulunamadı.")
        continue

    # Dosyaları sırayla yeniden adlandır
    for i, dosya in enumerate(dosyalar):
        eski_dosya_yolu = os.path.join(kisi_dizin, dosya)
        yeni_dosya_adi = f"{kisi}segment{i + 1}.npy"  # Yeni dosya ismi
        yeni_dosya_yolu = os.path.join(kisi_dizin, yeni_dosya_adi)
        os.rename(eski_dosya_yolu, yeni_dosya_yolu)
        print(f"Yeniden adlandırıldı: {eski_dosya_yolu} -> {yeni_dosya_yolu}")

print("Dosya isimlendirme tamamlandı!")