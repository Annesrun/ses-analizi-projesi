from pydub import AudioSegment
import os

# WAV dosyalarının bulunduğu tam dizin
wav_dir = r'C:\Users\Lenovo\Desktop\123\Sesler_wav'

# Çıkış dizini (bölünmüş dosyalar kaydedilecek)
output_dir = r'C:\Users\Lenovo\Desktop\123\bolunmus_wav'
os.makedirs(output_dir, exist_ok=True)

# Klasörleri tanımla
folders = ["Kerem_ses", "Emir_ses",]

# Her bir klasör için işlemleri uygula
for folder in folders:
    folder_path = os.path.join(wav_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)

    # Klasörün varlığını kontrol et
    if not os.path.exists(folder_path):
        print(f"Klasör bulunamadı: {folder_path}")
        continue

    # Klasördeki wav dosyalarını listele
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        
        # Eğer Kerem_ses klasöründeki dosya "kerem_fixed.wav" ise işlenir
        if folder == "Kerem_ses" and wav_file == "Kerem_fixed.wav":
            print(f"İşleniyor: {file_path}")

        try:
            # Ses dosyasını yükle
            audio = AudioSegment.from_wav(file_path)

            # Ses dosyasını 5 saniyelik parçalara böl
            chunk_length_ms = 5000  # 5 saniye
            chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Parçaları kaydet
            for i, chunk in enumerate(chunks):
             chunk.export(os.path.join(output_folder, f"{wav_file[:-4]}_part{i+1}.wav"), format="wav")

        
        except Exception as e:
            print(f"Hata oluştu: {wav_file} - {e}")

print("Ses dosyaları başarıyla işlendi ve kaydedildi.")