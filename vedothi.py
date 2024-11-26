import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

file_path = '/home/mint/Documents/VS Code/dog_bird_cat/test/cat/test_cat_1b0.wav'  # Đường dẫn file âm thanh
# Tải tệp âm thanh mẫu (thay 'dog_sample.wav' bằng tệp bạn ghi âm từ "dog")
y, sr = librosa.load(file_path)

# Trích xuất MFCC từ tệp âm thanh mẫu
mfcc_sample = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Lưu MFCC vào file .npy
np.save('dog_mfcc_sample.npy', mfcc_sample)

print("Mẫu MFCC đã được lưu thành công.")


# Bước 1: Đọc file âm thanh và chuyển thành phổ Mel

y, sr = librosa.load(file_path)  # Đọc âm thanh

# Tính phổ Mel
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # 128 dải tần Mel

# Chuyển đổi sang thang logarit (dB) để dễ phân tích hơn
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Bước 2: Vẽ phổ Mel sử dụng matplotlib
plt.figure(figsize=(10, 6))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')  # Thêm thanh màu cho mức dB
plt.title('Mel Spectrogram')
plt.xlabel('Thời gian (s)')
plt.ylabel('Tần số Mel')
plt.show()
