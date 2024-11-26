import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Đường dẫn tới dataset
DATASET_PATH = "data"  # Thư mục chứa các tệp âm thanh, chia thành các thư mục con (cat, dog, bird)
LABELS = ["cat", "dog", "bird"]

# Hàm chuyển đổi âm thanh thành ma trận phổ Mel
def extract_mel_spectrogram(file_path, n_mels=128, max_len=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] > max_len:
        mel_spec_db = mel_spec_db[:, :max_len]
    else:
        padding = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, padding)), mode='constant')
    return mel_spec_db

# Hàm load dữ liệu từ thư mục
def load_data(dataset_path):
    data = []
    targets = []
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(dataset_path, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            try:
                mel_spec = extract_mel_spectrogram(file_path)
                data.append(mel_spec)
                targets.append(label_idx)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return np.array(data), np.array(targets)

# Load dữ liệu
print("Loading dataset...")
data, targets = load_data(DATASET_PATH)
data = data[..., np.newaxis]  # Thêm chiều cho channel (dạng ảnh)

# Chia tập dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Xây dựng mô hình CNN
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Khởi tạo và huấn luyện mô hình
print("Building CNN model...")
model = build_cnn(input_shape=X_train[0].shape, num_classes=len(LABELS))

print("Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Đánh giá mô hình trên tập test
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Lưu mô hình đã huấn luyện
MODEL_PATH = "ok_model.keras"
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")

