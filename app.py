import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    for folder in os.listdir(dataset_path):
        label = folder  # Folder name sebagai label
        folder_path = os.path.join(dataset_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # filter grayscale
            img = cv2.resize(img, (32, 32))  # Resize gambar menjadi 32x32
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
print("Memuat dataset...")
dataset_path = "SIBI"
images, labels = load_dataset(dataset_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train KNN model
print("Melatih model...")
knn = KNeighborsClassifier(n_neighbors=10)  # Menambahkan jumlah tetangga untuk akurasi yang lebih baik
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)  # Hitung akurasi model
print(f"Akurasi: {accuracy * 100:.2f}%")

# OpenCV realtime
print("Memulai deteksi real-time...")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break
    roi = frame[100:300, 100:300]  # Sesuaikan ukuran ROI menjadi lebih kecil
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
    resized = cv2.resize(gray, (32, 32)).flatten()
    reshaped = resized.reshape(1, -1)
    predicted_letter = knn.predict(reshaped)[0]  # Prediksi setiap huruf
    cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.imshow("SIBI Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
