import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Memuat Data ---
try:
    df = pd.read_csv('iris.data.csv', header=None)
    print("File 'iris.data.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'iris.data.csv' tidak ditemukan.")
    exit()

# --- 2. Persiapan Data ---
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nUkuran Training Set: {X_train.shape}")
print(f"Ukuran Test Set: {X_test.shape}")

# --- 3. Inisialisasi dan Training Model k-NN ---
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# --- 4. Prediksi dan Evaluasi ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Hasil Prediksi ---")
print(f"Prediksi:      {y_pred}")
print(f"Label Aktual:  {y_test}")
print(f"Akurasi: {accuracy:.4f}")

# --- 5. Prediksi Data Baru ---
new_sample = np.array([[5.0, 3.3, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)

print("\n--- Prediksi Sampel Baru ---")
print(f"Fitur bunga baru: {new_sample[0]}")
print(f"Prediksi spesies: {prediction[0]}")

# --- 6. Visualisasi ---
_, X_test_original, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_test_original[:, 2], y=X_test_original[:, 3], hue=y_test,
                style=y_pred, palette='deep', s=100, alpha=0.85)

plt.scatter(new_sample[0][2], new_sample[0][3],
            marker='X', s=200, color='red', label=f'Sampel Baru (Prediksi: {prediction[0]})', zorder=5)

plt.title(f'Klasifikasi Iris dengan KNeighborsClassifier (k={k})')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Spesies / Prediksi', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
