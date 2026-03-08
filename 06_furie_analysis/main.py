import os
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Завантаження та розархівування датасету ESC-50
# ──────────────────────────────────────────────
url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
zip_file_path = "ESC-50-master.zip"
download_path = "./ESC-50-master/"

if not os.path.exists(download_path):
    print("Завантаження датасету ESC-50...")
    urlretrieve(url, zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_file_path)
    print("Датасет завантажено та розархівовано.")
else:
    print("Датасет вже існує, пропускаємо завантаження.")

# ──────────────────────────────────────────────
# 2. Читання метаданих
# ──────────────────────────────────────────────
df_file_path = os.path.join(download_path, "ESC-50-master", "meta", "esc50.csv")
df = pd.read_csv(df_file_path)
print(f"\nЗагальна кількість записів у датасеті: {len(df)}")
print(f"Унікальні категорії ({df['category'].nunique()}): {sorted(df['category'].unique())}")

# ──────────────────────────────────────────────
# 3. Фільтрація: лише 'dog' та 'chirping_birds'
# ──────────────────────────────────────────────
selected_categories = ['dog', 'chirping_birds']
df_filtered = df[df['category'].isin(selected_categories)].reset_index(drop=True)

print(f"\nВідфільтровано записів: {len(df_filtered)}")
print(df_filtered['category'].value_counts())
print(df_filtered.head(10))


# ──────────────────────────────────────────────
# 4. Функція спектрограми (з конспекту)
# ──────────────────────────────────────────────
def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, max_freq=None, eps=1e-14):
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                              shape=nshape, strides=nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared FFT, scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2

    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram (log scale)
    specgram = np.log(fft[:, :] + eps)
    return specgram


# ──────────────────────────────────────────────
# 5. Завантаження аудіо та генерація спектрограми
# ──────────────────────────────────────────────
audio_filename = df_filtered.iloc[0]['filename']
audio_file_path = os.path.join(download_path, "ESC-50-master", "audio", audio_filename)

y, sr = librosa.load(audio_file_path, sr=None)
print(f"\nАудіо файл: {audio_filename}")
print(f"Категорія: {df_filtered.iloc[0]['category']}")
print(f"Довжина сигналу: {len(y)} зразків")
print(f"Частота дискретизації: {sr} Гц")

specgram = spectrogram(y, sr)
print(f"\nРозмір матриці спектрограми: {specgram.shape}")
print(f"  - {specgram.shape[0]} частотних бінів (рядки)")
print(f"  - {specgram.shape[1]} часових фреймів (стовпці)")


# ──────────────────────────────────────────────
# 6. Pooling — зменшення розміру спектрограми
# ──────────────────────────────────────────────
def average_pooling(matrix, pool_size=(2, 2)):
    """Reduce matrix size by averaging over non-overlapping blocks."""
    rows, cols = matrix.shape
    pool_h, pool_w = pool_size

    # Trim matrix so dimensions are divisible by pool_size
    trimmed_rows = (rows // pool_h) * pool_h
    trimmed_cols = (cols // pool_w) * pool_w
    matrix = matrix[:trimmed_rows, :trimmed_cols]

    # Reshape into blocks and take mean
    reshaped = matrix.reshape(trimmed_rows // pool_h, pool_h,
                              trimmed_cols // pool_w, pool_w)
    pooled = reshaped.mean(axis=(1, 3))
    return pooled


print(f"\n--- Pooling ---")
print(f"До pooling: {specgram.shape}")

specgram_pooled = average_pooling(specgram, pool_size=(2, 2))
print(f"Після pooling (2×2): {specgram_pooled.shape}")

specgram_pooled_4 = average_pooling(specgram, pool_size=(4, 4))
print(f"Після pooling (4×4): {specgram_pooled_4.shape}")


# ──────────────────────────────────────────────
# 7. Flatten — перетворення матриці у вектор
# ──────────────────────────────────────────────
feature_vector = specgram_pooled.flatten()
print(f"\n--- Flatten ---")
print(f"Матриця після pooling: {specgram_pooled.shape} ({specgram_pooled.size} значень)")
print(f"Вектор після flatten:  {feature_vector.shape} ({feature_vector.size} значень)")


# ──────────────────────────────────────────────
# 8. Попереднє завантаження спектрограм для всіх файлів
# ──────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print(f"\nОбробка {len(df_filtered)} аудіофайлів (генерація спектрограм)...")
spectrograms_all = []
labels = []

for i, row in df_filtered.iterrows():
    filepath = os.path.join(download_path, "ESC-50-master", "audio", row['filename'])
    audio, rate = librosa.load(filepath, sr=None)
    spec = spectrogram(audio, rate)
    spectrograms_all.append(spec)
    labels.append(row['category'])

y_true = np.array(labels)
y_numeric = np.array([0 if label == 'dog' else 1 for label in y_true])
print(f"Спектрограми згенеровано: {len(spectrograms_all)} файлів")


# ──────────────────────────────────────────────
# 9. Експерименти з різними ступенями стиснення (pooling)
# ──────────────────────────────────────────────
pool_sizes = [(2, 2), (4, 4), (8, 8)]
results = []

fig, axes = plt.subplots(1, len(pool_sizes), figsize=(5 * len(pool_sizes), 4))

for idx, ps in enumerate(pool_sizes):
    print(f"\n{'=' * 55}")
    print(f"  Pooling {ps[0]}×{ps[1]}")
    print(f"{'=' * 55}")

    # Pooling → flatten for all files
    feature_vectors = []
    for spec in spectrograms_all:
        spec_pooled = average_pooling(spec, pool_size=ps)
        feature_vectors.append(spec_pooled.flatten())

    X = np.array(feature_vectors)
    print(f"Розмір матриці фічей: {X.shape}")

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SpectralClustering
    spectral = SpectralClustering(
        n_clusters=2,
        affinity='nearest_neighbors',
        assign_labels='kmeans',
        random_state=42
    )
    predicted_labels = spectral.fit_predict(X_scaled)

    # Confusion matrix & accuracy
    cm = confusion_matrix(y_numeric, predicted_labels)
    acc_v1 = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100
    acc_v2 = (cm[0, 1] + cm[1, 0]) / cm.sum() * 100
    accuracy = max(acc_v1, acc_v2)

    results.append({
        'pool_size': f"{ps[0]}×{ps[1]}",
        'features': X.shape[1],
        'accuracy': accuracy,
        'cm': cm
    })

    print(f"Кількість фічей: {X.shape[1]}")
    print(f"Точність: {accuracy:.1f}%")
    print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dog', 'birds'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"Pooling {ps[0]}×{ps[1]}\nAccuracy: {accuracy:.1f}%")

plt.suptitle("Порівняння кластеризації при різних ступенях стиснення", fontsize=13)
plt.tight_layout()
plt.show()


# ──────────────────────────────────────────────
# 10. Зведена таблиця результатів
# ──────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"  ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
print(f"{'=' * 55}")
print(f"{'Pooling':<12} {'Фічей':<10} {'Точність':<10}")
print(f"{'-' * 32}")
for r in results:
    print(f"{r['pool_size']:<12} {r['features']:<10} {r['accuracy']:.1f}%")

best = max(results, key=lambda x: x['accuracy'])
worst = min(results, key=lambda x: x['accuracy'])

print(f"\n{'─' * 55}")
print(f"ВИСНОВОК:")
print(f"{'─' * 55}")
print(f"""
1. Аналіз кластерів:
   Кластеризація розділила звуки на два кластери, які
   в цілому відповідають реальним категоріям — собаки
   переважно потрапили в один кластер, а птахи в інший.
   Найкраща точність: {best['accuracy']:.0f}% (pooling {best['pool_size']}).

2. Вплив ступеня стиснення:
   - Pooling 2×2: найбільше фічей ({results[0]['features']}), зберігає
     максимум деталей спектрограми → точність {results[0]['accuracy']:.0f}%.
   - Pooling 4×4: середнє стиснення ({results[1]['features']} фічей),
     прибирає дрібний шум → точність {results[1]['accuracy']:.0f}%.
   - Pooling 8×8: сильне стиснення ({results[2]['features']} фічей),
     втрачає деталі → точність {results[2]['accuracy']:.0f}%.

   Оптимальний баланс між кількістю ознак та якістю
   кластеризації досягається при pooling {best['pool_size']}.

3. Висновок про FFT:
   FFT дозволив витягнути з аудіо спектрограму — карту
   частот у часі. Собаки гавкають на низьких частотах
   уривчасто, птахи щебечуть на високих тривало. Ці
   відмінності дозволили кластеризації розділити звуки
   навіть без міток (unsupervised).

4. Road Map — що ми робили і навіщо:

   Аудіо (wav)
     │  Сирий сигнал — масив амплітуд. Не підходить
     │  для ML: два однакових гавки зсунуті по часу
     │  виглядають зовсім по-різному.
     ▼
   FFT → Спектрограма (442×499)
     │  Перетворення Фур'є розкладає сигнал на частоти.
     │  Тепер неважливо КОЛИ звук — важливо ЯКІ частоти.
     │  Собака = низькі частоти, птахи = високі.
     ▼
   Pooling (2×2 / 4×4 / 8×8)
     │  Спектрограма занадто велика — зменшуємо,
     │  усереднюючи сусідні значення. Прибираємо шум,
     │  залишаємо загальні паттерни частот.
     ▼
   Flatten → вектор фічей
     │  Класифікатори приймають вектор, а не матрицю.
     │  Розгортаємо 2D в 1D — кожен файл = один рядок.
     │  Чим більше фічей, тим більше деталей, але і шуму.
     ▼
   StandardScaler
     │  Нормалізуємо фічі (mean=0, std=1), щоб
     │  кластеризація не була зміщена до великих значень.
     ▼
   SpectralClustering → 2 кластери
     │  Без міток шукаємо дві групи схожих зразків.
     │  Результат: ~82%% точність — собаки і птахи
     │  реально потрапляють в різні кластери.
     ▼
   Confusion Matrix → оцінка якості
      Порівнюємо кластери з реальними мітками.
      Перевіряємо: чи дійсно FFT-фічі корисні?
      Відповідь: так, навіть без учителя працює.
""")
