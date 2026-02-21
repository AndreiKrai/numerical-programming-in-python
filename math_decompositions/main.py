import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import TruncatedSVD

# --- 1. Завантаження та відображення зображення ---
image = mpimg.imread("../1.jpg")

# mpimg.imread повертає float32 [0,1] для JPEG — переводимо в uint8 [0,255],
# щоб далі працювати у форматі, який вказано у завданні
if image.dtype != np.uint8:
    image = (image * 255).astype(np.uint8)

plt.imshow(image)
plt.title("Оригінальне зображення")
plt.axis("off")
plt.show()

# --- 2. Розмір зображення ---
print(f"Розмір зображення (shape): {image.shape}")

# --- 3. Перетворення 3D → 2D (укладання каналів горизонтально) ---
height, width, channels = image.shape
flat_image = image.reshape(-1, width * channels).astype(np.float64)
print(f"Розмір 2D-матриці: {flat_image.shape}")

# --- 4. SVD декомпозиція (numpy.linalg.svd) ---
U, S, Vt = np.linalg.svd(flat_image, full_matrices=False)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

# --- 5. Візуалізація перших k сингулярних значень матриці Σ ---
k = 50
plt.figure()
plt.plot(np.arange(k), S[:k])
plt.title(f"Перші {k} сингулярних значень (Σ)")
plt.xlabel("Індекс")
plt.ylabel("Сингулярне значення")
plt.grid(True)
plt.show()

# --- 6. Усічена SVD (TruncatedSVD) з k=100 ---
svd = TruncatedSVD(n_components=100)
truncated_image = svd.fit_transform(flat_image)

# --- 7. Реконструкція та похибка (MSE) ---
reconstructed_image = svd.inverse_transform(truncated_image)

reconstruction_error = np.mean(np.square(reconstructed_image - flat_image))
print(f"Похибка реконструкції (MSE) при k=100: {reconstruction_error:.4f}")

# --- 8. Візуалізація реконструйованого зображення ---
reconstructed_image = reconstructed_image.reshape(height, width, channels)
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype('uint8')

plt.imshow(reconstructed_image)
plt.axis('off')
plt.title("Реконструйоване зображення (k=100)")
plt.show()

# --- 9. Експерименти з різними значеннями k ---
k_values = [5, 10, 25, 50, 100, 200]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, k_val in enumerate(k_values):
    svd_k = TruncatedSVD(n_components=k_val)
    truncated_k = svd_k.fit_transform(flat_image)
    reconstructed_k = svd_k.inverse_transform(truncated_k)

    mse = np.mean(np.square(reconstructed_k - flat_image))

    reconstructed_k = reconstructed_k.reshape(height, width, channels)
    reconstructed_k = np.clip(reconstructed_k, 0, 255).astype('uint8')

    axes[i].imshow(reconstructed_k)
    axes[i].axis('off')
    axes[i].set_title(f"k = {k_val}, MSE = {mse:.2f}")

plt.suptitle("Реконструкція зображення при різних значеннях k", fontsize=16)
plt.tight_layout()
plt.show()

# --- Графік залежності MSE від k ---
k_range = [1, 2, 5, 10, 15, 25, 50, 75, 100, 150, 200]
mse_values = []

for k_val in k_range:
    svd_k = TruncatedSVD(n_components=k_val)
    truncated_k = svd_k.fit_transform(flat_image)
    reconstructed_k = svd_k.inverse_transform(truncated_k)
    mse = np.mean(np.square(reconstructed_k - flat_image))
    mse_values.append(mse)
    print(f"  k = {k_val:>3d} | MSE = {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, mse_values, 'o-')
plt.title("Залежність похибки реконструкції (MSE) від k")
plt.xlabel("k (кількість компонент)")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# --- Висновки ---
print("\n=== ВИСНОВКИ ===")
print("1. SVD декомпозиція дозволяє ефективно стискати зображення, зберігаючи лише")
print("   найбільш значущі сингулярні значення.")
print("2. Графік сингулярних значень показує різке зниження після перших компонент,")
print("   що підтверджує можливість стиснення без значної втрати якості.")
print("3. При k=5-10 — помітна значна втрата якості (зображення розмите).")
print("   При k=25-50 — зображення вже впізнаване, але деталі втрачені.")
print("   При k=100 — якість близька до оригіналу.")
print("   При k=200 — відмінності від оригіналу практично непомітні.")