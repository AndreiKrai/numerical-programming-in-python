import numpy as np
import matplotlib.pyplot as plt

# Координати першого вектора
v1 = np.array([2, 3])

# Координати другого вектора
v2 = np.array([-1, 5])

# Mark: Створення графіку
plt.figure()

# Зображення першого вектора
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='b', label='Вектор v1')

# Зображення другого вектора
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='r', label='Вектор v2')

# Налаштування осей
plt.xlim(-2, 3)
plt.ylim(-1, 6)
plt.xlabel('X')
plt.ylabel('Y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

# Відображення легенди
plt.legend()

# Показ графіка
plt.show()

# Mark: Обчислення скалярного добутку
# dot_product = np.dot(v1, v2)
dot_product = v1 @ v2

# Обчислення довжин векторів
length_v1 = np.linalg.norm(v1)
length_v2 = np.linalg.norm(v2)

# Обчислення косинуса кута між векторами
cos_theta = dot_product / (length_v1 * length_v2)

# Обчислення кута в радіанах
theta_radians = np.arccos(cos_theta)

# Переведення кута в градуси
theta_degrees = np.degrees(theta_radians)

print("Кут між векторами:", theta_degrees, "градусів")