import numpy as np

def are_vectors_linearly_independent(vectors):
    # Створення розширеної матриці з векторів
    matrix = np.array(vectors).T

    # Ранг матриці
    rank_matrix = np.linalg.matrix_rank(matrix)

    # Кількість векторів
    num_vectors = len(vectors)

    # Вектори лінійно незалежні, якщо ранг матриці рівний кількості векторів
    return rank_matrix == num_vectors

# Приклад використання
vectors1 = np.array([1, 2, 3])
vectors2 = np.array([-2, 1, -1])
vectors3 = np.array([3, 2, -1])

# Приклад використання
#vectors1 = np.array([1, 2, -3])
#vectors2 = np.array([-1, 2, 4])
#vectors3 = np.array([1, 6, -2])

# Перевірка лінійної незалежності векторів
result = are_vectors_linearly_independent([vectors1, vectors2, vectors3])

if result:
    print("Вектори лінійно незалежні.")
else:
    print("Вектори лінійно залежні.")
