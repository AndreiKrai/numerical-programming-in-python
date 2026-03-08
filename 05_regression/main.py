import numpy as np

# 1. Згенеруйте дані:
np.random.seed(42)  # для відтворюваності результатів
x1 = np.random.rand(100)
x2 = np.random.rand(100)

print("x1:", x1[:5])
print("x2:", x2[:5])

# 2. Реалізуйте функцію polynomial:
# Це штучне правило, яке зв'язує x1, x2 з результатом y.
# В реальності формула невідома — замість неї є реальні дані.
# Модель регресії спробує сама знайти це правило.
def polynomial(x1, x2):
    return 3*x1**2 + 2*x2 + 5*x1*x2 + 1

y = polynomial(x1, x2)
print("y:", y[:5])

# 3. Згенеруйте додаткові ознаки (PolynomialFeatures):
# PolynomialFeatures створює нові стовпці (x1², x1·x2, x2² і т.д.)
# з існуючих x1, x2 — щоб лінійна регресія могла знайти нелінійну залежність.
# Пробуємо степені 1-5, щоб порівняти, який дає найкращий результат.
from sklearn.preprocessing import PolynomialFeatures

X = np.column_stack([x1, x2])  # об'єднуємо x1, x2 в матрицю (100, 2)

for degree in range(1, 6):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    print(f"\nСтепінь {degree}: {X_poly.shape[1]} ознак")
    print(f"Назви: {poly.get_feature_names_out()}")

# 4. Градієнтний спуск для поліноміальної регресії:
# Модель шукає коефіцієнти (weights), які мінімізують помилку.
# На кожному кроці вона трохи підправляє коефіцієнти в бік меншої помилки.
def polynomial_regression_gradient_descent(X, y, degree=2, lr=0.01, epochs=5000):
    # Створюємо поліноміальні ознаки
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    n_samples, n_features = X_poly.shape
    weights = np.zeros(n_features)  # початкові коефіцієнти = 0

    for epoch in range(epochs):
        # Передбачення: y_pred = X * weights
        y_pred = X_poly.dot(weights)
        # Помилка: різниця між передбаченням і реальним y
        error = y_pred - y
        # Градієнт: напрямок, куди рухатися щоб зменшити помилку
        gradient = (2 / n_samples) * X_poly.T.dot(error)
        # Оновлюємо коефіцієнти
        weights = weights - lr * gradient

    return weights, poly

# Тестуємо градієнтний спуск
weights, poly = polynomial_regression_gradient_descent(X, y, degree=2, lr=0.1, epochs=30000)
print("\n--- Градієнтний спуск (степінь 2) ---")
print(f"Знайдені коефіцієнти: {weights.round(4)}")
print(f"Очікувані коефіцієнти: [1, 0, 2, 3, 5, 0]")
print(f"Назви ознак: {poly.get_feature_names_out()}")

# 5. SGD (Stochastic Gradient Descent) — стохастичний градієнтний спуск:
# Різниця: звичайний GD рахує градієнт по ВСІХ 100 зразках,
# а SGD — по ОДНОМУ випадковому зразку за крок. Це швидше, але "шумніше".
def polynomial_regression_SGD(X, y, degree=2, lr=0.01, epochs=1000):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    n_samples, n_features = X_poly.shape
    weights = np.zeros(n_features)

    for epoch in range(epochs):
        for i in range(n_samples):
            # Беремо ОДИН випадковий зразок
            idx = np.random.randint(n_samples)
            x_i = X_poly[idx]
            y_i = y[idx]
            # Градієнт по одному зразку
            y_pred = x_i.dot(weights)
            error = y_pred - y_i
            gradient = 2 * x_i * error
            weights = weights - lr * gradient

    return weights, poly

# Тестуємо SGD
weights_sgd, poly_sgd = polynomial_regression_SGD(X, y, degree=2, lr=0.1, epochs=3000)
print("\n--- SGD (степінь 2) ---")
print(f"Знайдені коефіцієнти: {weights_sgd.round(4)}")
print(f"Очікувані коефіцієнти: [1, 0, 2, 3, 5, 0]")

# 6. RMSProp — адаптивний градієнтний спуск:
# Проблема GD/SGD: один lr для всіх weights. Якщо одні ознаки великі, а інші маленькі —
# один lr не підходить для всіх. RMSProp автоматично підлаштовує lr для кожного weight.
def polynomial_regression_rmsprop(X, y, degree=2, lr=0.1, epochs=5000, beta=0.9, epsilon=1e-8):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    n_samples, n_features = X_poly.shape
    weights = np.zeros(n_features)
    # Накопичувач квадратів градієнтів (для адаптивного lr)
    s = np.zeros(n_features)

    for epoch in range(epochs):
        y_pred = X_poly.dot(weights)
        error = y_pred - y
        gradient = (2 / n_samples) * X_poly.T.dot(error)

        # Накопичуємо ковзне середнє квадратів градієнтів
        s = beta * s + (1 - beta) * gradient**2
        # Оновлюємо weights з адаптивним lr
        weights = weights - lr * gradient / (np.sqrt(s) + epsilon)

    return weights, poly

# Тестуємо RMSProp
weights_rms, poly_rms = polynomial_regression_rmsprop(X, y, degree=2, lr=0.1, epochs=1500)
print("\n--- RMSProp (степінь 2) ---")
print(f"Знайдені коефіцієнти: {weights_rms.round(4)}")
print(f"Очікувані коефіцієнти: [1, 0, 2, 3, 5, 0]")

# 7. Adam — комбінація SGD з моментом + RMSProp:
# Adam = найпопулярніший оптимізатор. Він поєднує дві ідеї:
# - m (момент) — "інерція", пам'ятає напрямок руху (як у SGD з моментом)
# - s (адаптивний lr) — підлаштовує lr для кожного weight (як у RMSProp)
def polynomial_regression_adam(X, y, degree=2, lr=0.001, epochs=5000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    n_samples, n_features = X_poly.shape
    weights = np.zeros(n_features)
    m = np.zeros(n_features)  # момент (середній градієнт)
    s = np.zeros(n_features)  # адаптивний lr (середній квадрат градієнта)

    for epoch in range(epochs):
        y_pred = X_poly.dot(weights)
        error = y_pred - y
        gradient = (2 / n_samples) * X_poly.T.dot(error)

        # Оновлюємо момент (напрямок руху)
        m = beta1 * m + (1 - beta1) * gradient
        # Оновлюємо адаптивний lr
        s = beta2 * s + (1 - beta2) * gradient**2

        # Корекція зміщення (на початку m і s занадто малі)
        m_hat = m / (1 - beta1**(epoch + 1))
        s_hat = s / (1 - beta2**(epoch + 1))

        # Оновлюємо weights
        weights = weights - lr * m_hat / (np.sqrt(s_hat) + epsilon)

    return weights, poly

# Тестуємо Adam
weights_adam, poly_adam = polynomial_regression_adam(X, y, degree=2, lr=0.1, epochs=5000)
print("\n--- Adam (степінь 2) ---")
print(f"Знайдені коефіцієнти: {weights_adam.round(4)}")
print(f"Очікувані коефіцієнти: [1, 0, 2, 3, 5, 0]")

# 8. Nadam — Adam + Nesterov момент:
# Nadam = Adam, але момент "дивиться вперед" (Nesterov).
# Замість руху за поточним моментом, він спочатку "стрибає" вперед,
# а потім коригує напрямок. Це дає швидшу збіжність.
def polynomial_regression_nadam(X, y, degree=2, lr=0.1, epochs=5000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    n_samples, n_features = X_poly.shape
    weights = np.zeros(n_features)
    m = np.zeros(n_features)
    s = np.zeros(n_features)

    for epoch in range(epochs):
        y_pred = X_poly.dot(weights)
        error = y_pred - y
        gradient = (2 / n_samples) * X_poly.T.dot(error)

        m = beta1 * m + (1 - beta1) * gradient
        s = beta2 * s + (1 - beta2) * gradient**2

        m_hat = m / (1 - beta1**(epoch + 1))
        s_hat = s / (1 - beta2**(epoch + 1))

        # Nadam: замість m_hat використовуємо "погляд вперед"
        # beta1 * m_hat — інерція + (1-beta1)*gradient — поточний градієнт
        m_nesterov = beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1**(epoch + 1))

        weights = weights - lr * m_nesterov / (np.sqrt(s_hat) + epsilon)

    return weights, poly

# Тестуємо Nadam
weights_nadam, poly_nadam = polynomial_regression_nadam(X, y, degree=2, lr=0.01, epochs=5000)
print("\n--- Nadam (степінь 2) ---")
print(f"Знайдені коефіцієнти: {weights_nadam.round(4)}")
print(f"Очікувані коефіцієнти: [1, 0, 2, 3, 5, 0]")

# 9. Порівняння часу роботи функцій:
# %timeit працює лише в Jupyter, тому використовуємо timeit з Python
import timeit

methods = {
    'GD':      lambda: polynomial_regression_gradient_descent(X, y, degree=2, lr=0.1, epochs=30000),
    'SGD':     lambda: polynomial_regression_SGD(X, y, degree=2, lr=0.1, epochs=3000),
    'RMSProp': lambda: polynomial_regression_rmsprop(X, y, degree=2, lr=0.1, epochs=1500),
    'Adam':    lambda: polynomial_regression_adam(X, y, degree=2, lr=0.1, epochs=5000),
    'Nadam':   lambda: polynomial_regression_nadam(X, y, degree=2, lr=0.01, epochs=5000),
}

print("\n--- Час роботи ---")
for name, func in methods.items():
    time = timeit.timeit(func, number=3) / 3  # середній час з 3 запусків
    print(f"{name:>8}: {time:.4f} сек")

# 10. Висновок:
# RMSProp — найшвидший (0.0065 сек), бо потребував лише 1500 epochs для точного результату.
# GD — помірний (0.0576 сек), потребував 30000 epochs, але кожен крок швидкий.
# Adam і Nadam — середні (0.03-0.04 сек), 5000 epochs, але кожен крок складніший.
# SGD — найповільніший (0.87 сек), бо має вкладений цикл по кожному зразку.
#
# Загалом: адаптивні методи (RMSProp, Adam, Nadam) потребують менше epochs,
# що компенсує складніші обчислення на кожному кроці.
# SGD повільний через цикл у Python, але в реальних бібліотеках він оптимізований.