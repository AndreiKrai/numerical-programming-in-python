from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt  # Library for plotting
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_genetic import GASearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn.metrics import f1_score

# Завантаження набору даних
data = load_breast_cancer()

# Ознаки
X = data.data

# Цільова змінна (0 - злоякісна, 1 - доброякісна)
y = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 2. Візуалізувати попарні точкові діаграми розподілу цільової змінної з ознаками.
# fig, axes = plt.subplots(6, 5)
# axes = axes.flatten()
# for i in range(X.shape[1]):
#     ax = axes[i]
#     ax.scatter(X[:, i], y, c=y, cmap='bwr', alpha=0.5)
#     ax.set_title(data.feature_names[i], fontsize=7)
# plt.show()

# 3. Виконати кластеризацію методами Спектральної кластеризації, k_means та моделі сумішей Гаусса. Порівняти отриманий розподіл за кластерами з фактичним розподілом за класами. Пояснити результати.
# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
print(confusion_matrix(y, kmeans_labels))
# Spectral Clustering
spectral = SpectralClustering(n_clusters=2,affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(X_scaled)
print(confusion_matrix(y, spectral_labels))
# Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
print(confusion_matrix(y, gmm_labels))

# ============================================================
# ВИСНОВКИ
# ============================================================
#
# 1. ДАТАСЕТ:
#    - Breast Cancer Wisconsin: 569 зразків, 30 ознак
#    - 212 злоякісних (0), 357 доброякісних (1)
#    - Класи перекриваються у просторі ознак — немає чіткої межі
#
# 2. ВІЗУАЛІЗАЦІЯ (крок 2):
#    - На scatter plots видно що більшість ознак частково розділяють класи
#    - Але жодна ознака окремо не дає повного розділення
#    - Це означає що кластеризація буде складною задачею
#
# 3. РЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ:
#
#    K-Means:           [[130  82] [ 1 356]] — точність ~85%
#    Spectral Clustering: [[182  30] [ 6 351]] — точність ~94%
#    Gaussian Mixture:  [[196  16] [18 339]] — точність ~94%
#
# 4. ПОРІВНЯННЯ МЕТОДІВ:
#    - K-Means найгірший: шукає сферичні кластери, погано справляється
#      з даними що перекриваються. Пропустив 82 злоякісних пухлини.
#    - Spectral Clustering кращий: будує граф сусідів і знаходить
#      складніші межі між кластерами.
#    - GMM найкращий по злоякісних (лише 16 пропущено): моделює кожен
#      кластер як еліпс з різною формою, що відповідає реальним даним.
#
# 5. ПРАКТИЧНИЙ ВИСНОВОК:
#    У медичній діагностиці пропустити злоякісну пухлину (false negative)
#    критично небезпечно. GMM пропустив лише 16 злоякісних vs 82 у K-Means.
#    Для таких задач GMM або Spectral Clustering переважають K-Means.
# ============================================================
# 4. Виконати зменшення розмірності даних за допомогою метода PCA.
pca = PCA(n_components=3)
vectors_2d = pca.fit_transform(X_scaled)
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {vectors_2d.shape}")
print(f"vectors_2d: {vectors_2d[:5]}")

plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=y, cmap='bwr', alpha=0.5)
plt.title("PCA of Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
print(pca.explained_variance_ratio_)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], vectors_2d[:, 2], c=y, cmap='bwr', alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()

# 6. Виконати класифікацію методом логістичної регресії LogisticRegression з бібліотеки sklearn.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# 7. Logistic Regression with different solvers
solvers = ['lbfgs', 'saga', 'liblinear', 'newton-cg']
for solver in solvers:
    m = LogisticRegression(solver=solver, max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{solver}: {acc:.4f}")

# 8. Виконати класифікацію методом логістичної регресії із оптимізацією параметрів Генетичним алгоритмом.
param_grid = {
    'C': Continuous(0.01, 10),
    'solver': Categorical(['lbfgs', 'saga', 'liblinear']),
    'max_iter': Integer(100, 1000)
}

cv = StratifiedKFold(n_splits=5)
ga_search = GASearchCV(
    estimator=LogisticRegression(),
    cv=cv,
    param_grid=param_grid,
    n_jobs=-1,
    verbose=True,
    generations=10
)
ga_search.fit(X_train, y_train)

print("Найкращі параметри:", ga_search.best_params_)
print("Точність:", ga_search.score(X_test, y_test))

# 9. Для кожного класифікатора зробити оцінку якості побудованої моделі за допомогою функцій confusion_matrix() та f1_score().

# Класифікатор 1 — дефолтний
print("=== LogReg дефолт ===")
print(confusion_matrix(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# Класифікатор 2 — GA оптимізований
ga_pred = ga_search.predict(X_test)
print("=== LogReg GA ===")
print(confusion_matrix(y_test, ga_pred))
print("F1:", f1_score(y_test, ga_pred))

# ============================================================
# 10. ЗАГАЛЬНІ ВИСНОВКИ
# ============================================================
#
# Ми порівняли два підходи: кластеризація (без вчителя) і класифікація (з вчителем).
#
# КЛАСТЕРИЗАЦІЯ (без міток y):
#   - K-Means:  ~85% — найгірший, бо шукає круглі кластери
#   - Spectral: ~94% — кращий, знаходить складні межі
#   - GMM:      ~94% — кращий, моделює еліптичні кластери
#
# КЛАСИФІКАЦІЯ (з мітками y):
#   - LogReg дефолт: 97.4%, F1=0.979
#   - LogReg GA:     97.4%, F1=0.979
#
# ВИСНОВОК 1: Класифікація краща за кластеризацію (~97% vs ~94%)
#   — бо модель знає правильні відповіді під час навчання.
#
# ВИСНОВОК 2: Генетичний алгоритм не покращив точність на цьому датасеті
#   — бо датасет невеликий і логрегресія вже близька до свого максимуму.
#   На великих і складних датасетах GA дає суттєвий приріст.
#
# ВИСНОВОК 3: Всі солвери (lbfgs, saga, liblinear, newton-cg) дали однаковий
#   результат — задача достатньо проста для будь-якого методу оптимізації.
#
# НАЙКРАЩИЙ МЕТОД для breast cancer: LogisticRegression
#   — простий, швидкий, інтерпретований, F1=0.979
# ============================================================