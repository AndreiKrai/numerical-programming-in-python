import pandas as pd  # Library for working with tables (DataFrames)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
df=pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
print(cancer.keys())
print(cancer['DESCR'])
print("----shape",cancer['data'].shape)

# 3. Виведіть інформацію про дані
print("\n--- DataFrame Info ---")
df.info()

# 4. Виведіть описові статистики
print("\n--- DataFrame Description ---")
print(df.describe())

# 5. Стандартизуйте дані:

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("\n--- Стандартизовані дані (перші 5 рядків) ---")
print(df_standardized.head())
print("\n--- Середнє (має бути ≈0) ---")
print(df_standardized.mean().round(2))
print("\n--- Стд. відхилення (має бути ≈1) ---")
print(df_standardized.std().round(2))

# 6. Побудуйте точкові діаграми:
df_standardized['target'] = cancer['target']
sns.pairplot(df_standardized, hue='target', diag_kind='hist')
plt.suptitle("Pairplot стандартизованих даних", y=1.02)
plt.show()

# 7. Обчисліть матриці відстаней:
from sklearn.metrics import pairwise_distances

# Видаляємо target для обчислення відстаней
df_features = df_standardized.drop(columns=['target'])

metrics = ['cityblock', 'cosine', 'euclidean']
# 'l1', 'manhattan' the same as 'cityblock'

for metric in metrics:
    dist_matrix = pairwise_distances(df_features, metric=metric)
    dist_df = pd.DataFrame(dist_matrix)
    print(f"\n--- Матриця відстаней ({metric}) [{dist_df.shape}] ---")
    print(dist_df.iloc[:5, :5])  # перші 5x5 для наочності

# 8. Візуалізуйте отримані матриці:
fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

for i, metric in enumerate(metrics):
    dist_matrix = pairwise_distances(df_features, metric=metric)
    sns.heatmap(dist_matrix, ax=axes[i], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[i].set_title(f'Відстані ({metric})')

plt.tight_layout()
plt.show()

# 9. Висновок:
#
# Ми завантажили набір даних Breast Cancer (569 зразків, 30 ознак),
# стандартизували його та обчислили матриці відстаней трьома метриками.
#
# Cityblock (Manhattan) — рахує відстань "по вулицях", як у місті.
# Euclidean — рахує пряму відстань між точками.
# Cosine — порівнює напрямок векторів, а не їх довжину.
#
# На heatmap видно, що всі метрики виділяють схожі групи зразків.
# Це означає, що дані добре структуровані і класи розрізняються.
# Стандартизація була необхідна, бо ознаки мали різні масштаби.