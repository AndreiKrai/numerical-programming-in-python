import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Простий DataFrame 5x5
data = {
    'зріст':  [170, 180, 160, 175, 165],
    'вага':   [70,  90,  55,  80,  60],
    'вік':    [25,  35,  22,  40,  28],
    'оцінка': [85,  75,  95,  70,  88],
}
target = [1, 0, 1, 0, 1]  # 1 = "спортсмен", 0 = "не спортсмен"

df = pd.DataFrame(data)
print("--- Оригінальні дані ---")
print(df)

# Стандартизація
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_standardized['target'] = target

print("\n--- Стандартизовані дані ---")
print(df_standardized)

# Pairplot
sns.pairplot(df_standardized, hue='target', diag_kind='hist')
plt.suptitle("Pairplot стандартизованих даних", y=1.02)
# plt.show()

# Візуалізуйте отримані матриці відстаней:
from sklearn.metrics import pairwise_distances

df_features = df_standardized.drop(columns=['target'])
metrics = ['cityblock', 'cosine', 'euclidean']

fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

for i, metric in enumerate(metrics):
    dist_matrix = pairwise_distances(df_features, metric=metric)
    sns.heatmap(dist_matrix, ax=axes[i], cmap='viridis', annot=True, fmt='.2f')
    axes[i].set_title(f'Відстані ({metric})')

plt.tight_layout()
plt.show()