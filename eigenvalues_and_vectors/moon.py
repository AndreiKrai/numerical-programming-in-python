# === Example: Spectral Clustering on make_moons dataset ===

from sklearn.datasets import make_moons
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# --- Step 1: Generate synthetic data (2 classes, half-moon shape) ---
X, y = make_moons(n_samples=500, noise=0.07, random_state=42)

df = pd.DataFrame(X, columns=["x1", "x2"])
df['true_label'] = y

print(df.head())

# --- Step 2: Statistical description ---
print("\nСтатистика:")
print(df.describe())

# --- Step 3: Visualize distribution by classes ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="x1", y="x2", hue="true_label", palette="Set1")
plt.title("Справжні класи (true_label)")
plt.show()

# --- Step 4: Standardization ---
scaler = StandardScaler()
X_scaled = scaler.set_output(transform='pandas').fit_transform(X)
print("\nСтандартизовані дані:")
print(X_scaled.describe())

# --- Step 5: Spectral Clustering ---
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
predicted_labels = spectral.fit_predict(X_scaled)

# --- Step 6: Add predictions to DataFrame ---
df["predicted_label"] = predicted_labels
print("\nДатафрейм з передбаченнями:")
print(df.head())

# --- Step 7: Compare clusters with true classes (Confusion Matrix) ---
cm = confusion_matrix(df["true_label"], df["predicted_label"])
print("\nМатриця невідповідностей (confusion matrix):")
print(cm)

# Visualize confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Матриця невідповідностей")
plt.show()

# --- Step 8: Visualize clusters ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="x1", y="x2", hue="predicted_label", palette="Set2")
plt.title("Кластери, знайдені Spectral Clustering")
plt.show()
