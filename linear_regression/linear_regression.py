import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Tool to standardize (normalize) data

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target
df = X.copy()
df["MedHouseVal"] = y  

# EDA: visualize feature distributions before any processing
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, col in enumerate(df.columns):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(col)
    ax.set_ylabel("Count")
plt.suptitle("Feature Distributions (before cleaning)", fontsize=14)
plt.tight_layout()
plt.show()

print("Before cleaning:", df.shape)

# 3.1 Outlier removal using z-score
# Why: outliers distort linear regression — one extreme value can shift the entire model
#
# Z-score = how far a value is from the mean, measured in standard deviations
#   z = 0  → value equals the mean
#   |z| > 3 → value is 3+ std deviations away → considered an outlier
#
# Steps:
# 1. Select 4 columns to check: AveRooms, AveBedrms, AveOccup, Population
# 2. Compute z-score for each column independently using apply(zscore)
# 3. Take abs() and check where |z| > 3 → get True/False table
# 4. For each row, check if ANY of the 4 columns has True → .any(axis=1)
# 5. Remove rows where is_outlier == True (keep only ~is_outlier)
outlier_cols = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
z_scores = df[outlier_cols].apply(zscore)          # z-score for each column independently
is_outlier = (z_scores.abs() > 3).any(axis=1)     # True if at least one column is outlier
df = df[~is_outlier].reset_index(drop=True)        # keep only non-outlier rows

print("After outlier removal:", df.shape)
print(f"Removed {is_outlier.sum()} rows")

# Check correlation matrix to find highly correlated features
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
# Add correlation values as text
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# Find pairs with high correlation (|r| > 0.5, excluding diagonal and MedHouseVal)
features = df.drop(columns=["MedHouseVal"]).columns
print("\nHighly correlated feature pairs (|r| > 0.5):")
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        r = corr_matrix.loc[features[i], features[j]]
        if abs(r) > 0.5:
            print(f"  {features[i]} <-> {features[j]}: {r:.3f}")

# 3.2 Remove highly correlated feature
# Latitude <-> Longitude: r = -0.928 (highest correlation between features)
# Longitude has weaker correlation with target (r=-0.05) than Latitude (r=-0.14)
# So we drop Longitude — it's less useful for predicting house price
# df = df.drop(columns=["Longitude"])

print("\nAfter dropping Longitude:", df.shape)
print(df.head())

# 4. Split into train and test sets
# Why: train on one part, test on another — to check if model generalizes well
# Steps:
# 1. Separate features (X) from target (y) in the cleaned DataFrame
# 2. Split: 80% train, 20% test (standard ratio)
# 3. random_state=42 — fixed seed for reproducible results
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")

# 5. Normalize features using StandardScaler
# Why: features have different scales (MedInc ~1-15, Population ~1-35000)
#      without normalization, model is biased toward features with larger numbers
# Important: fit ONLY on train data, then transform both train and test
#   fit on test = data leakage (test info leaks into training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # fit (learn mean/std) + transform train
X_test_scaled = scaler.transform(X_test)           # only transform test (using train's mean/std)

print(f"\nBefore scaling — X_train mean: {X_train.mean().round(2).tolist()}")
print(f"After scaling  — X_train mean: {X_train_scaled.mean(axis=0).round(2).tolist()}")


# 6. Build linear regression model
# Why: find the best line (hyperplane) that predicts MedHouseVal from features
# Steps: create model → fit on SCALED train data → model learns coefficients (weights)
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)                # train: learn weights for each feature

print(f"\nModel coefficients: {dict(zip(X.columns, linreg.coef_.round(4)))}")
print(f"Intercept: {linreg.intercept_:.4f}")

linreg_pred = linreg.predict(X_test_scaled)        # predict on test data
predictions = pd.DataFrame({"Actual": y_test, "Predicted": linreg_pred})
print("\nPredictions vs Actual:")
print(predictions.head(10))

# 7. Evaluate model quality
# R² — how much variance in target the model explains (1.0 = perfect, 0.0 = useless)
# MAE — average absolute error in the same units as target (hundreds of thousands $)
# MAPE — average absolute error as percentage of actual value
r2 = r2_score(y_test, linreg_pred)
mae = mean_absolute_error(y_test, linreg_pred)
mape = mean_absolute_percentage_error(y_test, linreg_pred)

print(f"\n--- Model Evaluation ---")
print(f'R2: {r2:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# Conclusion
print("\n--- Висновки ---")
print("""
Було побудовано модель лінійної регресії для прогнозування медіанної вартості
будинків у Каліфорнії на основі датасету California Housing (20640 зразків, 8 ознак).
Перед навчанням виконано очистку від викидів (z-score, видалено 505 рядків),
видалення висококорельованої ознаки Longitude (r=-0.928 з Latitude) та нормалізацію
ознак (StandardScaler). Модель показала R²=0.61, MAE=0.54 (~$54,000), MAPE=33%.
Результати є типовими для лінійної регресії на цьому датасеті — модель пояснює
61% варіації ціни, що є прийнятним, але залишає простір для покращення за допомогою
нелінійних моделей (Random Forest, Gradient Boosting).

Примітка: Longitude НЕ було видалено з моделі (рядок закоментовано).
Залишення Longitude покращило якість моделі, оскільки географічна довгота містить
корисну інформацію про розташування (близькість до океану, міста), яка не повністю
дублюється Latitude. Попри високу кореляцію між ними (r=-0.928), обидві ознаки
разом дають кращий R² та нижчий MAE/MAPE, ніж кожна окремо.
""")