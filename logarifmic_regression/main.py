import pandas as pd

# Load dataset
df = pd.read_csv("~/Downloads/weatherAUS.csv")
print("Shape:", df.shape)
print(df.head())

# 3.1 Drop features with too many missing values (threshold: >40%)
# If a column has more than 40% NaN — it's unreliable to fill, better to remove
missing_pct = df.isnull().mean() * 100            # percentage of NaN per column
high_missing = missing_pct[missing_pct > 40]      # filter columns above threshold
print(f"\nFeatures with >40% missing values:\n{high_missing.round(1)}")
df = df.drop(columns=high_missing.index)          # remove those columns
print(f"Shape after dropping high-missing columns: {df.shape}")

# =============================================================================
# 3.3 Convert Date to datetime, create Year and Month columns
# =============================================================================
# Date is a string like "2008-12-01" — model needs numbers, not strings.
# We parse it into datetime and extract numeric features that capture time patterns.
df["Date"] = pd.to_datetime(df["Date"])           # parse string → datetime object
df["Year"] = df["Date"].dt.year                   # e.g. 2008, 2009... (int64)
df["Month"] = df["Date"].dt.month                 # 1-12 (captures seasonality)
df["Day"] = df["Date"].dt.day                     # 1-31
df = df.drop(columns=["Date"])                    # drop original — no longer needed

print(f"\nDate processed → new columns: Year, Month, Day")
print(f"Year dtype: {df['Year'].dtype}, Month dtype: {df['Month'].dtype}")

# Separate features (X) and target (y)
y = df["RainTomorrow"]                            # target: "Yes"/"No" — will it rain tomorrow?
X = df.drop(columns=["RainTomorrow"])             # all other columns = input features

# Drop rows where target is NaN — can't train without a label
mask = y.notna()
X = X[mask]
y = y[mask]

# =============================================================================
# 3.5 Split by year: test = last year, train = all other years
# =============================================================================
# Why: time-based split is more realistic than random split for time-series data.
# Random split leaks future info into training (e.g. train on 2016, test on 2014).
# Using last year as test simulates real prediction: train on past → predict future.
max_year = X["Year"].max()                        # find the latest year in data
test_mask = X["Year"] == max_year                 # True for rows with max year
train_mask = ~test_mask                           # all other years → train

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\n--- 3.5 Time-based split ---")
print(f"Test year: {max_year}")
print(f"Train: {X_train.shape[0]} rows (years {X_train['Year'].min()}-{X_train['Year'].max()})")
print(f"Test:  {X_test.shape[0]} rows (year {max_year})")
print(f"\nTarget distribution (train):\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\nTarget distribution (test):\n{y_test.value_counts(normalize=True).round(3)}")

# =============================================================================
# 3.2 Create subsets: numeric and categorical features
# =============================================================================
# Why: numeric and categorical features require DIFFERENT preprocessing:
#   - Numeric: fill NaN with median/mean, optionally scale
#   - Categorical: fill NaN with mode, then encode (e.g. one-hot or label encoding)
# We split them now to apply appropriate transformations to each group separately.

# Select numeric columns (int, float) from training set
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

# Select categorical columns (object, category) from training set
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\n--- 3.2 Feature subsets ---")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Create numeric and categorical subsets for train and test
X_train_num = X_train[numeric_cols]               # numeric part of train
X_test_num = X_test[numeric_cols]                 # numeric part of test
X_train_cat = X_train[categorical_cols]           # categorical part of train
X_test_cat = X_test[categorical_cols]             # categorical part of test

print(f"\nX_train_num shape: {X_train_num.shape}")
print(f"X_train_cat shape: {X_train_cat.shape}")
print(f"\nSample numeric:\n{X_train_num.head(3)}")
print(f"\nSample categorical:\n{X_train_cat.head(3)}")

# =============================================================================
# 3.4 Move Year → numeric, Month → categorical
# =============================================================================
# Year is ordinal (2008 < 2015 matters) → numeric
# Month is cyclical (after 12 goes 1, not linear) → categorical (will be one-hot encoded)

# Ensure Year is in numeric
if "Year" in categorical_cols:
    categorical_cols.remove("Year")
    numeric_cols.append("Year")

# Move Month from numeric to categorical
if "Month" in numeric_cols:
    numeric_cols.remove("Month")
    categorical_cols.append("Month")

# Rebuild subsets with corrected column lists
X_train_num = X_train[numeric_cols]
X_test_num = X_test[numeric_cols]
X_train_cat = X_train[categorical_cols]
X_test_cat = X_test[categorical_cols]

print("\n--- 3.4 Year → numeric, Month → categorical ---")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# =============================================================================
# 4. Fill missing values using SimpleImputer
# =============================================================================
# Why: most ML models cannot handle NaN. We must fill gaps before training.
# Important: fit imputer ONLY on train data, then transform both train and test.
#   fit on test = data leakage (test statistics leak into training process).
from sklearn.impute import SimpleImputer
import numpy as np

# 4.1 Numeric features — fill NaN with median
# Why median: robust to outliers (unlike mean which is skewed by extreme values)
num_imputer = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train_num),        # fit on train + transform
    columns=numeric_cols,
    index=X_train_num.index
)
X_test_num = pd.DataFrame(
    num_imputer.transform(X_test_num),             # only transform (using train's medians)
    columns=numeric_cols,
    index=X_test_num.index
)

# 4.2 Categorical features — fill NaN with most_frequent (mode)
# Why mode: most common value is the safest guess for a category
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_train_cat),        # fit on train + transform
    columns=categorical_cols,
    index=X_train_cat.index
)
X_test_cat = pd.DataFrame(
    cat_imputer.transform(X_test_cat),             # only transform (using train's modes)
    columns=categorical_cols,
    index=X_test_cat.index
)

print(f"\n--- 4. Imputation done ---")
print(f"NaN in X_train_num: {X_train_num.isnull().sum().sum()}")
print(f"NaN in X_train_cat: {X_train_cat.isnull().sum().sum()}")
print(f"NaN in X_test_num:  {X_test_num.isnull().sum().sum()}")
print(f"NaN in X_test_cat:  {X_test_cat.isnull().sum().sum()}")

# =============================================================================
# 5. Normalize numeric features using StandardScaler
# =============================================================================
# Why: features have different scales (e.g. Pressure ~1000, Humidity ~0-100).
# Without scaling, features with larger values dominate distance-based models.
# StandardScaler transforms each feature to mean=0, std=1.
# Note: Decision Trees don't require scaling, but it's good practice and
#        helps if we later compare with other models (SVM, KNN, Logistic Regression).
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_num),             # fit (learn mean/std from train) + transform
    columns=numeric_cols,
    index=X_train_num.index
)
X_test_num_scaled = pd.DataFrame(
    scaler.transform(X_test_num),                  # only transform (using train's mean/std)
    columns=numeric_cols,
    index=X_test_num.index
)

print(f"\n--- 5. Scaling done ---")
print(f"Before scaling — train mean:\n{X_train_num.mean().round(2)}")
print(f"\nAfter scaling — train mean:\n{X_train_num_scaled.mean().round(4)}")
print(f"\nAfter scaling — train std:\n{X_train_num_scaled.std().round(4)}")

# =============================================================================
# 6. Encode categorical features using OneHotEncoder
# =============================================================================
# Why: model needs numbers. Categorical values like "Sydney", "NW", "Yes"
# must be converted to numeric representation.
# OneHotEncoder creates a binary column for each unique category value:
#   WindDir: ["N", "S", "E"] → WindDir_N=1/0, WindDir_S=1/0, WindDir_E=1/0
# handle_unknown="ignore" — if test has a category not seen in train, fill with zeros
# sparse_output=False — return dense array (easier to work with as DataFrame)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat_encoded = pd.DataFrame(
    encoder.fit_transform(X_train_cat),            # fit on train (learn categories) + transform
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train_cat.index
)
X_test_cat_encoded = pd.DataFrame(
    encoder.transform(X_test_cat),                 # only transform (using train's categories)
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test_cat.index
)

print(f"\n--- 6. One-Hot Encoding done ---")
print(f"Categorical columns before: {len(categorical_cols)}")
print(f"Columns after encoding: {X_train_cat_encoded.shape[1]}")
print(f"\nEncoded column names (first 15): {list(X_train_cat_encoded.columns[:15])}")
print(f"\nSample encoded:\n{X_train_cat_encoded.head(3)}")

# =============================================================================
# 7. Combine numeric + categorical and build Logistic Regression model
# =============================================================================
# Merge scaled numeric features with one-hot encoded categorical features
# into a single DataFrame — ready for model training.
X_train_final = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_test_final = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)

print(f"\n--- 7. Final dataset ---")
print(f"X_train_final shape: {X_train_final.shape}")
print(f"X_test_final shape:  {X_test_final.shape}")

# Build Logistic Regression model
# max_iter=1000 — increase iterations to ensure convergence with many features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_final, y_train)                  # train on combined features

# Predict on test set
y_pred = model.predict(X_test_final)

# Evaluate
print(f"\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# =============================================================================
# 8. Comparison with baseline Logistic Regression (without proper preprocessing)
# =============================================================================
# In the previous lesson ("Логістична регресія. Оцінка якості класифікації"),
# a baseline model was trained without:
#   - Removing high-missing columns
#   - Time-based split (used random split instead)
#   - Proper date feature engineering (Year, Month as separate features)
#   - Moving Month to categorical (one-hot encoded)
#
# Baseline results (random split, minimal preprocessing):
#              precision    recall  f1-score   support
#          No      0.92      0.79      0.85     22098
#         Yes      0.51      0.76      0.61      6341
#    accuracy                          0.79     28439
#
# Our improved model uses:
#   ✅ Removed columns with >40% NaN (less noise)
#   ✅ Time-based split (more realistic evaluation)
#   ✅ Date → Year + Month features (captures seasonality)
#   ✅ Month as categorical (one-hot) — no false ordinality
#   ✅ Proper imputation (median for numeric, mode for categorical)
#   ✅ StandardScaler for numeric features
#   ✅ OneHotEncoder with handle_unknown="ignore"

print("\n--- Висновки ---")
print(f"""
Порівняння з базовою моделлю логістичної регресії
(з розділу «Практика застосування логістичної регресії. Оцінювання точності моделі»):

                    Базова модель          Поточна модель
                    (random split)         (time-based split)
                    ──────────────         ──────────────────
Accuracy:           0.79                   0.85
Precision (No):     0.92                   0.87
Recall (No):        0.79                   0.95
F1 (No):            0.85                   0.91
Precision (Yes):    0.51                   0.72
Recall (Yes):       0.76                   0.45
F1 (Yes):           0.61                   0.55

Аналіз результатів:
1. Accuracy покращилась: 0.79 → 0.85 (+6%). Модель загалом точніша.

2. Precision (Yes) значно зросла: 0.51 → 0.72 (+21%).
   Коли модель каже "буде дощ" — вона правильна у 72% випадків (було 51%).
   Менше хибних тривог.

3. Recall (Yes) знизився: 0.76 → 0.45 (-31%).
   Модель пропускає більше дощових днів — з 76% знаходить лише 45%.
   Це trade-off: модель стала "обережнішою" у прогнозах дощу.

4. F1 (No) покращився: 0.85 → 0.91. Клас "No" прогнозується значно краще.

5. F1 (Yes) трохи знизився: 0.61 → 0.55. Баланс precision/recall для "Yes"
   змістився в бік precision (менше хибних тривог, але більше пропущених дощів).

Причини відмінностей:
- Time-based split — складніша задача (модель не бачила жодного дня з тестового року),
  тому accuracy 0.85 є більш чесною оцінкою реальної якості ніж 0.79 з random split.
- Базова модель мала високий Recall(Yes)=0.76 але низький Precision(Yes)=0.51,
  тобто часто "кричала дощ" помилково. Наша модель навпаки — рідше прогнозує дощ,
  але коли прогнозує — частіше правильно.
- Покращений препроцесинг (нормалізація, one-hot Month, видалення шумних колонок)
  допоміг моделі краще розрізняти "No" (F1: 0.85→0.91).

Загальний висновок: поточна модель з правильним препроцесингом та time-based split
показує вищу загальну accuracy (0.85 vs 0.79) і значно кращу precision для класу "Yes"
(0.72 vs 0.51), хоча recall для "Yes" нижчий. Для практичного використання це означає
більш надійні прогнози дощу з меншою кількістю хибних тривог.
""")
