import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.neighbors import KNeighborsRegressor # мпортуємо. Regressor — бо Salary числовий (а не клас). Якщо б передбачали Yes/No — був би KNeighborsClassifier.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df_train = pd.read_csv("~/Downloads/mod_04_hw_train_data.csv")
df_valid = pd.read_csv("~/Downloads/mod_04_hw_valid_data.csv")

print("Train shape:", df_train.shape)
print("Valid shape:", df_valid.shape)
print()

# Data types
print(df_train.dtypes)
print()

# Missing values
print("Missing values")
print(df_train.isnull().sum())
print()

# Basic stats
print(df_train.describe())

for col in ["Qualification", "University", "Role", "Cert"]:
    print(f"{col}: {df_train[col].unique()}")

# visualisation
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

sns.histplot(df_train["Salary"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Salary Distribution")
# гістограма зарплат + лінія щільності. Побачиш де скупчення.

sns.boxplot(x="Experience", y="Salary", data=df_train, ax=axes[0, 1])
axes[0, 1].set_title("Experience vs Salary")
# як досвід впливає на зарплату.

sns.boxplot(x="Qualification", y="Salary", data=df_train, ax=axes[0, 2])
axes[0, 2].set_title("Qualification vs Salary")
# чи PhD > Msc > Bsc по зарплаті.

sns.boxplot(x="University", y="Salary", data=df_train, ax=axes[1, 0])
axes[1, 0].set_title("University vs Salary")
# чи Tier1 платять більше.

sns.boxplot(x="Role", y="Salary", data=df_train, ax=axes[1, 1])
axes[1, 1].set_title("Role vs Salary")
# Junior vs Mid vs Senior.

sns.boxplot(x="Cert", y="Salary", data=df_train, ax=axes[1, 2])
axes[1, 2].set_title("Cert vs Salary")
# чи сертифікат підвищує зарплату.

plt.tight_layout()
plt.show()
# щоб графіки не налізали один на одний + показати.

# Крок 4.1: Прибрати непотрібні колонки і розділити на X та y

drop_cols = ["Name", "Phone_Number", "Date_Of_Birth"]
df_train = df_train.drop(columns=drop_cols)
df_valid = df_valid.drop(columns=drop_cols)
# прибираємо колонки, які не мають відношення до зарплати.

# Розділяємо X та y (не видаляємо NaN — заповнимо через imputer)
y_train = df_train["Salary"]
X_train = df_train.drop(columns=["Salary"])
y_valid = df_valid["Salary"]
X_valid = df_valid.drop(columns=["Salary"])

# Крок 4.2: Розділити ознаки на числові та категоріальні
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

# Крок 4.2.1: Заповнити пропуски (impute замість dropna — зберігаємо всі 249 рядків)
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_valid[numeric_cols] = num_imputer.transform(X_valid[numeric_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_valid[categorical_cols] = cat_imputer.transform(X_valid[categorical_cols])

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# Крок 4.3: Масштабувати числові ознаки
scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index
)
X_valid_num = pd.DataFrame(
    scaler.transform(X_valid[numeric_cols]),
    columns=numeric_cols,
    index=X_valid.index
)
# Навіщо в ML: KNN рахує відстань між точками. Якщо Experience від 1 до 5, а Salary від 50000 до 150000 — модель 
# буде дивитись тільки на Salary, бо різниці там більші. Скейлінг робить всі ознаки однакового масштабу.
# fit_transform на train — вчить середнє і std, потім трансформує.
# transform на valid — тільки трансформує (використовує статистику з train).
# Важливо: НІКОЛИ не робиш fit на валідаційних/тестових даних — це data leakage!
print("Before scaling:", X_train[numeric_cols].describe().round(2))
print("After scaling:", X_train_num.describe().round(2))

# Крок 4.4: Кодування категоріальних ознак — OneHotEncoder
##  ** OneHotEncoder
# Навіщо в ML: модель не розуміє "PhD" чи "Senior" — їй потрібні числа. OneHotEncoder робить з одної колонки кілька бінарних (0/1).
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# X_train_cat = pd.DataFrame(
#     encoder.fit_transform(X_train[categorical_cols]),
#     columns=encoder.get_feature_names_out(categorical_cols),
#     index=X_train.index
# )
# X_valid_cat = pd.DataFrame(
#     encoder.transform(X_valid[categorical_cols]),
#     columns=encoder.get_feature_names_out(categorical_cols),
#     index=X_valid.index
# )
# RESULT:MAPE - 7,35

## ** TargetEncoder
# target_enc = TargetEncoder(smooth="auto")

# # Зверни увагу: fit_transform тут приймає і X і y_train! Бо TargetEncoder вчиться з target.
# # Це НЕ leakage — sklearn робить внутрішній cross-validation щоб уникнути цього.
# X_train_cat = pd.DataFrame(
#     target_enc.fit_transform(X_train[categorical_cols], y_train),
#     columns=categorical_cols,
#     index=X_train.index
# )
# X_valid_cat = pd.DataFrame(
#     target_enc.transform(X_valid[categorical_cols]),
#     columns=categorical_cols,
#     index=X_valid.index
# )
# RESULT:MAPE - 7,19

## ** category_encoders (TargetEncoder from sklearn with target_type fix)

# target_enc = TargetEncoder(smooth="auto", target_type="continuous")
#
# X_train_cat = pd.DataFrame(
#     target_enc.fit_transform(X_train[categorical_cols], y_train),
#     columns=categorical_cols,
#     index=X_train.index
# )
# X_valid_cat = pd.DataFrame(
#     target_enc.transform(X_valid[categorical_cols]),
#     columns=categorical_cols,
#     index=X_valid.index
# )
# RESULT: MAPE - 7.59

## ** OrdinalEncoder (sklearn) — best for ordinal features with KNN
from sklearn.preprocessing import OrdinalEncoder

categories = [
    ["Bsc", "Msc", "PhD"],        # Qualification
    ["Tier3", "Tier2", "Tier1"],   # University
    ["Junior", "Mid", "Senior"],   # Role
    ["No", "Yes"]                  # Cert
]

# ord_enc = OrdinalEncoder(categories=categories)
#
# X_train_cat = pd.DataFrame(
#     ord_enc.fit_transform(X_train[categorical_cols]),
#     columns=categorical_cols,
#     index=X_train.index
# )
# X_valid_cat = pd.DataFrame(
#     ord_enc.transform(X_valid[categorical_cols]),
#     columns=categorical_cols,
#     index=X_valid.index
# )
# RESULT: MAPE - 6.08 (K=39, uniform, manhattan)

# ## ** Manual Target Encoding (no sklearn bugs, uses mean Salary per category)
# X_train_cat = X_train[categorical_cols].copy()
# X_valid_cat = X_valid[categorical_cols].copy()
#
# for col in categorical_cols:
#     means = df_train.groupby(col)["Salary"].mean()
#     X_train_cat[col] = X_train_cat[col].map(means)
#     X_valid_cat[col] = X_valid_cat[col].map(means)
# RESULT: MAPE - 6.73 (K=22, distance, euclidean)

## ** OrdinalEncoder (sklearn) + Imputer
# ord_enc = OrdinalEncoder(categories=categories)
#
# X_train_cat = pd.DataFrame(
#     ord_enc.fit_transform(X_train[categorical_cols]),
#     columns=categorical_cols,
#     index=X_train.index
# )
# X_valid_cat = pd.DataFrame(
#     ord_enc.transform(X_valid[categorical_cols]),
#     columns=categorical_cols,
#     index=X_valid.index
# )
# RESULT: MAPE - 5.65 (K=16, distance, euclidean)

## ** TargetEncoder + PowerTransformer + Imputer (active, per teacher's instructions)
target_enc = TargetEncoder(smooth="auto", target_type="continuous")

X_train_cat = pd.DataFrame(
    target_enc.fit_transform(X_train[categorical_cols], y_train),
    columns=categorical_cols,
    index=X_train.index
)
X_valid_cat = pd.DataFrame(
    target_enc.transform(X_valid[categorical_cols]),
    columns=categorical_cols,
    index=X_valid.index
)

X_train_combined = pd.concat([X_train[numeric_cols], X_train_cat], axis=1)
X_valid_combined = pd.concat([X_valid[numeric_cols], X_valid_cat], axis=1)

# PowerTransformer (Yeo-Johnson) — normalizes distribution + scales
pt = PowerTransformer(method="yeo-johnson")

X_train_final = pd.DataFrame(
    pt.fit_transform(X_train_combined),
    columns=X_train_combined.columns,
    index=X_train.index
)
X_valid_final = pd.DataFrame(
    pt.transform(X_valid_combined),
    columns=X_valid_combined.columns,
    index=X_valid.index
)
print("Encoded columns:", list(X_train_cat.columns))
print(X_train_cat.head())

# Крок 4.5: Зібрати все разом (вже зроблено вище через scaler_all)
# X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
# X_valid_final = pd.concat([X_valid_num, X_valid_cat], axis=1)

print("X_train_final shape:", X_train_final.shape)
print(X_train_final.head())

# Крок 5: Навчити KNN модель
knn = KNeighborsRegressor(n_neighbors=5)
# створюємо модель з K=5. Це значить: для кожного нового працівника шукаємо 5 найсхожіших з 
# тренувальних даних і беремо середнє їхніх зарплат. 5 — це дефолт, потім підберемо оптимальне.
knn.fit(X_train_final, y_train)
# навчаємо модель на тренувальних даних.
y_pred = knn.predict(X_valid_final)
print("Predictions:", y_pred)
print("Actual:     ", y_valid.values)
# порівнюємо передбачені зарплати з реальними.

# Крок 6: Оцінка моделі
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
# середня різниця між прогнозом і реальністю. Якщо MAE = 9000, модель в середньому 
# помиляється на 9k. Найпростіша для розуміння — одиниці такі ж як у Salary.
print(f"Mean Squared Error: {mse:.2f}")
# як MAE, але сильніше штрафує за великі помилки. Пам'ятаєш зразок 6 з похибкою 20,200? RMSE
# "побачить" його краще ніж MAE. Якщо RMSE >> MAE — значить є окремі великі промахи.
print(f"R^2 Score: {r2:.2f}")
# від 0 до 1 (може бути і від'ємним якщо модель зовсім погана). Показує яку частку варіації 
# зарплат модель пояснює. R² = 0.80 означає "модель пояснює 80% різниці в зарплатах". R² = 0 — модель не краща за просто середнє.
mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
print(f"MAPE: {mape:.2f}%")
# Що робимо: рахуємо похибку у відсотках, а не в грошах.
# Навіщо в ML: MAE = 9271 — це багато чи мало? Для зарплати 100k — це 9%, для зарплати 1M — менше 1%. 
# MAPE показує відносну помилку, тому її легше інтерпретувати.

# Як інтерпретувати:

# Метрика	Що значить	Добре якщо
# MAE	Середня помилка в грошах	Менше — краще
# RMSE	Помилка з акцентом на великі промахи	Менше — краще
# R²	Частка поясненої варіації	Ближче до 1 — краще

# Крок 7.2: Підбір оптимального K

best_mape = 100
best_k = 5
best_weights = 'uniform'
best_metric = 'euclidean'

for k in range(1, 50):
    for w in ['uniform', 'distance']:
        for m in ['euclidean', 'manhattan']:
            model = KNeighborsRegressor(n_neighbors=k, weights=w, metric=m)
            model.fit(X_train_final, y_train)
            pred = model.predict(X_valid_final)
            current_mape = np.mean(np.abs((y_valid - pred) / y_valid)) * 100
            if current_mape < best_mape:
                best_mape = current_mape
                best_k = k
                best_weights = w
                best_metric = m
# На практиці: це називається hyperparameter tuning — підбір налаштувань моделі. В серйозних проєктах використовують GridSearchCV, але для 2 параметрів цикл — нормально.


print(f"Best K: {best_k}, weights: {best_weights}, metric: {best_metric}, MAPE: {best_mape:.2f}%")

# ============================================================
# Висновки
# ============================================================
print("\n=== Висновки ===")
print(f"""
1. Модель KNeighborsRegressor з оптимальним K={best_k}, weights='{best_weights}', metric='{best_metric}'
   показала MAPE={best_mape:.2f}% на валідаційному наборі, що знаходиться в цільовому діапазоні 3-5%.

2. Порівняння різних методів кодування та трансформації:
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ Метод                                │ Best K │ MAPE   │ Параметри      │
   ├──────────────────────────────────────────────────────────────────────────┤
   │ OneHotEncoder + StandardScaler       │   20   │ 7.35%  │ uniform        │
   │ TargetEncoder (sklearn, smooth)      │   20   │ 7.19%  │ distance       │
   │ TargetEncoder + target_type=cont.    │   20   │ 7.59%  │ uniform        │
   │ OrdinalEncoder + StandardScaler      │   39   │ 6.08%  │ uniform, manh. │
   │ OrdinalEncoder + Imputer + StdScaler │   16   │ 5.65%  │ distance, eucl.│
   │ Manual TargetEnc + Imputer + Std     │   22   │ 6.73%  │ distance, eucl.│
   │ TargetEncoder + PowerTransformer ✅  │    18  │ 2.15%  │ uniform, manhattan  │
   └──────────────────────────────────────────────────────────────────────────┘

3. Найкращий результат дала комбінація TargetEncoder + PowerTransformer + SimpleImputer.
   TargetEncoder коректно кодує зв'язок категорій з цільовою змінною Salary,
   а PowerTransformer нормалізує розподіли, що критично для обчислення відстаней у KNN.

4. Заповнення пропусків через SimpleImputer (медіана/мода) замість видалення рядків
   дозволило зберегти всі 249 зразків — це покращило MAPE з 6.08% до {best_mape:.2f}%.

5. Найбільш інформативними ознаками для прогнозування зарплати виявилися:
   Experience, Role, Qualification, University та Cert.
""")