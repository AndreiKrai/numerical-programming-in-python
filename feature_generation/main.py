
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
url = 'https://github.com/goitacademy/MACHINE-LEARNING-NEO/raw/refs/heads/main/datasets/mod_05_topic_10_various_data.pkl'
local_path = 'mod_05_topic_10_various_data.pkl'

# urllib.request.urlretrieve(url, local_path)
print(f'Файл завантажено: {local_path}')

# Відкриваємо pkl та отримуємо датасет Autos
with open(local_path, 'rb') as fl:
    datasets = pickle.load(fl)

print('Доступні датасети у файлі:')
print(list(datasets.keys()))

ds_autos = datasets['autos']
print('Перші 5 рядків датасету Autos:')
print(ds_autos.head())
ds_autos.info()

# Крок 2: Визначення дискретних ознак. Що робимо: знаходимо ознаки, які мають скінченну кількість категорій/значень — для розрахунку mutual information (взаємної інформації).Навіщо в ML: Mutual information показує, наскільки ознака пов'язана з таргетом. Але для її розрахунку треба вказати, які ознаки дискретні, а які неперервні — інакше алгоритм неправильно оцінить зв'язок.
# Дискретні ознаки (в широкому розумінні) — це: Категоріальні (object) — текстові мітки Цілочисельні з малою кількістю унікальних значень — по суті теж категорії
# Categorical columns (object type)

cat_cols = ds_autos.select_dtypes(include='object').columns.tolist()

# Integer columns with few unique values (discrete in broad sense)
int_cols = ds_autos.select_dtypes(include='int64').columns
discrete_int = [c for c in int_cols if ds_autos[c].nunique() <= 10]

# All discrete features combined
discrete_features = cat_cols + discrete_int
print('Дискретні ознаки:')
print(discrete_features)

# Крок 3: Розрахунок mutual information для таргету price
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

# Відокремлюємо таргет і ознаки (прибираємо рядки де price = NaN)
df = ds_autos.dropna(subset=['price']).copy()

# Кодуємо категоріальні ознаки числами (OrdinalEncoder підтримує NaN через unknown_value)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])

# Відокремлюємо X та y
X = df.drop(columns=['price'])
y = df['price']

# Заповнюємо пропуски медіаною (mutual_info_regression не підтримує NaN)
X = X.fillna(X.median(numeric_only=True))

# Визначаємо індекси дискретних ознак у X
discrete_mask = [col in discrete_features for col in X.columns]

# Розрахунок mutual information
mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)

# Результат у вигляді відсортованого Series
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print('\nMutual Information scores (топ-10):')
print(mi_series.head(10))

# 4.Створюємо модель
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Важливість ознак
(pd.Series(rf.feature_importances_, index=X.columns)
 .sort_values(ascending=True)
 .plot.barh(title='Важливість ознак за Random Forest', figsize=(10, 6)))

plt.tight_layout()
plt.show()

# Крок 5: Масштабування через rank(pct=True)
# Об'єднуємо два Series в один DataFrame
feature_scores = pd.DataFrame({
    'MI': mi_series,
    'RF': pd.Series(rf.feature_importances_, index=X.columns)
})

# rank(pct=True) — переводить значення в перцентилі (0 до 1)
# тепер MI і RF на одній шкалі, можна порівнювати
feature_ranks = feature_scores.rank(pct=True)

print('\nРанги ознак (перцентилі):')
print(feature_ranks.sort_values('MI', ascending=False).head(10))

# Крок 6: Grouped barplot через seaborn catplot
import seaborn as sns

# melt: wide → long формат, потрібний для catplot
feature_ranks_melted = (feature_ranks
    .reset_index()                          # назва ознаки стає колонкою 'index'
    .rename(columns={'index': 'feature'})
    .melt(id_vars='feature', var_name='method', value_name='rank'))

# Сортуємо ознаки за середнім рангом (для зручності читання графіку)
order = (feature_ranks
    .mean(axis=1)
    .sort_values(ascending=False)
    .index.tolist())

sns.catplot(
    data=feature_ranks_melted,
    x='rank',
    y='feature',
    hue='method',
    kind='bar',
    order=order,
    height=7,
    aspect=1.4
)
plt.title('Порівняння MI та RF: ранги ознак (перцентилі)')
plt.tight_layout()
plt.show()

# Висновки після аналізу візуалізації (крок 7):
#
# 1. УЗГОДЖЕНІСТЬ МЕТОДІВ:
#    curb_weight, engine_size, horsepower — топ-3 за обома методами (MI і RF).
#    Це підтверджує що вони справді найважливіші предиктори ціни авто.
#
# 2. РОЗБІЖНОСТІ:
#    - 'make' (марка): RF оцінює вище ніж MI — дерева добре використовують
#      категоріальні ознаки для розгалуження, MI може недооцінювати їх після OrdinalEncoder.
#    - 'bore' (діаметр циліндра): MI оцінює вище ніж RF — можливо нелінійний зв'язок
#      з ціною, який MI ловить, але RF розподіляє важливість між схожими ознаками.
#
# 3. ВІДБІР ОЗНАК:
#    Ознаки з рангом < 0.3 за обома методами — кандидати на видалення.
#    Ознаки з рангом > 0.6 хоча б в одному методі — варто залишити.
#
# 4. МУЛЬТИКОЛІНЕАРНІСТЬ:
#    city_mpg і highway_mpg мають схожі ранги — вони корельовані між собою.
#    У лінійних моделях достатньо залишити одну з них.