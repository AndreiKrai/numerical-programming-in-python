import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.mixture import GaussianMixture


df = pd.read_csv("archive/2017.csv")
print( "Shape of the dataset: ", df.shape )
print("\nColumn types:")
print(df.dtypes)
print("\nStatistics:")
print(df.describe())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_cols)

# Plot distribution for each numeric feature
df[numeric_cols].hist(bins=20, figsize=(14, 10))
plt.suptitle("Distribution of numeric features", fontsize=14)
plt.tight_layout()
plt.savefig("distributions.png")
plt.show()

# Happiness.Score, Economy, Health — більш схожі на нормальний розподіл (симетричні)
# Generosity, Corruption, Freedom, Family — скошені (skewed), хвіст праворуч або ліворуч

# Крок 6: Кореляційна матриця
# Навіщо в ML: якщо дві ознаки сильно корелюють — вони дають одну і ту ж інформацію моделі. Можна одну прибрати, щоб зменшити розмірність.

# Select features for correlation analysis
features = ['Happiness.Score', 'Economy..GDP.per.Capita.', 
            'Family', 'Health..Life.Expectancy.', 
            'Freedom', 'Generosity', 'Trust..Government.Corruption.']

# Correlation matrix
corr_matrix = df[features].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation.png")
plt.show()

# Economy (GDP per Capita) — найсильніша кореляція з Happiness.Score (~0.81).
# Дивлячись на матрицю, можна зробити такі висновки:

# Economy → Happiness.Score: сильний позитивний зв'язок (~0.81) — багатші країни щасливіші
# Health → Happiness.Score: сильний (~0.78) — довше живуть = щасливіші
# Family → Happiness.Score: середній-сильний (~0.74)
# Freedom → Happiness.Score: слабший (~0.57)
# Generosity, Trust → слабкий зв'язок (~0.15–0.40)

# Крок 8: Теплова мапа по країнах

# Навіщо в ML: візуалізація цільової ознаки географічно — одразу видно регіональні патерни.

fig = px.choropleth(df,
                    locations="Country",
                    color="Happiness.Score",
                    locationmode="country names",
                    color_continuous_scale="RdYlGn")
fig.update_layout(title="Happiness Index 2017")
fig.write_html("happiness_map.html")
fig.show()

def data_scale(data, scaler_type='minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    if scaler_type == 'std':
        scaler = StandardScaler()
    if scaler_type == 'norm':
        scaler = Normalizer()
    scaler.fit(data)
    return scaler.transform(data)

# Select only numeric columns for scaling
original_df = df[numeric_cols]

data_scaled = data_scale(original_df)
df_scaled = pd.DataFrame(data_scaled, columns=numeric_cols)

print("=== ORIGINAL ===")
print(original_df.describe().round(3))
print("\n=== SCALED (MinMax) ===")
print(df_scaled.describe().round(3))

# Step 11: GaussianMixture clustering
cluster_features = ['Economy..GDP.per.Capita.', 'Health..Life.Expectancy.',
                    'Family', 'Freedom', 'Trust..Government.Corruption.']

X = data_scale(df[cluster_features])

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
df['Cluster'] = gmm.predict(X)

print(df[['Country', 'Happiness.Score', 'Cluster']].sort_values('Cluster').to_string())
print("\nКраїн у кожному кластері:")
print(df['Cluster'].value_counts())

# Step 12: Choropleth map colored by cluster
fig2 = px.choropleth(df,
                     locations="Country",
                     color="Cluster",
                     locationmode="country names",
                     color_continuous_scale="RdYlGn",
                     title="Country Clusters (GaussianMixture)")
fig2.write_html("clusters_map.html")
fig2.show()

# Step 13: Different feature sets — compare clustering results

# Without Trust and Freedom (only economic/health)
features_v2 = ['Economy..GDP.per.Capita.', 'Health..Life.Expectancy.', 'Family']
X2 = data_scale(df[features_v2])
gmm2 = GaussianMixture(n_components=3, random_state=42)
df['Cluster_v2'] = gmm2.fit_predict(X2)

print("\nКластери збіглись (%):", 
      round((df['Cluster'] == df['Cluster_v2']).mean() * 100, 1))

fig3 = px.choropleth(df,
                     locations="Country",
                     color="Cluster_v2",
                     locationmode="country names",
                     color_continuous_scale="RdYlGn",
                     title="Clusters v2 — Economy + Health + Family only")
fig3.write_html("clusters_map_v2.html")
fig3.show()

# ============================================================
# ВИСНОВОК
# ============================================================
# Кластеризація GaussianMixture на 3 кластери частково відтворює
# реальний розподіл країн за Happiness.Score:
#
# - Кластер з високим Happiness (~6-8): Скандинавія, Канада, Австралія
#   → високий GDP, здоров'я, сімейні цінності
#
# - Кластер з середнім Happiness (~4-6): Латинська Америка, Східна Європа
#   → середні показники по всіх ознаках
#
# - Кластер з низьким Happiness (~2-4): Африка, Південна Азія
#   → низький GDP, здоров'я, довіра до уряду
#
# Зміна набору ознак (крок 13) впливає на результат:
# - Без Freedom і Trust кластери стають більш "економічними"
# - З усіма ознаками кластери краще відображають соціальний контекст
#
# Висновок: кластеризація без міток добре групує країни за рівнем
# розвитку, що відповідає оригінальному Happiness.Score (~70-80%)
# ============================================================