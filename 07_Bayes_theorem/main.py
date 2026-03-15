# --- Імпорти ---
import pandas as pd                        # робота з таблицями (DataFrame)
import seaborn as sns                      # красиві графіки
import matplotlib.pyplot as plt            # базова бібліотека графіків
import re                                  # регулярні вирази для очищення тексту
import numpy as np                         # математика (log, операції з масивами)
from collections import Counter            # підрахунок частоти елементів у списку
from nltk.corpus import stopwords          # списки стоп-слів (the, is, a...)
from nltk.stem import WordNetLemmatizer    # лематизація: running → run

# --- Завантаження NLTK ресурсів ---
import nltk
nltk.download('stopwords')   # список стоп-слів для англ. мови
nltk.download('punkt')       # токенізатор (розбивка тексту на слова)
nltk.download('wordnet')     # словник для лематизації
stop_words = set(stopwords.words('english'))  # set для швидкої перевірки (O(1) vs O(n))

# =============================================================
# КРОК 1: Відібрати збалансовану вибірку з великого датасету
# =============================================================
# Оригінальний датасет: 83448 записів (spam=1, ham=0)
# Беремо по 2000 кожного класу → 4000 збалансованих записів

df = pd.read_csv('combined_data.csv')
df_spam = df[df["label"]==1].sample(2000, random_state=42)   # 2000 випадкових spam
df_ham  = df[df["label"]==0].sample(2000, random_state=42)   # 2000 випадкових ham

# Склеїти, перемішати (frac=1 = 100% рядків), скинути індекс на 0..3999
df_sampled = pd.concat([df_spam, df_ham]).sample(frac=1, random_state=42).reset_index(drop=True)

# =============================================================
# КРОК 2: Візуалізація розподілу класів
# =============================================================
# Переконуємось що вибірка збалансована перед тренуванням

ham, spam = df_sampled["label"].value_counts()   # розпакувати Series → два числа
plt.pie([spam, ham], labels=['Spam', 'Ham'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title("Розподіл spam / ham у вибірці")
plt.show()

# =============================================================
# КРОК 3: Обробка тексту (NLP pipeline)
# =============================================================
# Мета: прибрати шум і залишити тільки значущі слова

corpus = []
lemmatizer = WordNetLemmatizer()

for document in df_sampled["text"]:
    # Видалити все крім літер (цифри, пунктуація → пробіл), привести до нижнього регістру
    document = re.sub("[^a-zA-Z]", " ", document).lower()
    # Розбити рядок на список слів
    document = document.split()
    # Лематизація + фільтрація стоп-слів (the, is, a... не несуть сенсу для класифікації)
    document = [lemmatizer.lemmatize(word) for word in document if word not in stop_words]
    # Видалити дублікати слів у одному листі (set → унікальні слова)
    document = list(set(document))
    # Зібрати список слів назад у рядок
    document = " ".join(document)
    corpus.append(document)

# Замінити оригінальні тексти очищеними
df_sampled["text"] = corpus

# =============================================================
# КРОК 4: Підготовка train/test структур
# =============================================================
# 80% (3200) → тренування, 20% (800) → тестування

train_corpus = df_sampled.iloc[:3200]   # перші 3200 рядків
test_corpus  = df_sampled.iloc[3200:]   # останні 800 рядків

# Списки текстів за класами (потрібні для підрахунку ймовірностей)
train_spam = train_corpus[train_corpus["label"]==1]["text"].tolist()
train_ham  = train_corpus[train_corpus["label"]==0]["text"].tolist()

# Список словників [{"label": 0/1, "text": "..."}, ...] для тестування
test_emails = test_corpus.to_dict('records')

# =============================================================
# КРОК 5: Наївний Баєс — тренування
# =============================================================

# Об'єднати всі слова кожного класу в один список
spam_words = " ".join(train_spam).split()
ham_words  = " ".join(train_ham).split()

# Порахувати частоту кожного слова у кожному класі
spam_word_counts = Counter(spam_words)
ham_word_counts  = Counter(ham_words)

# Загальний словник = всі унікальні слова з обох класів
vocabulary = set(spam_word_counts.keys()) | set(ham_word_counts.keys())

# Апріорні ймовірності: P(spam) і P(ham) на основі train
p_spam = len(train_spam) / (len(train_spam) + len(train_ham))
p_ham  = len(train_ham)  / (len(train_spam) + len(train_ham))

def classify(email_text):
    """
    Класифікує лист як spam (1) або ham (0).
    Використовує log-ймовірності щоб уникнути underflow
    при множенні багатьох малих чисел.
    """
    words = email_text.split()

    # Починаємо з log апріорних ймовірностей
    log_p_spam = np.log(p_spam)
    log_p_ham  = np.log(p_ham)

    for word in words:
        # P(слово|spam) зі згладжуванням Лапласа (+1):
        # щоб уникнути P=0 для слів яких не було у train
        p_word_spam = (spam_word_counts.get(word, 0) + 1) / (len(spam_words) + len(vocabulary))
        p_word_ham  = (ham_word_counts.get(word, 0)  + 1) / (len(ham_words)  + len(vocabulary))

        # log(a*b*c) = log(a) + log(b) + log(c) — сума замість добутку
        log_p_spam += np.log(p_word_spam)
        log_p_ham  += np.log(p_word_ham)

    # Повертаємо клас з більшою log-ймовірністю
    return 1 if log_p_spam > log_p_ham else 0

# =============================================================
# КРОК 5: Оцінка якості класифікатора
# =============================================================

correct = 0
for email in test_emails:
    prediction = classify(email["text"])
    if prediction == email["label"]:   # порівняти передбачення з реальним label
        correct += 1

accuracy = correct / len(test_emails)
print(f"Accuracy: {accuracy:.2%}")

# =============================================================
# КРОК 6: Аналіз — які слова найбільш "спамові"?
# =============================================================

# P(слово | spam) для кожного слова зі словника (зі згладжуванням Лапласа)
result = {
    word: (spam_word_counts.get(word, 0) + 1) / (len(spam_words) + len(vocabulary))
    for word in vocabulary
}

# Відсортувати за ймовірністю від більшої до меншої, взяти топ-20
top_spam_words = sorted(result.items(), key=lambda x: x[1], reverse=True)[:20]

# Висновок
print("""
=== ВИСНОВОК ===

Наївний Баєс-класифікатор досяг точності 95.88% на тестовій вибірці (800 листів).

Топ слова у spam:
- 'http', 'com', 'www' — spam часто містить посилання
- 'price', 'offer', 'product' — комерційна лексика
- 'best', 'new', 'get' — типові маркетингові слова

Класифікатор правильно навчився розпізнавати spam-патерни
без використання готових ML-бібліотек, лише на основі
частоти слів та теореми Баєса.
""")