import pickle
import numpy as np
import pandas as pd

# Loading data
with open('word_embeddings_subset.p', 'rb') as f:
    word_embeddings = pickle.load(f)

# Each word is a row, each column is a vector element
df = pd.DataFrame.from_dict(word_embeddings, orient='index')
print(df.head())

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar(word, top_n=5):
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return []
    
    target_vec = word_embeddings[word]
    similarities = {}
    
    for other_word, other_vec in word_embeddings.items():
        if other_word != word:
            sim = cosine_similarity(target_vec, other_vec)
            similarities[other_word] = sim
            
    # Sort by similarity and return top N
    most_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return most_similar

def find_most_similar_one(word):
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return []
    target_vec = word_embeddings[word]
    best_word = None
    best_sim = -1  # cosine similarity ranges from -1 to 1
    for other_word, other_vec in word_embeddings.items():
        if other_word != word:
            sim = cosine_similarity(target_vec, other_vec)
            if sim > best_sim:
                best_word = other_word
                best_sim = sim
    return (best_word, best_sim)

def find_closest_to_vector(vec, exclude=[]):
    best_word = None
    best_sim = -1
    for w, v in word_embeddings.items():
        if w in exclude:
            continue
        sim = cosine_similarity(vec, v)
        if sim > best_sim:
            best_word = w
            best_sim = sim
    return (best_word, best_sim)

# Mark: Function to calculate angle between words
def angle_between_words(word1, word2):
    """Calculates the angle between two words in degrees."""
    if word1 not in word_embeddings or word2 not in word_embeddings:
        print(f"Word '{word1}' or '{word2}' not found.")
        return None
    vec1 = word_embeddings[word1]
    vec2 = word_embeddings[word2]
    cos_theta = cosine_similarity(vec1, vec2)
    # Clip value to [-1, 1] to avoid rounding errors
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees


# Usage examples
# pairs = [
#     ('king', 'queen'),
#     ('king', 'cat'),
#     ('happy', 'sad'),
#     ('France', 'Paris'),
#     ('Minsk', 'Belarus'),
# ]

# print("\n--- Кут між словами ---")
# for w1, w2 in pairs:
#     angle = angle_between_words(w1, w2)
#     print(f"  '{w1}' ↔ '{w2}': {angle:.2f}°")

# The smaller the angle — the more similar the words:
# ~0°  → almost identical in meaning
# ~90° → unrelated
# ~180° → opposite
# France → Paris, as Belarus → ?
# Formula: Belarus - France + Paris = Minsk
vec_analogy = word_embeddings['Belarus'] - word_embeddings['France'] + word_embeddings['Paris']
print(f"\n--- Vector analogy: 'Belarus' - 'France' + 'Paris' = ? ---")
result = find_closest_to_vector(vec_analogy, exclude=['Belarus', 'France', 'Paris'])
print(f"Closest word: '{result[0]}' (similarity: {result[1]:.4f})")


# Top-5 most similar words for each word in pairs
# print("\n--- Top-5 most similar words ---")

print(f"\n'{result[0]}': {find_most_similar(result[0], top_n=5)}")

# Conclusions:
# 1. Analogy Belarus - France + Paris = Minsk (similarity 0.7234) — the model correctly
#    found the capital of Belarus using the "country → capital" relationship.
#
# 2. Top-5 most similar words to 'Minsk':
#    - Kiev (0.771) — capital of neighboring Ukraine
#    - Chisinau (0.730) — capital of Moldova (another neighbor)
#    - Moscow (0.729) — capital of Russia (neighboring country)
#    - Baku (0.706) — capital of Azerbaijan (post-Soviet country)
#    - Belarus (0.700) — the country Minsk belongs to
#
# 3. All nearest words are capitals of post-Soviet / neighboring countries.
#    This confirms that word embeddings preserve geographic and political relationships.

print(f"\n--- Angle between words 'Belarus' and 'Minsk' ---")
angle = angle_between_words("Belarus", "Minsk")
print(f"  'Belarus' ↔ 'Minsk': {angle:.2f}°")