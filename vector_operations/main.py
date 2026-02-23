import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Loading data
with open('word_embeddings_subset.p', 'rb') as f:
    word_embeddings = pickle.load(f)

# Each word is a row, each column is a vector element (300 dimensions)
df_full = pd.DataFrame.from_dict(word_embeddings, orient='index')
print(f"Original shape: {df_full.shape}")  # (N, 300)

# Reduce 300 dimensions to 3 using PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(df_full.values)
df = pd.DataFrame(vectors_3d, index=df_full.index, columns=['x', 'y', 'z'])
print(f"Reduced shape: {df.shape}")  # (N, 3)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
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

# Mark: Cross Product (vector product) — works only in 3D
# Using PCA-reduced 3D vectors from df
def find_closest_to_vector_3d(vec_3d, exclude=[]):
    """Find the closest word to a 3D vector using PCA-reduced data."""
    best_word = None
    best_sim = -1
    for word in df.index:
        if word in exclude:
            continue
        word_vec = df.loc[word].values
        sim = cosine_similarity(vec_3d, word_vec)
        if sim > best_sim:
            best_word = word
            best_sim = sim
    return (best_word, best_sim)

# Cross product of two words in 3D space
word_c, word_d = 'France', 'Paris'
vec_c = df.loc[word_c].values
vec_d = df.loc[word_d].values

cross_product_2 = np.cross(vec_c, vec_d)
print(f"\n--- Cross product: '{word_c}' × '{word_d}' ---")
print(f"  Cross product: {cross_product_2}")
result_cross_2 = find_closest_to_vector_3d(cross_product_2, exclude=[word_c, word_d])
print(f"  Closest word to cross product: '{result_cross_2[0]}' (similarity: {result_cross_2[1]:.4f})")

# Note: Cross product returns a vector PERPENDICULAR to both input vectors.
# In word embedding space this doesn't have a clear semantic meaning
# (unlike dot product or vector addition), but it demonstrates
# that cross product only works in 3D — which is why we needed PCA first.

# Analysis of Cross Product results:
#
# Cross product: 'France' × 'Paris' → closest word: 'Zagreb' (similarity: 0.92)
#
# Why Zagreb? Interpretation:
#
# 1. Cross product gives a vector PERPENDICULAR to both France and Paris.
#    It does NOT combine their meanings — it finds a direction orthogonal to both.
#
# 2. The result has NO semantic meaning in NLP. Unlike:
#    - Dot product → measures similarity (France · Paris = high → related)
#    - Vector addition → combines meanings (Belarus + capital ≈ Minsk)
#    - Analogy (a - b + c) → transfers relationships (Belarus - France + Paris = Minsk)
#
# 3. Zagreb appeared because in the compressed 3D PCA space, it happens to lie
#    in the direction perpendicular to the France-Paris plane. This is a geometric
#    coincidence, not a meaningful relationship.
#
# 4. The high similarity (0.92) is misleading — it only means Zagreb's 3D PCA vector
#    points in a similar direction to the cross product result. In the original 300D
#    space, this relationship would not hold.
#
# 5. Cross product is useful in physics and 3D graphics (calculating surface normals,
#    torque, angular momentum), but NOT in NLP/word embeddings.
#
# Conclusion: Cross product demonstrates a mathematical operation on 3D vectors,
# but the result should NOT be interpreted semantically in word embedding context.