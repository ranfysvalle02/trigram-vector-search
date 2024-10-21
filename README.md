# trigram-vector-search

## Enhancing Text Similarity Search with Trigram Hashing and Combined Similarity Measures

---

## Introduction

In the world of text processing and search, finding similar strings or documents is a common yet challenging task. Whether it's for spell-checking, plagiarism detection, or recommendation systems, efficient and accurate text similarity measures are essential.

In this blog post, I share my journey of implementing a text similarity search using **trigram hashing**, optimizing the code, and improving the similarity measures to achieve better results. I'll explore:

- Understanding **trigram hashing** and how to implement it.
- Evaluating different **similarity measures** and their suitability.
- Techniques for **optimizing code** for better performance and accuracy.
- Lessons learned from troubleshooting and refining the approach.

By sharing these insights, I aim to help others avoid common pitfalls and streamline their text similarity projects.

---

## What is Trigram Hashing?

### Understanding Trigrams

A **trigram** is a contiguous sequence of three characters extracted from a string. For example, the word `"hello"` can be broken down into trigrams: `["hel", "ell", "llo"]`. Trigrams capture local character patterns and are useful in text analysis, especially for languages without clear word boundaries or for fuzzy matching.

### Hashing Trigrams

To convert trigrams into numerical representations (embeddings), I applied a **hashing function**. Hashing maps trigrams to numerical values in a consistent and deterministic way. By hashing trigrams, I created embeddings that can be used for similarity comparisons.

### Implementing Trigram Hashing

Here's how I implemented trigram hashing in Python:

```python
import hashlib

def trigram_hash(word, vocab_size=32000):
    """Create a trigram hash for a given word using a stable hashing function."""
    word = "_" + word + "_"  # Adding delimiters to mark start and end
    embeddings = []
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        # Use MD5 to hash the trigram and convert to an integer
        hashed_trigram = int(hashlib.md5(trigram.encode('utf-8')).hexdigest(), 16) % vocab_size
        embeddings.append(hashed_trigram)
    return embeddings
```

- **Delimiters:** Adding underscores `_` at the start and end of the word helps capture trigrams at the boundaries.
- **Hashing Function:** I used MD5 to hash the trigrams and took modulo `vocab_size` to limit the range of hash values.

---

## Choosing the Right Similarity Measure

After obtaining embeddings for my text data, the next step was to compare them to find similarities. The choice of similarity measure significantly impacts the effectiveness of the search.

### Initial Approach: Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors, providing a value between -1 and 1. It's widely used for continuous, dense vectors, such as those from word embeddings like Word2Vec or BERT.

**Why It Didn't Work Well:**

- **Discrete Embeddings:** My trigram hash embeddings were discrete and sparse, not continuous and dense.
- **Misleading Results:** Cosine similarity on hashed trigrams often returned high similarity scores for unrelated strings due to numerical coincidences.

### Set-Based Similarity Measures

#### Jaccard Similarity

**Definition:** Measures the overlap between two sets.

\[
\text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|}
\]

- **Suitable for:** Comparing sets of trigrams to see how many trigrams are shared.
- **Limitations:** Can be too strict, as it doesn't account for the size of the intersection relative to the sizes of the individual sets.

#### Dice Coefficient

**Definition:** A variation of Jaccard that weighs overlap more heavily.

\[
\text{Dice Coefficient} = \frac{2 \times |A \cap B|}{|A| + |B|}
\]

- **Benefits:** More lenient than Jaccard, giving higher similarity scores for partial overlaps.

### Why Set-Based Measures Were Better

- **Alignment with Data:** Trigram hash embeddings are essentially sets of hashed trigrams.
- **Avoiding Numerical Coincidences:** Set-based measures focus on shared elements rather than numerical vector values.

---

## Combining Similarity Measures

To achieve a more balanced and accurate similarity score, I decided to combine the strengths of both set-based and vector-based measures.

### Combined Similarity Approach

- **Dice Coefficient:** Captures the overlap of trigrams between two strings.
- **Cosine Similarity:** Accounts for the magnitude and direction of the embeddings.

**Implementation:**

```python
def combined_similarity(set1, set2, vec1, vec2):
    """Combine Dice coefficient and cosine similarity for a balanced measure."""
    # Dice coefficient on sets
    dice_sim = dice_coefficient(set1, set2)
    # Cosine similarity on vectors
    cosine_sim = cosine_similarity(vec1, vec2)
    # Weighted combination
    combined_score = 0.5 * dice_sim + 0.5 * cosine_sim
    return combined_score
```

- **Balanced Weighting:** I assigned equal weights to both similarity measures.
- **Enhanced Accuracy:** This approach leverages both the overlap of trigrams and the vector representation's direction and magnitude.

---

## Optimizing the Code

During the implementation, I encountered several challenges:

### Issue 1: Embedding Length Mismatch

- **Problem:** Embeddings of different lengths caused shape mismatch errors during cosine similarity calculation.
- **Solution:** Dynamically pad all embeddings to the same length.

**Updated Padding Function:**

```python
def pad_embedding(embedding, max_length):
    """Pad or truncate embedding to a fixed max length."""
    if len(embedding) < max_length:
        embedding += [0] * (max_length - len(embedding))
    elif len(embedding) > max_length:
        embedding = embedding[:max_length]
    return embedding
```

### Issue 2: Hash Collisions

- **Problem:** Different trigrams mapping to the same hash value due to limited `vocab_size`.
- **Solution:** Increase `vocab_size` to reduce collisions.

```python
def trigram_hash(word, vocab_size=32000):
    # ...
```

### Issue 3: Similarity Scores Feeling "Off"

- **Problem:** Similarity scores didn't align with intuitive expectations.
- **Solution:** Combine Dice coefficient with cosine similarity for a more nuanced measure.

---

## Final Implementation

Putting it all together, here's the complete code:

```python
import numpy as np
import hashlib
import pymongo

def trigram_hash(word, vocab_size=32000):
    """Create a trigram hash for a given word using a stable hashing function."""
    word = "_" + word + "_"
    embeddings = []
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        hashed_trigram = int(hashlib.md5(trigram.encode('utf-8')).hexdigest(), 16) % vocab_size
        embeddings.append(hashed_trigram)
    return embeddings

def get_max_embedding_length(embeddings_list):
    """Get the maximum length of embeddings in the vector store."""
    return max([len(embedding) for embedding in embeddings_list])

def pad_embedding(embedding, max_length):
    """Pad or truncate embedding to a fixed max length."""
    if len(embedding) < max_length:
        embedding += [0] * (max_length - len(embedding))
    elif len(embedding) > max_length:
        embedding = embedding[:max_length]
    return embedding

def dice_coefficient(set1, set2):
    """Calculate Dice coefficient between two sets."""
    intersection = len(set(set1) & set(set2))
    return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) != 0 else 0

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

def combined_similarity(set1, set2, vec1, vec2):
    """Combine Dice coefficient and cosine similarity for a balanced measure."""
    dice_sim = dice_coefficient(set1, set2)
    cosine_sim = cosine_similarity(vec1, vec2)
    combined_score = 0.5 * dice_sim + 0.5 * cosine_sim
    return combined_score

# In-memory vector store
vector_store = {}

def add_to_store(word, embedding):
    """Add word and its trigram embedding to the in-memory store."""
    vector_store[word] = embedding

def search_store_combined(query_embedding, top_k=5):
    """Search for the top_k most similar embeddings using combined similarity."""
    query_set = set(query_embedding)
    embeddings_list = list(vector_store.values()) + [query_embedding]
    max_length = get_max_embedding_length(embeddings_list)
    padded_query_embedding = pad_embedding(query_embedding.copy(), max_length)
    similarities = {}
    for word, stored_embedding in vector_store.items():
        stored_set = set(stored_embedding)
        padded_stored_embedding = pad_embedding(stored_embedding.copy(), max_length)
        similarity = combined_similarity(query_set, stored_set, padded_query_embedding, padded_stored_embedding)
        similarities[word] = similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# Example usage with a MongoDB database
# MongoDB connection (replace with your connection string)
mdb_client = pymongo.MongoClient('mongodb+srv://127.0.0.1/?directConnection=true')
db = mdb_client['sample_mflix']
movies_collection = db['movies']

# Process movies from MongoDB
movies_cursor = movies_collection.find({}, {'title': 1})

# Adding movie titles to vector store
for movie in movies_cursor:
    title = movie.get('title')
    if title:
        embedding = trigram_hash(title)
        add_to_store(title, embedding)

# Example search query
query_title = "The Matrix"
query_embedding = trigram_hash(query_title)

# Search with combined similarity
similar_movies = search_store_combined(query_embedding)
for movie, similarity in similar_movies:
    print(f"{movie}: {similarity}")
```

---

## Results

With the combined similarity measure, the top similar movies to `"The Matrix"` are:

```
The Matrix: 1.0
The Matriarch: 0.8332
The Matrimony: 0.7725
The Mark: 0.7387
The Animatrix: 0.7356
```

- **Accurate Rankings:** The results are more intuitive, with titles that are lexically or contextually similar ranking higher.
- **Balanced Similarity Scores:** The scores reflect both the shared trigrams and the overall embedding similarity.

---

## Lessons Learned

1. **Choose the Right Similarity Measure:** Not all similarity measures are suitable for all types of embeddings. It's crucial to select one that aligns with the nature of your data.
2. **Beware of Hash Collisions:** When hashing, ensure your `vocab_size` is large enough to minimize collisions.
3. **Consistent Embedding Lengths:** Aligning the lengths of embeddings is essential for vector-based similarity calculations.
4. **Combine Multiple Measures:** Sometimes, combining different similarity measures yields better results than relying on a single one.
5. **Test and Iterate:** Regularly test your implementation with real data to identify issues and areas for improvement.

---

## Conclusion

Implementing an effective text similarity search involves careful consideration of how text is represented and compared. By using trigram hashing and combining set-based and vector-based similarity measures, I was able to achieve more accurate and intuitive results.

I hope this blog post helps others navigate similar challenges and accelerates their development process. Remember, experimentation and iteration are key to finding the optimal solution for your specific use case.

---

**References:**

- [Text Similarity Measures](https://en.wikipedia.org/wiki/String_metric)
- [Trigram Analysis](https://en.wikipedia.org/wiki/N-gram)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
- [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
