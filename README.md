# Enhancing Text Similarity Search with Trigram Hashing and Combined Similarity Measures

## Introduction

In the world of text processing and search, finding similar strings or documents is a common yet challenging task. Whether it's for spell-checking, plagiarism detection, or recommendation systems, efficient and accurate text similarity measures are essential.

In this blog post, I share my journey of implementing a text similarity search using **trigram hashing**, optimizing the code, and improving the similarity measures to achieve better results. Along the way, I encountered several challenges and learned valuable lessons that I hope will help others avoid similar pitfalls.

We'll explore:

- Understanding **trigram hashing** and how to implement it.
- Evaluating different **similarity measures** and their suitability for different types of embeddings.
- Techniques for **optimizing code** for better performance and accuracy.
- **Lessons learned** from troubleshooting and refining the approach.

By sharing these insights, I aim to provide a comprehensive and educational resource for anyone looking to implement text similarity search in their projects.

---

## What is Trigram Hashing?

### Understanding Trigrams

A **trigram** is a contiguous sequence of three characters extracted from a string. Trigrams capture local character patterns and are particularly useful in text analysis tasks such as spell-checking, language detection, and fuzzy matching.

**Example:**

For the word `"hello"`, the trigrams would be:

- Add start and end delimiters: `"_hello_"`
- Extract trigrams:

  ```
  "_he", "hel", "ell", "llo", "lo_"
  ```

These trigrams help in capturing the structure of the word, including its beginning and end.

### Hashing Trigrams

To convert trigrams into numerical representations (embeddings), we can apply a **hashing function**. Hashing maps trigrams to numerical values in a consistent and deterministic way. By hashing trigrams, we create embeddings that can be used for similarity comparisons.

**Why Hash Trigrams?**

- **Fixed Size Representation:** Hashing allows us to map a potentially vast number of trigrams into a fixed range of integers.
- **Efficiency:** Hashing enables us to store and compare trigrams efficiently.
- **Privacy:** Hash functions are one-way, which can be useful if we need to obscure the original trigrams for privacy reasons.

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

**Key Components:**

- **Delimiters:** Adding underscores `_` at the start and end of the word helps capture trigrams at the boundaries. For example, `"hello"` becomes `"_hello_"`.
- **Hashing Function:** I used MD5 to hash the trigrams and then took the modulo with `vocab_size` to limit the range of hash values.
- **Vocabulary Size (`vocab_size`):** This parameter defines the size of the embedding space. A larger `vocab_size` reduces the chance of hash collisions (different trigrams mapping to the same hash value) but requires more memory.

---

## Choosing the Right Similarity Measure

After obtaining embeddings for my text data, the next step was to compare them to find similarities. The choice of similarity measure significantly impacts the effectiveness of the search.

### Initial Approach: Cosine Similarity

**Cosine similarity** measures the cosine of the angle between two vectors, providing a value between -1 and 1. It's widely used for continuous, dense vectors, such as those from word embeddings like Word2Vec or BERT.

**Implementation:**

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)
```

**Why It Didn't Work Well:**

- **Discrete Embeddings:** My trigram hash embeddings were **discrete and sparse**, meaning they consisted of integer values representing hashed trigrams, and the embeddings could vary significantly in length.
- **Not Continuous and Dense:** Cosine similarity is best suited for **continuous, dense vectors** where the direction and magnitude of the vectors carry meaningful information about the data.
- **Misleading Results:** Applying cosine similarity to discrete embeddings resulted in high similarity scores for unrelated strings due to numerical coincidences rather than meaningful relationships.

**Example Issue:**

- Titles like `"The Matrix"` and `"Chrysalis"` might receive high similarity scores because their embeddings, though numerically similar, don't reflect any actual similarity in content or meaning.

### Set-Based Similarity Measures

Given the limitations of cosine similarity with discrete embeddings, I explored set-based similarity measures that are more appropriate for comparing sets of discrete items.

#### Jaccard Similarity

**Definition:** Measures the overlap between two sets by comparing the size of their intersection to the size of their union.

![](https://imgs.search.brave.com/5puihAbIeyg4_PpgohYkeaqQQyu-d2krI2OVHdvq_cw/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZWVrc2Zvcmdl/ZWtzLm9yZy93cC1j/b250ZW50L3VwbG9h/ZHMvMjAyMzA4MTEx/MzI2MDgvSG93LXRv/LUNhbGN1bGF0ZS1K/YWNjYXJkLVNpbWls/YXJpdHktaW4tUHl0/aG9uLTIucG5n)

- **Suitable for:** Comparing sets of trigrams to see how many trigrams are shared between two strings.
- **Limitations:** Can be too strict, especially when dealing with small sets or when the intersection is small compared to the union.

**Implementation:**

```python
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    if union == 0:
        return 0.0
    return intersection / union
```

#### Dice Coefficient

**Definition:** A variation of Jaccard that weighs the overlap more heavily by doubling the size of the intersection.

![](https://www.quantib.com/hs-fs/hubfs/Content%20and%20assets/Blog%20and%20news%20images/Dice%20coefficient.jpg?width=1427&name=Dice%20coefficient.jpg)

- **Benefits:** More lenient than Jaccard, giving higher similarity scores for partial overlaps, which is beneficial when comparing strings that may have minor differences.

**Implementation:**

```python
def dice_coefficient(set1, set2):
    """Calculate Dice coefficient between two sets."""
    intersection = len(set(set1) & set(set2))
    total_size = len(set1) + len(set2)
    if total_size == 0:
        return 0.0
    return (2 * intersection) / total_size
```

### Why Set-Based Measures Were Better

- **Alignment with Data:** Trigram hash embeddings are essentially **sets of hashed trigrams**. Set-based measures naturally align with this data representation.
- **Avoiding Numerical Coincidences:** Unlike cosine similarity, set-based measures focus on **shared elements** rather than numerical values, reducing the chance of misleading similarity scores due to numerical coincidences.
- **Better Reflecting Actual Similarity:** They provide a more intuitive measure of similarity based on the actual content of the strings.

---

## Combining Similarity Measures

While set-based measures improved the results, I noticed that they sometimes didn't fully capture the similarity between strings, especially when the sets had few overlapping trigrams. To achieve a more balanced and accurate similarity score, I decided to combine the strengths of both set-based and vector-based measures.

### Combined Similarity Approach

- **Dice Coefficient:** Captures the **overlap of trigrams** between two strings, emphasizing shared content.
- **Cosine Similarity:** Accounts for the **magnitude and direction** of the embeddings, considering the overall structure and length.

**Implementation:**

```python
def combined_similarity(set1, set2, vec1, vec2):
    """Combine Dice coefficient and cosine similarity for a balanced measure."""
    # Calculate Dice coefficient
    dice_sim = dice_coefficient(set1, set2)
    # Calculate Cosine similarity
    cosine_sim = cosine_similarity(vec1, vec2)
    # Weighted combination (equal weights)
    combined_score = 0.5 * dice_sim + 0.5 * cosine_sim
    return combined_score
```

**Why Combine Both Measures:**

- **Complementary Strengths:** Set-based measures capture exact matches of trigrams, while cosine similarity considers the overall shape and magnitude of the embeddings.
- **Balanced Weighting:** Assigning equal weights ensures neither measure dominates, providing a more nuanced similarity score.
- **Improved Accuracy:** This approach leverages both the content overlap and the structural similarities, leading to more accurate and intuitive results.

---

## Optimizing the Code

During the implementation, I encountered several challenges that required code optimization and adjustments.

### Issue 1: Embedding Length Mismatch

**Problem:**

- **Variable Lengths:** Trigram hash embeddings varied in length depending on the length of the strings.
- **Shape Mismatch Errors:** When computing cosine similarity, embeddings of different lengths caused errors because vector operations require inputs of the same shape.

**Solution:**

- **Dynamic Padding:** Pad all embeddings to the same length based on the maximum length in the dataset.

**Updated Padding Function:**

```python
def get_max_embedding_length(embeddings_list):
    """Get the maximum length of embeddings."""
    return max(len(embedding) for embedding in embeddings_list)

def pad_embedding(embedding, max_length):
    """Pad or truncate embedding to a fixed length."""
    if len(embedding) < max_length:
        embedding += [0] * (max_length - len(embedding))
    else:
        embedding = embedding[:max_length]
    return embedding
```

**Explanation:**

- **Consistency:** Ensures all embeddings have the same length, allowing for valid vector operations.
- **Zero Padding:** Adds zeros to shorter embeddings, which doesn't significantly affect the cosine similarity calculation.

### Issue 2: Hash Collisions

**Problem:**

- **Hash Collisions:** Different trigrams could map to the same hash value, especially with a small `vocab_size`.
- **Reduced Uniqueness:** Collisions decrease the uniqueness of embeddings, leading to less accurate similarity measures.

**Solution:**

- **Increase `vocab_size`:** By increasing the `vocab_size`, the range of possible hash values expands, reducing the likelihood of collisions.

```python
def trigram_hash(word, vocab_size=32000):
    # ...
```

**Explanation:**

- **Trade-off:** A larger `vocab_size` requires more memory but improves the uniqueness of embeddings.
- **Practicality:** Choose a `vocab_size` that balances memory usage and collision reduction.

### Issue 3: Similarity Scores Feeling "Off"

**Problem:**

- **Unexpected Results:** Similarity scores didn't align with intuitive expectations, sometimes ranking unrelated titles highly.
- **Inadequate Measures:** Relying solely on one similarity measure didn't capture all aspects of similarity.

**Solution:**

- **Combined Similarity:** By combining the Dice coefficient and cosine similarity, I achieved a more balanced and accurate similarity score.

**Benefits:**

- **Captures Multiple Aspects:** The combined measure considers both content overlap and structural similarity.
- **More Intuitive Rankings:** Results better matched human expectations of similarity.

---

## Final Implementation

Putting it all together, here's the complete code:

```python
import numpy as np
import hashlib
import pymongo

# Trigram hashing function
def trigram_hash(word, vocab_size=32000):
    """Create a trigram hash for a given word."""
    word = "_" + word + "_"
    embeddings = []
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        hashed_trigram = int(hashlib.md5(trigram.encode('utf-8')).hexdigest(), 16) % vocab_size
        embeddings.append(hashed_trigram)
    return embeddings

# Get maximum embedding length
def get_max_embedding_length(embeddings_list):
    """Get the maximum length of embeddings."""
    return max(len(embedding) for embedding in embeddings_list)

# Padding function
def pad_embedding(embedding, max_length):
    """Pad or truncate embedding to a fixed length."""
    if len(embedding) < max_length:
        embedding += [0] * (max_length - len(embedding))
    else:
        embedding = embedding[:max_length]
    return embedding

# Dice coefficient function
def dice_coefficient(set1, set2):
    """Calculate Dice coefficient between two sets."""
    intersection = len(set1 & set2)
    total_size = len(set1) + len(set2)
    if total_size == 0:
        return 0.0
    return (2 * intersection) / total_size

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# Combined similarity function
def combined_similarity(set1, set2, vec1, vec2):
    """Combine Dice coefficient and cosine similarity."""
    dice_sim = dice_coefficient(set1, set2)
    cosine_sim = cosine_similarity(vec1, vec2)
    combined_score = 0.5 * dice_sim + 0.5 * cosine_sim
    return combined_score

# In-memory vector store
vector_store = {}

# Add to store function
def add_to_store(word, embedding):
    """Add word and its embedding to the store."""
    vector_store[word] = embedding

# Search store function
def search_store_combined(query_embedding, top_k=5):
    """Search for similar embeddings using combined similarity."""
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

# Example usage with MongoDB
# MongoDB connection (replace with your connection string)
mdb_client = pymongo.MongoClient('mongodb://127.0.0.1/?directConnection=true')
db = mdb_client['sample_mflix']
movies_collection = db['movies']

# Process movies from MongoDB
movies_cursor = movies_collection.find({}, {'title': 1})

# Add movie titles to vector store
for movie in movies_cursor:
    title = movie.get('title')
    if title:
        embedding = trigram_hash(title)
        add_to_store(title, embedding)

# Example search query
query_title = "The Matrix"
query_embedding = trigram_hash(query_title)

# Search for similar titles
similar_movies = search_store_combined(query_embedding)
for movie, similarity in similar_movies:
    print(f"{movie}: {similarity:.4f}")

"""
Top 5 similar movie titles to 'The Matrix':
The Matrix: 1.0
The Matriarch: 0.833208385651946
The Matrimony: 0.7725459392292349
The Mark: 0.7387173574923005
The Animatrix: 0.7356053620466556
"""
```

**Notes:**

- **Dynamic Padding:** Ensures embeddings are properly aligned for cosine similarity.
- **Combined Similarity Measure:** Provides a balanced score reflecting both content and structure.
- **In-Memory Store:** Stores embeddings for quick access and comparison.

---

## Results

With the combined similarity measure, the top similar movies to `"The Matrix"` are:

```
Top 5 similar movie titles to 'The Matrix':
The Matrix: 1.0
The Matriarch: 0.833208385651946
The Matrimony: 0.7725459392292349
The Mark: 0.7387173574923005
The Animatrix: 0.7356053620466556
```

**Analysis:**

- **Accurate Rankings:** The results are intuitive, with titles that are lexically similar ranking higher.
- **Balanced Similarity Scores:** The scores reflect both the shared trigrams (captured by the Dice coefficient) and the overall embedding similarity (captured by cosine similarity).
- **Contextual Relevance:** Titles like `"The Animatrix"` are appropriately ranked due to shared trigrams and contextual similarity.

---

## Lessons Learned

1. **Choose the Right Similarity Measure:**

   - Not all similarity measures are suitable for all types of embeddings.
   - **Discrete vs. Continuous Embeddings:** Discrete, sparse embeddings (like trigram hashes) are better compared using set-based measures, while continuous, dense embeddings suit vector-based measures.

2. **Beware of Hash Collisions:**

   - When hashing, ensure your `vocab_size` is large enough to minimize collisions.
   - **Trade-off Between Memory and Uniqueness:** Larger `vocab_size` reduces collisions but requires more memory.

3. **Consistent Embedding Lengths:**

   - Aligning the lengths of embeddings is essential for vector-based similarity calculations.
   - **Dynamic Padding:** Adjust embeddings to a common length to avoid shape mismatch errors.

4. **Combine Multiple Measures:**

   - Combining different similarity measures can yield better results than relying on a single one.
   - **Balanced Approach:** Leveraging both set-based and vector-based measures captures multiple aspects of similarity.

5. **Test and Iterate:**

   - Regularly test your implementation with real data to identify issues and areas for improvement.
   - **Iterative Development:** Don't hesitate to adjust parameters, try different measures, and refine your approach based on results.

---

## Conclusion

Implementing an effective text similarity search involves careful consideration of how text is represented and compared. By using trigram hashing and combining set-based and vector-based similarity measures, I was able to achieve more accurate and intuitive results.

This journey taught me the importance of selecting appropriate similarity measures for the type of embeddings used and the value of combining different approaches to capture the nuances of text similarity.

I hope this blog post helps others navigate similar challenges and accelerates their development process. Remember, experimentation and iteration are key to finding the optimal solution for your specific use case.

---

## References

- **Text Similarity Measures:** [Wikipedia](https://en.wikipedia.org/wiki/String_metric)
- **Trigram Analysis:** [Wikipedia on N-gram](https://en.wikipedia.org/wiki/N-gram)
- **Cosine Similarity:** [Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- **Jaccard Index:** [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- **Dice Coefficient:** [Wikipedia](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- **Hash Functions:** [MD5](https://en.wikipedia.org/wiki/MD5)
