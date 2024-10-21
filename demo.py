import numpy as np
import hashlib
import pymongo
MDB_URI=""
def trigram_hash(word, vocab_size=32000):
    """Create a trigram hash for a given word using a stable hashing function."""
    word = "_" + word + "_"  # Adding delimiters to mark start and end
    embeddings = []
    
    for i in range(len(word) - 2):
        trigram = word[i:i+3]
        # Use MD5 to generate a stable hash and convert to an integer
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

# Dice coefficient for set-based similarity
def dice_coefficient(set1, set2):
    """Calculate Dice coefficient between two sets."""
    intersection = len(set(set1) & set(set2))
    return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) != 0 else 0

# Cosine similarity for vector-based similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

# Combined similarity (Dice coefficient + Cosine similarity)
def combined_similarity(set1, set2, vec1, vec2):
    """Combine Dice coefficient and cosine similarity for a more balanced measure."""
    # Dice coefficient on sets
    dice_sim = dice_coefficient(set1, set2)
    
    # Cosine similarity on vectors
    cosine_sim = cosine_similarity(vec1, vec2)
    
    # Weighted combination
    combined_score = 0.5 * dice_sim + 0.5 * cosine_sim
    return combined_score

# In-memory vector store (using a dictionary)
vector_store = {}

def add_to_store(word, embedding):
    """Add word and its trigram embedding to the in-memory store."""
    vector_store[word] = embedding

# Search store using combined similarity
def search_store_combined(query_embedding, top_k=5):
    """Search for the top_k most similar embeddings using combined similarity."""
    query_set = set(query_embedding)
    
    # Get the maximum length for padding
    max_length = get_max_embedding_length([query_embedding] + list(vector_store.values()))
    padded_query_embedding = pad_embedding(query_embedding, max_length)
    
    similarities = {}
    for word, stored_embedding in vector_store.items():
        stored_set = set(stored_embedding)
        padded_stored_embedding = pad_embedding(stored_embedding, max_length)
        similarity = combined_similarity(query_set, stored_set, padded_query_embedding, padded_stored_embedding)
        similarities[word] = similarity
    
    # Sort by highest similarity and return top_k results
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# MongoDB connection
mdb_client = pymongo.MongoClient(MDB_URI)
db = mdb_client['sample_mflix']
movies_collection = db['movies']

# Process movies from MongoDB
movies_cursor = movies_collection.find({}, {'title': 1})

# Adding movie titles to vector store
for movie in movies_cursor:
    title = movie.get('title')
    if title:
        print(f"Processing movie: {title}")
        embedding = trigram_hash(title)
        print(f"Trigram embedding for '{title}': {embedding}")
        add_to_store(title, embedding)

print("Finished embedding movie titles!")

# Example search query
query_title = "The Matrix"
query_embedding = trigram_hash(query_title)
print(f"Trigram embedding for query '{query_title}': {query_embedding}")

# Search with combined similarity
print(f"Top 5 similar movie titles to '{query_title}':")
similar_movies = search_store_combined(query_embedding)
for movie, similarity in similar_movies:
    print(f"{movie}: {similarity}")

"""
Top 5 similar movie titles to 'The Matrix':
The Matrix: 1.0
The Matriarch: 0.833208385651946
The Matrimony: 0.7725459392292349
The Mark: 0.7387173574923005
The Animatrix: 0.7356053620466556
"""
