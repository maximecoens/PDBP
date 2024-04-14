from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# Define two vectors
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

# Compute cosine similarity and cosine distance
similarity = cosine_similarity([vector1], [vector2])
distance = cosine_distances([vector1], [vector2])

print("Cosine similarity:", similarity)
print("Cosine distance:", distance)


from gensim import corpora, models, similarities
from gensim.matutils import softcossim

# Load the documents
doc1 = "This is a sample document."
doc2 = "Here is another document to compare."

# Create a dictionary from the documents
dictionary = corpora.Dictionary([doc1.split(), doc2.split()])

# Create a bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc.split()) for doc in [doc1, doc2]]

# Train a TF-IDF model on the corpus
tfidf = models.TfidfModel(corpus)

# Create a similarity matrix using the soft cosine similarity function
similarity_matrix = similarities.SoftCosineSimilarity(
    tfidf[corpus], similarity_matrix=None, num_features=len(dictionary))

# Compute the soft cosine similarity between the two documents
soft_similarity = softcossim(tfidf[dictionary.doc2bow(doc1.split())],
                              tfidf[dictionary.doc2bow(doc2.split())],
                              similarity_matrix)

print("Soft cosine similarity between the two documents: ", soft_similarity)