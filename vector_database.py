import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # IndexFlatIP para busquedas basadas en similitud de coseno
        self.documents = []

    def add_document(self, embedding, document):
        if len(embedding) != self.dimension:
            print(f"Error: Embedding dimension {len(embedding)} does not match expected dimension {self.dimension}")
            return
        # Normalizar embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        self.documents.append(document)
        self.index.add(np.array([normalized_embedding]).astype('float32'))

    def search(self, embedding, top_k=1):
        if len(embedding) != self.dimension:
            print(f"Error: Embedding dimension {len(embedding)} does not match expected dimension {self.dimension}")
            return []
        # Normalizar embedding
        normalized_embedding = embedding / np.linalg.norm(embedding)
        print(f"Searching for embedding: {normalized_embedding[:10]}...") 
        D, I = self.index.search(np.array([normalized_embedding]).astype('float32'), top_k)
        print("Distances: ", D)
        print("Indices: ", I)
        results = []
        for i in range(top_k):
            if I[0][i] != -1:
                results.append((D[0][i], self.documents[I[0][i]]))
        print("Search results: ", results)
        return results
