from sklearn.neighbors import NearestNeighbors

def train_knn(matrix, k=5):
    model = NearestNeighbors(
        n_neighbors=k,
        metric="cosine"
    )
    model.fit(matrix)
    return model
