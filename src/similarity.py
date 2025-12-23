from sklearn.metrics.pairwise import cosine_similarity

def compute_user_similarity(matrix):
    return cosine_similarity(matrix)

def compute_item_similarity(matrix):
    return cosine_similarity(matrix.T)
