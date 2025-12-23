import numpy as np
from preprocess import load_and_preprocess
from knn_model import train_knn

def recommend_movies(user_id, top_n=5):
    user_movie, matrix = load_and_preprocess()
    model = train_knn(matrix)

    user_index = user_movie.index.tolist().index(user_id)

    distances, indices = model.kneighbors(
        [matrix[user_index]]
    )

    similar_users = indices.flatten()[1:]

    scores = np.zeros(user_movie.shape[1])

    for u in similar_users:
        scores += user_movie.iloc[u].values

    recommendations = (
        user_movie.columns[
            scores.argsort()[::-1]
        ]
    )

    watched = user_movie.iloc[user_index]
    recommendations = [
        m for m in recommendations
        if watched[m] == watched.mean()
    ]

    return recommendations[:top_n]

if __name__ == "__main__":
    print("Recommended movies:", recommend_movies(user_id=1))
