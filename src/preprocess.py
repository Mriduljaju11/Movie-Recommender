import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/ratings.csv"):
    df = pd.read_csv(path)

    user_movie = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    # Fill missing values with user mean
    user_movie = user_movie.apply(
        lambda row: row.fillna(row.mean()),
        axis=1
    )

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(user_movie)

    return user_movie, scaled_matrix
