import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import load_and_preprocess
from knn_model import train_knn

def evaluate():
    user_movie, matrix = load_and_preprocess()
    model = train_knn(matrix)

    preds = []
    actuals = []

    for i in range(matrix.shape[0]):
        distances, indices = model.kneighbors([matrix[i]])
        neighbors = indices.flatten()[1:]

        pred = user_movie.iloc[neighbors].mean().mean()
        actual = user_movie.iloc[i].mean()

        preds.append(pred)
        actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)

    print("RMSE:", rmse)
    print("MAE:", mae)

if __name__ == "__main__":
    evaluate()
