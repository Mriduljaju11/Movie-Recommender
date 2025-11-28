# 🎬 Movie Recommendation System (scikit-learn)

A fully implemented movie recommendation system using **collaborative filtering**, **cosine similarity**, and **classical machine learning techniques**.  
The system analyzes user–movie rating patterns and recommends movies that a user is most likely to enjoy.

---

## 🚀 Features
- Personalized movie recommendations  
- **User-based** and **item-based** collaborative filtering  
- **KNN (k-nearest neighbors)** model using scikit-learn  
- Cosine similarity for user–movie closeness  
- Data preprocessing: handling missing values, normalization, pivot tables  
- Evaluation using **RMSE** and **MAE**  
- Visual analysis of rating distributions and user behavior  
- Clean, modular code structure

---

## 🧠 How the Recommendation Works

### 1️⃣ Data Preprocessing
- Load MovieLens dataset  
- Clean and encode user–movie rating matrix  
- Normalize rating vectors  
- Handle sparsity with imputation techniques  

### 2️⃣ Similarity Computation
- Compute **cosine similarity** for:
  - user–user similarity  
  - movie–movie similarity  

### 3️⃣ KNN Collaborative Filtering
- Find nearest neighbors  
- Aggregate neighbor ratings  
- Predict ratings for unseen movies  

### 4️⃣ Generate Recommendations
- Rank movies based on predicted rating  
- Filter out movies already watched  
- Return top-N recommendations  

---

## 📊 Evaluation Metrics
- **RMSE** (Root Mean Square Error)  
- **MAE** (Mean Absolute Error)  

---

## 📁 Project Structure

```plaintext
movie-recommender/
│── data/                     # MovieLens dataset
│── src/
│   ├── preprocess.py         # Preprocessing & cleaning
│   ├── similarity.py         # Cosine similarity functions
│   ├── knn_model.py          # KNN collaborative filtering
│   ├── recommend.py          # Recommendation generator
│   └── evaluate.py           # RMSE/MAE evaluation scripts
│── notebooks/
│   └── analysis.ipynb        # EDA & visualization
│── README.md
│── requirements.txt
```

---

## 📥 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mriduljaju11/Movie-Recommender.git
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Generate recommendations:
```bash
python src/recommend.py
```

### Train/evaluate the model:
```bash
python src/evaluate.py
```

---

## 📚 Dataset

Dataset used: **MovieLens 100k/1M**  
Download from: https://grouplens.org/datasets/movielens/  
Place the dataset files inside the `/data` folder.

---

## 🤝 Contributions
Open to improvements and enhancements — feel free to open issues or submit PRs!

---

## 📜 License
Released under the **MIT License**.
