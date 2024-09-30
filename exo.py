import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data_path = r'C:/Users/acer/Desktop/TP/ml-100k//'
 

ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_ratings = pd.read_csv(data_path + 'u.data', sep='\t', names=ratings_columns, encoding='latin-1')

movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 
                  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                  'Sci-Fi', 'Thriller', 'War', 'Western']
movies_df = pd.read_csv(data_path + 'u.item', sep='|', names=movies_columns, encoding='latin-1')


movies_df = movies_df[['movie_id', 'title', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]


print("Premières lignes de df_ratings :")
print(df_ratings.head())
print("\nPremières lignes de movies_df :")
print(movies_df.head())

from sklearn.neighbors import NearestNeighbors
import numpy as np

user_item_matrix = df_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print("Matrice utilisateur-item (extrait) :")
print(user_item_matrix.head())

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)


user_id = 1
user_index = user_id - 1  # Si les IDs commencent à 1
distances, indices = knn.kneighbors([user_item_matrix.iloc[user_index]], n_neighbors=6)  # Inclut l'utilisateur lui-même

similar_users = indices.flatten()[1:]
similar_distances = distances.flatten()[1:]

print(f"Utilisateurs similaires à l'utilisateur {user_id} : {similar_users}")
print(f"Distances de similarité : {similar_distances}")



# Liste des colonnes de genres
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                 'Sci-Fi', 'Thriller', 'War', 'Western']

def combine_genres(row):
    genres = [genre for genre in genre_columns if row[genre] == 1]
    return ' '.join(genres)


movies_df['genres_combined'] = movies_df.apply(combine_genres, axis=1)

print("Exemple de genres combinés :")
print(movies_df[['title', 'genres_combined']].head())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres_combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    sim_scores = sim_scores[1:6]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df['title'].iloc[movie_indices]


print("\nRecommandations basées sur le contenu pour 'Toy Story (1995)' :")
print(content_based_recommendations('Toy Story (1995)'))

        