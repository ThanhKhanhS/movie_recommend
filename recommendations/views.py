from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from django.shortcuts import render
import joblib
import pandas as pd
from thefuzz import process
import ast
import logging
import requests
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = '7f1241a08d90522cd83adf30cbc19d6f'
# Tải mô hình và dữ liệu
df = pd.read_csv('1000bophim.csv')
content_columns = {
    'content_overview': 0.3,
    'content_metadata': 0.1,
    'content_genres': 0.2,
    'content_keywords': 0.2,
    'content_companies': 0.05,
    'content_countries': 0.05,
    'content_numeric': 0.05,
    'content_time': 0.05
}

# Tạo ma trận TF-IDF cho mỗi cột content
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrices = {}
for col in content_columns.keys():
    df[col] = df[col].fillna('')
    tfidf_matrices[col] = tfidf.fit_transform(df[col])

# Tính toán ma trận similarity tổng hợp
def compute_weighted_similarity(matrices, weights):
    total_similarity = np.zeros((df.shape[0], df.shape[0]))
    for col, weight in weights.items():
        similarity = cosine_similarity(matrices[col])
        total_similarity += weight * similarity
    return total_similarity

cosine_sim = compute_weighted_similarity(tfidf_matrices, content_columns)
def find_closest_title(input_title, choices, limit=5):
    results = process.extract(input_title, choices, limit=limit)
    return results

def get_recommendations(title, cosine_sim=cosine_sim, n=10):
    try:
        possible_titles = find_closest_title(title, df['title'].tolist())
        
        if not possible_titles:
            logger.info(f"No matching titles found for '{title}'")
            return None

        best_match_title = possible_titles[0][0]
        idx = df.index[df['title'] == best_match_title].tolist()
        
        if not idx:
            logger.info(f"No index found for title '{best_match_title}'")
            return None
        
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[movie_indices].copy()

        def convert_genres(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return x.split(', ')
            return x

        recommendations.loc[:, 'genres'] = recommendations['genres'].apply(convert_genres)
        recommendations.loc[:, 'genres'] = recommendations['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return None
def get_cast(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        cast = data.get('cast', [])
        top_cast = [
            {
                'name': member['name'],
                'profile_path': f"https://image.tmdb.org/t/p/w200{member['profile_path']}" if member['profile_path'] else None
            }
            for member in cast[:5]  # Lấy top 5 diễn viên
        ]
        return top_cast
    else:
        logger.error(f"Failed to fetch cast for movie_id: {movie_id}, Status Code: {response.status_code}")
        return []

def index(request):
    recommendations = None
    if request.method == "POST":
        title = request.POST.get("title")
        recommendations = get_recommendations(title)
    
    return render(request, "index.html", {'recommendations': recommendations})

def movie_detail(request, movie_id):
    movie_id = int(movie_id)
    logger.info(f"Fetching details for movie_id: {movie_id}")
    movie = df[df['id'] == movie_id]
    
    if movie.empty:
        logger.warning(f"No movie found with id: {movie_id}")
        return render(request, 'movie_detail.html', {'error': 'Movie not found'})
    
    movie = movie.iloc[0]
    logger.info(f"Movie found in DataFrame: {movie['title']}")
    if isinstance(movie['genres'], str):
        try:
            # Sử dụng ast.literal_eval để chuyển đổi chuỗi thành danh sách
            movie['genres'] = ast.literal_eval(movie['genres'])
        except (ValueError, SyntaxError):
            # Nếu không thể chuyển đổi, tách chuỗi bằng dấu phẩy
            movie['genres'] = movie['genres'].split(', ')
    
    # Chuyển đổi danh sách genres thành chuỗi
    if isinstance(movie['genres'], list):
        movie['genres'] = ', '.join(movie['genres'])

    cast = get_cast(movie_id)
    return render(request, 'movie_detail.html', {'movie': movie, 'cast': cast})

    
