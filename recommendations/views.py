from django.shortcuts import render
import joblib
import pandas as pd
from thefuzz import process
import ast
import logging
import requests
from huggingface_hub import hf_hub_download
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = '7f1241a08d90522cd83adf30cbc19d6f'
# Tải mô hình và dữ liệu
tfidf = joblib.load('tfidf_model.pkl')
cosine_sim = joblib.load('cosine_similarity_matrix.pkl')
df = pd.read_csv('processed_movie_data_10k.csv')

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

    
