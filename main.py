from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import json
import os
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommendationSystem:
    POSTER_CACHE_FILE = "images/poster_cache.json"
    poster_data = {}

    def __init__(self, data_csv_path='movie_data.csv'):
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.content_similarity = None
        self.predicted_ratings_df = None
        self.user_movie_matrix = None
        self.movie_id_to_idx = None
        self.load_poster_data()
        self.load_data(data_csv_path)
        self.build_models()

    def load_poster_data(self):
        """Load poster URLs from the JSON file."""
        if os.path.exists(self.POSTER_CACHE_FILE):
            try:
                with open(self.POSTER_CACHE_FILE, 'r') as f:
                    self.poster_data = json.load(f)
                    self.poster_data = {int(k): v for k, v in self.poster_data.items()}
                logger.info(f"Loaded {len(self.poster_data)} posters from {self.POSTER_CACHE_FILE}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not load poster file: {e}")
                self.poster_data = {}
        else:
            logger.error(f"Poster file {self.POSTER_CACHE_FILE} not found")
            self.poster_data = {}

    def load_data(self, data_csv_path):
        """Load data from CSV."""
        try:
            full_df = pd.read_csv(data_csv_path)
            logger.info(f"Loaded raw data from {data_csv_path}")
            original_required_cols = ['MovieID', 'Title', 'Genres', 'UserID', 'Rating']
            if not all(col in full_df.columns for col in original_required_cols):
                raise ValueError(f"CSV is missing columns. Needed: {original_required_cols}")
            rename_map = {'MovieID': 'movie_id', 'Title': 'title', 'Genres': 'genres', 'UserID': 'user_id', 'Rating': 'rating'}
            full_df.rename(columns=rename_map, inplace=True)
            full_df['year'] = full_df['title'].str.extract(r'\((\d{4})\)', expand=False).fillna('2000').astype(int)
            full_df['title'] = full_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
            full_df['genres'] = full_df['genres'].str.replace('|', ',', regex=False)
            
            full_df['runtime'] = full_df.get('runtime', 'N/A')
            full_df['description'] = full_df.get('description', 'No description available for this movie.')
            
            popularity = full_df.groupby('movie_id')['rating'].count().reset_index(name='popularity_score')
            full_df = pd.merge(full_df, popularity, on='movie_id', how='left')
            
            self.ratings_df = full_df[['user_id', 'movie_id', 'rating']].copy()
            
            movie_cols = ['movie_id', 'title', 'year', 'genres', 'runtime', 'description', 'popularity_score']
            self.movies_df = full_df[movie_cols].drop_duplicates(subset=['movie_id']).copy()
            
            avg_ratings = full_df.groupby('movie_id')['rating'].mean().round(1).reset_index(name='avg_rating')
            self.movies_df = pd.merge(self.movies_df, avg_ratings, on='movie_id', how='left')
            self.movies_df.rename(columns={'movie_id': 'id', 'avg_rating': 'rating'}, inplace=True)
            self.movies_df.set_index('id', inplace=True, drop=False)
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split(',')
            
            logger.info(f"Created unique movies table with {len(self.movies_df)} movies. Server ready!")
        except FileNotFoundError:
            logger.error(f"Error: Data file not found. Make sure '{data_csv_path}' exists.")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def format_movies(self, movies_df, similarity_scores=None):
        """Format movies for API, using poster URLs from loaded data."""
        movies_list = []
        for _, movie in movies_df.iterrows():
            movie_id = int(movie['id'])
            poster_url = self.poster_data.get(movie_id, f"https://via.placeholder.com/300x450/333/fff?text={movie['title'].replace(' ', '+')}")
            movie_dict = {
                'id': movie_id,
                'title': movie['title'],
                'year': int(movie['year']),
                'genres': movie['genres'].split(',') if isinstance(movie['genres'], str) else [],
                'rating': float(movie.get('rating', 0.0)),
                'runtime': movie['runtime'],
                'description': movie['description'],
                'poster_url': poster_url,
                'popularity_score': int(movie.get('popularity_score', 0))
            }
            
            if similarity_scores and movie_id in similarity_scores:
                movie_dict['similarity_score'] = float(similarity_scores[movie_id])
            
            movies_list.append(movie_dict)
        
        return movies_list

    def build_models(self):
        """Build both content-based and collaborative filtering models."""
        try:
            self.build_content_based_model()
            self.build_collaborative_filtering_model()
            logger.info("Models built successfully")
        except Exception as e:
            logger.error(f"Error building models: {e}")
            raise

    def build_content_based_model(self):
        """Build content-based recommendation model using TF-IDF."""
        self.movies_df['description'] = self.movies_df['description'].fillna('')
        movies_in_order = self.movies_df.sort_index()
        
        features = []
        for _, movie in movies_in_order.iterrows():
            genre_text = ' '.join(movie['genres'].split(','))
            feature_text = f"{genre_text} {movie['description']}"
            features.append(feature_text)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(features)
        
        self.movie_id_to_idx = pd.Series(range(len(movies_in_order)), index=movies_in_order['id'])
        
        self.content_similarity = cosine_similarity(self.tfidf_matrix)
        logger.info("Content-based model built successfully")

    def build_collaborative_filtering_model(self):
        """Build collaborative filtering model using SVD."""
        try:
            self.user_movie_matrix = self.ratings_df.pivot(
                index='user_id',
                columns='movie_id',
                values='rating'
            ).fillna(0)
            
            user_ratings_mean = np.mean(self.user_movie_matrix.to_numpy(), axis=1)
            ratings_denorm = self.user_movie_matrix.to_numpy() - user_ratings_mean.reshape(-1, 1)
            n_features = self.user_movie_matrix.shape[1]
            
            k = min(50, n_features - 1) if n_features > 1 else 1
            
            U, sigma, Vt = svds(ratings_denorm, k=k)
            sigma = np.diag(sigma)
            self.predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            self.predicted_ratings_df = pd.DataFrame(
                self.predicted_ratings,
                index=self.user_movie_matrix.index,
                columns=self.user_movie_matrix.columns
            )
            
            logger.info(f"SVD-based collaborative filtering model built with {k} components")
        except Exception as e:
            logger.error(f"Error in collaborative filtering model: {e}")
            self.predicted_ratings_df = None

    def get_popular_movies(self, limit=12):
        """Get most popular movies based on rating count."""
        popular = self.movies_df.nlargest(limit, 'popularity_score')
        return self.format_movies(popular)

    def get_user_content_based_recommendations(self, user_id, limit=12):
        """Get content-based recommendations for a user based on their highly rated movies."""
        try:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            high_rated_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
            
            if not high_rated_movies:
                logger.warning(f"No highly rated movies found for user {user_id}")
                return self.get_popular_movies(limit)
            
            movie_scores = {}
            for movie_id in high_rated_movies:
                if movie_id not in self.movie_id_to_idx:
                    continue
                movie_idx = self.movie_id_to_idx[movie_id]
                similarity_scores = sorted(list(enumerate(self.content_similarity[movie_idx])),
                                        key=lambda x: x[1], reverse=True)
                top_similar = similarity_scores[1:limit//len(high_rated_movies) + 1]
                for idx, score in top_similar:
                    similar_movie_id = self.movie_id_to_idx.index[idx]
                    if similar_movie_id not in movie_scores:
                        movie_scores[similar_movie_id] = score
                    else:
                        movie_scores[similar_movie_id] = max(movie_scores[similar_movie_id], score)
            
            top_movie_ids = sorted(movie_scores, key=movie_scores.get, reverse=True)[:limit]
            similar_movies = self.movies_df.loc[top_movie_ids]
            return self.format_movies(similar_movies, similarity_scores=movie_scores)
            
        except Exception as e:
            logger.error(f"Error in user content-based recommendations: {e}")
            return self.get_popular_movies(limit)

    def get_similar_movies(self, title, limit=12):
        """Get similar movies based on a specific movie title using content similarity."""
        try:
            movie = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
            if movie.empty:
                logger.warning(f"Movie not found for title: {title}")
                return self.get_popular_movies(limit)
            
            movie_id = movie.index[0]
            if movie_id not in self.movie_id_to_idx:
                return self.get_popular_movies(limit)
            
            idx = self.movie_id_to_idx[movie_id]
            sim_scores = sorted(enumerate(self.content_similarity[idx]), key=lambda x: x[1], reverse=True)
            top_similar = sim_scores[1:limit + 1]  # exclude self
            
            movie_scores = {self.movie_id_to_idx.index[i]: score for i, score in top_similar}
            top_movie_ids = [self.movie_id_to_idx.index[i] for i, _ in top_similar]
            
            similar_movies = self.movies_df.loc[top_movie_ids]
            return self.format_movies(similar_movies, movie_scores)
            
        except Exception as e:
            logger.error(f"Error in similar movies for title {title}: {e}")
            return self.get_popular_movies(limit)

    def get_collaborative_recommendations(self, user_id=1, limit=12):
        """Get collaborative filtering recommendations for a user using SVD."""
        try:
            if self.predicted_ratings_df is None:
                logger.warning("Collaborative filtering model not available")
                return self.get_popular_movies(limit)
            
            if user_id not in self.user_movie_matrix.index:
                logger.warning(f"User {user_id} not found in ratings matrix")
                return self.get_popular_movies(limit)
            
            user_ratings = self.user_movie_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()
            predicted_ratings = self.predicted_ratings_df.loc[user_id, unrated_movies]
            
            recommended_ids = predicted_ratings.sort_values(ascending=False).index[:limit]
            
            if recommended_ids.empty:
                logger.warning("No collaborative recommendations found")
                return self.get_popular_movies(limit)
            
            recommended_movies = self.movies_df[self.movies_df['id'].isin(recommended_ids)]
            return self.format_movies(recommended_movies)
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return self.get_popular_movies(limit)

    def get_hybrid_recommendations(self, user_id=1, limit=12):
        """Get hybrid recommendations combining content-based and SVD-based approaches."""
        try:
            content_recs = self.get_user_content_based_recommendations(user_id, limit)
            collab_recs = self.get_collaborative_recommendations(user_id, limit)
            
            all_recommendations = content_recs + collab_recs
            seen_ids = set()
            unique_recommendations = []
            
            if self.predicted_ratings_df is not None and user_id in self.predicted_ratings_df.index:
                for movie in all_recommendations:
                    movie_id = movie['id']
                    if movie_id not in seen_ids:
                        if movie_id in self.predicted_ratings_df.columns:
                            predicted_rating = self.predicted_ratings_df.loc[user_id, movie_id]
                            movie['predicted_rating'] = float(predicted_rating)
                        unique_recommendations.append(movie)
                        seen_ids.add(movie_id)
                    if len(unique_recommendations) >= limit:
                        break
            else:
                unique_recommendations = all_recommendations[:limit]
            
            if self.predicted_ratings_df is not None:
                unique_recommendations.sort(key=lambda x: x.get('predicted_rating', 0), reverse=True)
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self.get_popular_movies(limit)

    def search_movies(self, query, limit=10):
        """Search movies by title, description, or genre with improved scoring."""
        try:
            query_lower = query.lower().strip()
            if not query_lower:
                return []
            
            movie_scores = []
            
            for _, movie in self.movies_df.iterrows():
                score = 0
                title_lower = movie['title'].lower()
                description_lower = movie['description'].lower()
                genres_lower = movie['genres'].lower()
                
                if query_lower in title_lower:
                    if title_lower.startswith(query_lower):
                        score += 100
                    elif title_lower == query_lower:
                        score += 150
                    else:
                        score += 50
                
                if query_lower in genres_lower:
                    score += 30
                
                if query_lower in description_lower:
                    score += 10
                
                score += movie['popularity_score'] * 0.1
                
                if score > 0:
                    movie_scores.append((movie['id'], score))
            
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_movie_ids = [movie_id for movie_id, score in movie_scores[:limit]]
            
            if not top_movie_ids:
                return []
            
            results = self.movies_df[self.movies_df['id'].isin(top_movie_ids)]
            results = results.set_index('id').reindex(top_movie_ids).reset_index()
            
            return self.format_movies(results)
            
        except Exception as e:
            logger.error(f"Error in movie search: {e}")
            return []

    def get_movie_details(self, movie_id):
        """Get detailed information for a specific movie."""
        try:
            movie = self.movies_df[self.movies_df['id'] == movie_id]
            if movie.empty:
                return None
            return self.format_movies(movie)[0]
        except Exception as e:
            logger.error(f"Error getting movie details for ID {movie_id}: {e}")
            return None

try:
    recommender = MovieRecommendationSystem('dataset/movies.csv')
    logger.info("Movie recommendation system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommendation system: {e}")
    recommender = None

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/v1/health')
def health_check():
    """Health check endpoint."""
    if recommender:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'total_movies': len(recommender.movies_df),
            'total_ratings': len(recommender.ratings_df),
            'poster_data_size': len(recommender.poster_data)
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Recommendation system not initialized',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/v1/movies/all')
def get_all_movies():
    """Endpoint to get all movies with pagination, sorting, search, and starts_with."""
    if not recommender:
        return jsonify({'error': 'Recommendation system not available'}), 500
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 24, type=int)
        sortBy = request.args.get('sortBy', 'popularity_score')
        search = request.args.get('search', '').lower().strip()
        starts_with = request.args.get('starts_with', '').upper().strip()
        
        page = max(1, page)
        limit = min(10000, max(1, limit))
        
        all_movies_df = recommender.movies_df.copy()
        
        if search:
            mask = (
                all_movies_df['title'].str.lower().str.contains(search, na=False) |
                all_movies_df['genres'].str.lower().str.contains(search, na=False) |
                all_movies_df['description'].str.lower().str.contains(search, na=False)
            )
            all_movies_df = all_movies_df[mask]
        
        if starts_with:
            all_movies_df = all_movies_df[all_movies_df['title'].str.upper().str.startswith(starts_with)]
        
        valid_sort_options = ['popularity_score', 'rating', 'year', 'title']
        if sortBy not in valid_sort_options:
            sortBy = 'popularity_score'
        
        ascending = True if sortBy == 'title' else False
        sorted_df = all_movies_df.sort_values(by=sortBy, ascending=ascending)
        
        start_index = (page - 1) * limit
        end_index = start_index + limit
        paginated_df = sorted_df.iloc[start_index:end_index]
        
        movies_list = recommender.format_movies(paginated_df)
        
        total_movies = len(sorted_df)
        total_pages = (total_movies + limit - 1) // limit
        
        return jsonify({
            'movies': movies_list,
            'total_movies': total_movies,
            'page': page,
            'limit': limit,
            'total_pages': total_pages,
            'sort_by': sortBy
        })
    except Exception as e:
        logger.error(f"Error in get_all_movies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/recommend')
def get_recommendations():
    """Get movie recommendations with multiple algorithms."""
    try:
        if not recommender:
            return jsonify({'error': 'Recommendation system not available'}), 500
        
        title = request.args.get('title')
        user_id = request.args.get('user_id', 1, type=int)
        algorithm = request.args.get('algorithm', 'popularity')
        limit = request.args.get('limit', 12, type=int)
        genre = request.args.get('genre')
        
        limit = min(50, max(1, limit))
        
        if title:
            recommendations = recommender.get_similar_movies(title, limit)
            algorithm_used = 'content_based'
        elif algorithm == 'collaborative':
            recommendations = recommender.get_collaborative_recommendations(user_id, limit)
            algorithm_used = 'collaborative_filtering'
        elif algorithm == 'content_based':
            recommendations = recommender.get_user_content_based_recommendations(user_id, limit)
            algorithm_used = 'content_based'
        elif algorithm == 'hybrid':
            recommendations = recommender.get_hybrid_recommendations(user_id, limit)
            algorithm_used = 'hybrid'
        else:
            recommendations = recommender.get_popular_movies(limit)
            algorithm_used = 'popularity_based'
        
        if genre:
            genre_list = [g.strip().lower() for g in genre.split(',')]
            recommendations = [
                movie for movie in recommendations
                if any(g.lower() in [genre.lower() for genre in movie['genres']] for g in genre_list)
            ]
        
        return jsonify({
            'movies': recommendations,
            'total': len(recommendations),
            'algorithm_used': algorithm_used,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/search')
def search_movies():
    """Search movies with improved relevance scoring."""
    try:
        if not recommender:
            return jsonify({'error': 'Recommendation system not available'}), 500
        
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 10, type=int)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        limit = min(50, max(1, limit))
        
        results = recommender.search_movies(query, limit)
        
        suggestions = [movie['title'] for movie in results[:5]]
        
        return jsonify({
            'movies': results,
            'total': len(results),
            'suggestions': suggestions,
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/movies/<int:movie_id>')
def get_movie_details(movie_id):
    """Get detailed movie information."""
    try:
        if not recommender:
            return jsonify({'error': 'Recommendation system not available'}), 500
        
        movie = recommender.get_movie_details(movie_id)
        
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        return jsonify({
            'movie': movie,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/users/<int:user_id>/ratings', methods=['POST'])
def add_user_rating(user_id):
    """Add or update user rating for a movie."""
    try:
        if not recommender:
            return jsonify({'error': 'Recommendation system not available'}), 500
        
        data = request.json
        if not data:
            return jsonify({'error': 'Request body required'}), 400
            
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        if not movie_id or rating is None:
            return jsonify({'error': 'movie_id and rating are required'}), 400
        
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            return jsonify({'error': 'Rating must be a number'}), 400
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        if movie_id not in recommender.movies_df['id'].values:
            return jsonify({'error': 'Movie not found'}), 404
        
        new_rating = pd.DataFrame([{
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        }])
        
        recommender.ratings_df = recommender.ratings_df[
            ~((recommender.ratings_df['user_id'] == user_id) &
              (recommender.ratings_df['movie_id'] == movie_id))
        ]
        recommender.ratings_df = pd.concat([recommender.ratings_df, new_rating], ignore_index=True)
        
        recommender.build_collaborative_filtering_model()
        
        logger.info(f"User {user_id} rated movie {movie_id} with {rating} stars")
        
        return jsonify({
            'message': 'Rating added successfully',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding rating: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors."""
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)