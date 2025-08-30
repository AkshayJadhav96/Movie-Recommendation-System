document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION ---
    const API_BASE_URL = 'http://127.0.0.1:5000/api/v1';
    const CURRENT_USER_ID = 1;

    // --- DOM ELEMENT REFERENCES ---
    // Page containers
    const homePage = document.getElementById('home-page-content');
    const allMoviesPage = document.getElementById('all-movies-page-content');
    // Navigation
    const homeNav = document.querySelector('.nav-link[href="#home"]'); // Assuming you add #home to your nav link
    const moviesNav = document.querySelector('.nav-link[href="#movies"]'); // Assuming you add #movies
    // Home Page
    const moviesGrid = document.getElementById('movies-grid');
    const featured = { poster: document.getElementById('featured-poster'), title: document.getElementById('featured-title'), year: document.getElementById('featured-year'), genres: document.getElementById('featured-genres'), runtime: document.getElementById('featured-runtime'), rating: document.getElementById('featured-rating'), description: document.getElementById('featured-description') };
    const recommendationsTitle = document.getElementById('recommendations-title');
    const algorithmButtons = document.querySelectorAll('.algorithm-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    // All Movies Page
    const allMoviesGrid = document.getElementById('all-movies-grid');
    const allMoviesSearch = document.getElementById('movies-page-search');
    const allMoviesSort = document.getElementById('movies-page-sort');
    const loadMoreBtn = document.getElementById('load-more-all-movies-btn');
    // Modal & Toast
    const modal = { overlay: document.getElementById('modal-overlay'), closeBtn: document.getElementById('modal-close'), poster: document.getElementById('modal-poster-img'), title: document.getElementById('modal-title'), year: document.getElementById('modal-year'), genres: document.getElementById('modal-genres'), runtime: document.getElementById('modal-runtime'), ratingText: document.getElementById('modal-rating-text'), description: document.getElementById('modal-description'), getSimilarBtn: document.getElementById('get-similar-btn'), ratingStars: document.getElementById('rating-stars') };
    const toast = { element: document.getElementById('toast'), message: document.getElementById('toast-message') };
    
    // --- APPLICATION STATE ---
    let currentMovies = [];
    let currentAlgorithm = 'popularity';
    let activeMovieId = null;
    let allMoviesState = {
        page: 1,
        limit: 24,
        sortBy: 'popularity_score',
        searchQuery: '',
        isLoading: false,
        totalPages: 1
    };

    // --- API FUNCTIONS ---
    const api = { /* ... (API object remains the same as before) ... */ };
    api.get = async (endpoint) => { try { showLoading(true); const response = await fetch(`${API_BASE_URL}${endpoint}`); if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`); return await response.json(); } catch (error) { console.error('API GET Error:', error); showToast(`Error: ${error.message}`, 'error'); return null; } finally { showLoading(false); } };
    api.post = async (endpoint, body) => { try { const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`); return await response.json(); } catch (error) { console.error('API POST Error:', error); showToast(`Error: ${error.message}`, 'error'); return null; } };

    // --- RENDER FUNCTIONS ---
    const renderMovieGrid = (container, movies, append = false) => {
        if (!append) container.innerHTML = '';
        if (!movies || movies.length === 0) {
            if (!append) container.innerHTML = '<p class="no-movies-found">No movies found.</p>';
            return;
        }
        movies.forEach(movie => {
            const movieCard = document.createElement('div');
            movieCard.className = 'movie-card';
            movieCard.dataset.movieId = movie.id;
            const scoreHtml = movie.similarity_score ? `<div class="similarity-score">🔥 ${Math.round(movie.similarity_score * 100)}% Match</div>` : '';
            movieCard.innerHTML = `
                ${scoreHtml}
                <img src="${movie.poster_url || 'https://via.placeholder.com/300x450/333/fff?text=No+Poster'}" alt="${movie.title}" loading="lazy">
                <div class="movie-card-info"><h3>${movie.title}</h3><p>${movie.year}</p></div>`;
            movieCard.addEventListener('click', () => openMovieModal(movie.id));
            container.appendChild(movieCard);
        });
    };

    const updateFeaturedMovie = (movie) => { /* ... (remains the same) ... */ };
    const populateModal = (movie) => { /* ... (remains the same) ... */ };
    const showLoading = (isLoading) => { loadingIndicator.style.display = isLoading ? 'flex' : 'none'; };
    const showToast = (message, type = 'success') => { /* ... (remains the same) ... */ };
    // Helper to fill modal and featured movie
    updateFeaturedMovie = (movie) => { if (!movie) return; featured.poster.src = movie.poster_url; featured.title.textContent = movie.title; featured.year.textContent = movie.year; featured.genres.textContent = movie.genres.join(', '); featured.runtime.textContent = movie.runtime; featured.rating.textContent = `${movie.rating.toFixed(1)}/10`; featured.description.textContent = movie.description; };
    populateModal = (movie) => { activeMovieId = movie.id; modal.poster.src = movie.poster_url; modal.title.textContent = movie.title; modal.year.textContent = movie.year; modal.genres.textContent = movie.genres.join(', '); modal.runtime.textContent = movie.runtime; modal.ratingText.textContent = `${movie.rating.toFixed(1)}/10 IMDb`; modal.description.textContent = movie.description; modal.overlay.classList.add('active'); };
    showToast = (message, type = 'success') => { toast.message.textContent = message; toast.element.className = `toast show ${type}`; setTimeout(() => { toast.element.className = 'toast'; }, 3000); };
    
    // --- CORE LOGIC & EVENT HANDLERS ---
    
    // Home Page Logic
    const fetchAndRenderRecommendations = async (algorithm = 'popularity', title = null) => {
        currentAlgorithm = algorithm;
        recommendationsTitle.textContent = `Based on: ${algorithm.replace('_', ' ')}`;
        let endpoint = `/recommend?algorithm=${algorithm}&user_id=${CURRENT_USER_ID}&limit=20`;
        if (title) {
            endpoint = `/recommend?title=${encodeURIComponent(title)}&limit=20`;
            recommendationsTitle.textContent = `Similar to: ${title}`;
        }
        const data = await api.get(endpoint);
        if (data && data.movies) {
            currentMovies = data.movies;
            renderMovieGrid(moviesGrid, currentMovies); // Render to home page grid
            if (!title) updateFeaturedMovie(currentMovies[0]);
        }
    };
    
    // All Movies Page Logic
    const fetchAndRenderAllMovies = async (append = false) => {
        if (allMoviesState.isLoading) return;
        allMoviesState.isLoading = true;
        
        if (append) {
            allMoviesState.page++;
        } else {
            allMoviesState.page = 1; // Reset for new search/sort
        }
        
        let endpoint = `/movies/all?page=${allMoviesState.page}&limit=${allMoviesState.limit}&sortBy=${allMoviesState.sortBy}`;
        // Note: Full search on the 'all movies' page will be client-side for now for simplicity,
        // but a proper implementation would have a backend search endpoint.
        
        const data = await api.get(endpoint);
        if (data && data.movies) {
            allMoviesState.totalPages = data.total_pages;
            renderMovieGrid(allMoviesGrid, data.movies, append);
            loadMoreBtn.style.display = allMoviesState.page >= allMoviesState.totalPages ? 'none' : 'block';
        }
        allMoviesState.isLoading = false;
    };
    
    const openMovieModal = async (movieId) => {
        const data = await api.get(`/movies/${movieId}`);
        if (data && data.movie) populateModal(data.movie);
    };

    // View Switching
    const switchView = (view) => {
        if (view === 'home') {
            homePage.style.display = 'block';
            allMoviesPage.style.display = 'none';
        } else if (view === 'movies') {
            homePage.style.display = 'none';
            allMoviesPage.style.display = 'block';
            // If the grid is empty, do an initial fetch
            if (allMoviesGrid.innerHTML === '') {
                fetchAndRenderAllMovies();
            }
        }
    };

    // Debounce for search input
    const debounce = (func, delay) => { let timeout; return (...args) => { clearTimeout(timeout); timeout = setTimeout(() => func.apply(this, args), delay); }; };

    // --- INITIALIZATION ---
    const init = () => {
        // Event Listeners
        algorithmButtons.forEach(btn => btn.addEventListener('click', (e) => {
             algorithmButtons.forEach(b => b.classList.remove('active'));
             e.target.classList.add('active');
             fetchAndRenderRecommendations(e.target.dataset.algorithm);
        }));
        
        modal.closeBtn.addEventListener('click', () => modal.overlay.classList.remove('active'));
        modal.overlay.addEventListener('click', (e) => { if (e.target === modal.overlay) modal.overlay.classList.remove('active'); });
        modal.getSimilarBtn.addEventListener('click', () => { modal.overlay.classList.remove('active'); fetchAndRenderRecommendations('content_based', modal.title.textContent); });
        
        // New Listeners for All Movies Page
        loadMoreBtn.addEventListener('click', () => fetchAndRenderAllMovies(true));
        allMoviesSort.addEventListener('change', (e) => {
            allMoviesState.sortBy = e.target.value;
            fetchAndRenderAllMovies(false); // Fetch from page 1 with new sort
        });

        // Simple Client-side Search for All Movies Page
        allMoviesSearch.addEventListener('input', debounce((e) => {
            const query = e.target.value.toLowerCase().trim();
            const cards = allMoviesGrid.querySelectorAll('.movie-card');
            cards.forEach(card => {
                const title = card.querySelector('h3').textContent.toLowerCase();
                card.style.display = title.includes(query) ? 'block' : 'none';
            });
        }, 300));

        // Navigation
        // This is a simple way to handle nav. For a real app, a router library is better.
        document.querySelector('nav').addEventListener('click', (e) => {
            if (e.target.matches('.nav-link')) {
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
                if (e.target.textContent === 'Movies') {
                    switchView('movies');
                } else if (e.target.textContent === 'Home') {
                    switchView('home');
                }
            }
        });
        
        // Initial Load
        switchView('home'); // Start on the home page
        fetchAndRenderRecommendations('popularity');
    };

    init();
});