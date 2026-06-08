# Hybrid Movie Recommendation System

## Overview

This project develops a personalized movie recommendation engine using the MovieLens dataset. The system combines multiple recommendation strategies, including collaborative filtering, content-based filtering, matrix factorization, and hybrid recommendation techniques to generate personalized movie suggestions.

Rather than relying on a single recommendation algorithm, the project explores and compares multiple approaches before combining them into a hybrid recommendation framework that leverages the strengths of each method.

---

## Motivation

Modern streaming platforms such as Netflix, Amazon Prime, and Disney+ rely heavily on recommendation systems to improve user experience and content discovery.

However, a single recommendation strategy often suffers from limitations:

* Collaborative filtering struggles with cold-start problems.
* Content-based filtering may overspecialize recommendations.
* Matrix factorization models can miss rich content information.

This project investigates multiple recommendation paradigms and combines them into a hybrid recommendation system to generate more relevant and personalized recommendations.

---

## Dataset

### MovieLens 1M Dataset

The system is built using the MovieLens 1M dataset.

Dataset characteristics:

* Over 1 million movie ratings
* Approximately 6,000 users
* Approximately 4,000 movies
* User demographic information
* Movie metadata and genres

The dataset provides explicit user ratings that enable both collaborative and content-based recommendation approaches.

---

## Recommendation Approaches

### 1. Item-Item Collaborative Filtering

This approach recommends movies based on similarities between movies.

If users who liked Movie A also liked Movie B, the system learns that the two movies are related.

#### Methodology

* Compute movie-to-movie similarity
* Identify common reviewers
* Calculate similarity scores
* Recommend similar movies

#### Advantages

* Simple and interpretable
* Captures collective user preferences

---

### 2. User-User Collaborative Filtering

This approach identifies users with similar rating behavior.

Movies highly rated by similar users are recommended to the target user.

#### Methodology

* Compute user similarity
* Identify nearest neighbors
* Predict ratings using neighboring users
* Generate recommendations

#### Advantages

* Produces highly personalized recommendations
* Learns latent preference patterns

---

### 3. Matrix Factorization using SVD

To uncover hidden user preferences, Singular Value Decomposition (SVD) is applied to the user-item rating matrix.

#### Methodology

* Construct user-movie rating matrix
* Perform matrix factorization using SVD
* Learn latent user and movie representations
* Predict unseen ratings

SVD enables the system to discover hidden relationships that are not directly observable from explicit ratings.

---

### 4. Content-Based Filtering

Content-based recommendations are generated using movie genre information.

#### Methodology

* Extract movie genre metadata
* Generate TF-IDF representations
* Compute cosine similarity between movies
* Recommend movies with similar content profiles

Example:

A user who enjoys science-fiction movies is more likely to receive recommendations from the same genre family.

---

## Hybrid Recommendation Model

The final system combines:

* Content-Based Filtering
* SVD-Based Collaborative Filtering

### Hybrid Workflow

```text
User History
      │
      ▼
Content-Based Filtering
      │
      ▼
Candidate Movies
      │
      ▼
SVD Rating Prediction
      │
      ▼
Final Ranking
      │
      ▼
Recommended Movies
```

The content-based model identifies relevant candidate movies, while the SVD model predicts personalized ratings and ranks the recommendations.

This combination improves recommendation quality by incorporating both movie content information and collaborative user behavior.

---

## System Architecture

```text
MovieLens Dataset
        │
        ▼
Data Preprocessing
        │
        ├─────────────┐
        ▼             ▼
Collaborative     Content-Based
Filtering         Filtering
        │             │
        └──────┬──────┘
               ▼
        Hybrid Model
               ▼
     Personalized Recommendations
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Surprise Library
* SciPy
* Matplotlib
* Jupyter Notebook

---

## Key Techniques

### Collaborative Filtering

* User-User Similarity
* Item-Item Similarity
* Nearest Neighbor Recommendation

### Matrix Factorization

* Singular Value Decomposition (SVD)
* Latent Factor Modeling

### Content-Based Filtering

* TF-IDF Vectorization
* Cosine Similarity

### Hybrid Recommendation

* Candidate Generation
* Personalized Ranking

---

## Example Recommendation Pipeline

1. User watches and rates movies.
2. Similar users and movies are identified.
3. Content similarity is computed using genres.
4. SVD predicts ratings for unseen movies.
5. Hybrid model ranks candidate movies.
6. Top recommendations are returned.
---

## Future Improvements

* Deep learning based recommenders
* Context-aware recommendations
* Real-time recommendation serving

---

## Learning Outcomes

Through this project, the following recommendation techniques were implemented and studied:

* User-User Collaborative Filtering
* Item-Item Collaborative Filtering
* Matrix Factorization (SVD)
* Content-Based Filtering
* Hybrid Recommendation Systems

The project demonstrates how multiple recommendation paradigms can be integrated into a single recommendation engine to improve personalization and recommendation quality.

