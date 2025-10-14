---
layout: project
title: Multi-Tenant LMS Recommender System
tech_stack: PySpark, Databricks, Azure, ALS, NLP, Python
date: 2024-01-15
company: Cresteo
tags: [recommender-systems, machine-learning, big-data, nlp]
excerpt: Production-grade hybrid recommender system serving 267K users with 2.8M interactions, achieving 95% user coverage through collaborative filtering and NLP-based content analysis.
---

# Multi-Tenant LMS Recommender System

## Project Overview

Led the development of a production-grade recommender system for a multi-tenant Learning Management System (LMS) serving enterprise clients. The system processes 2.8 million learning interactions across 267,000 users and 395,000 courses, delivering personalized course recommendations at scale.

## Business Challenge

Learning management systems face a critical challenge: how to help learners discover relevant courses from massive catalogs. The client needed a solution that could:

- Handle multiple enterprise tenants with different learning catalogs
- Provide personalized recommendations for both active and new users (cold-start problem)
- Scale to millions of interactions without degrading performance
- Deploy across multiple cloud platforms (AWS, Azure, GCP)
- Maintain high recommendation quality despite data sparsity

## Technical Solution

### Architecture

Designed a **hybrid recommender pipeline** combining three complementary approaches:

1. **Collaborative Filtering (ALS)**
   - Implemented Alternating Least Squares (ALS) matrix factorization using PySpark MLlib
   - Captured user-course interaction patterns from 2.8M historical interactions
   - Identified latent factors representing learning preferences

2. **Content-Based Filtering (NLP)**
   - Extracted semantic features from course titles, descriptions, and metadata
   - Used TF-IDF and sentence transformers for content similarity
   - Enabled recommendations based on course content relationships

3. **Popularity-Based Models**
   - Incorporated trending courses and completion rates
   - Provided fallback recommendations for cold-start scenarios
   - Ensured diverse recommendation mix

### Key Technical Components

**Data Processing Pipeline (PySpark/Databricks)**
```python
# Example: Collaborative filtering with ALS
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Configure ALS model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="user_id",
    itemCol="course_id",
    ratingCol="interaction_score",
    coldStartStrategy="drop",
    nonnegative=True
)

# Train model on interaction data
model = als.fit(training_data)

# Generate top-N recommendations
user_recs = model.recommendForAllUsers(10)
```

**Content Similarity (NLP)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract course content features
vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words='english',
    ngram_range=(1, 2)
)

# Create content similarity matrix
course_features = vectorizer.fit_transform(course_descriptions)
similarity_matrix = cosine_similarity(course_features)

# Content-based recommendations
def get_similar_courses(course_id, top_n=10):
    idx = course_id_to_idx[course_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return [courses[i] for i in similar_indices]
```

**Hybrid Recommendation Engine**
```python
class HybridRecommender:
    def __init__(self, als_model, content_model, popularity_model):
        self.als_model = als_model
        self.content_model = content_model
        self.popularity_model = popularity_model

    def recommend(self, user_id, n_items=10, weights=None):
        """
        Generate hybrid recommendations combining multiple models

        Args:
            user_id: User identifier
            n_items: Number of recommendations
            weights: Dict with model weights (als, content, popularity)
        """
        if weights is None:
            weights = {'als': 0.5, 'content': 0.3, 'popularity': 0.2}

        # Get predictions from each model
        als_scores = self.als_model.predict(user_id, n_items * 2)
        content_scores = self.content_model.predict(user_id, n_items * 2)
        pop_scores = self.popularity_model.predict(user_id, n_items * 2)

        # Combine scores with weighted average
        combined_scores = self._combine_scores(
            als_scores, content_scores, pop_scores, weights
        )

        # Return top N recommendations
        return self._rank_and_filter(combined_scores, n_items)
```

### Intelligent Content Deduplication

Implemented sophisticated deduplication logic to handle:
- Duplicate courses across different tenants
- Same content with different titles/formats
- Course versions and updates

**Deduplication Strategy:**
```python
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

def deduplicate_courses(courses, similarity_threshold=0.85):
    """
    Identify and group duplicate/similar courses
    """
    # Extract content embeddings
    embeddings = get_course_embeddings(courses)

    # Cluster similar courses using DBSCAN
    clusterer = DBSCAN(
        eps=1-similarity_threshold,
        min_samples=2,
        metric='cosine'
    )

    clusters = clusterer.fit_predict(embeddings)

    # Keep highest quality course from each cluster
    deduplicated = []
    for cluster_id in set(clusters):
        if cluster_id == -1:  # Unique courses
            continue
        cluster_courses = courses[clusters == cluster_id]
        best_course = select_best_quality(cluster_courses)
        deduplicated.append(best_course)

    return deduplicated
```

### Cold-Start Handling

Addressed the cold-start problem for new users with limited interaction history:

1. **Onboarding questionnaire**: Captured initial preferences
2. **Demographic-based recommendations**: Used role, department, skills
3. **Popularity-weighted content filtering**: Combined trending with relevant content
4. **Progressive personalization**: Transitioned to collaborative filtering as interactions accumulated

### Multi-Cloud Deployment

Architected system for deployment across AWS, Azure, and GCP:

- **Data layer**: Abstracted storage (S3, Azure Blob, GCS)
- **Compute layer**: Containerized Spark jobs using Docker
- **Orchestration**: Apache Airflow for pipeline scheduling
- **API layer**: FastAPI for recommendation serving
- **Monitoring**: MLflow for experiment tracking and model versioning

## Results & Impact

### Performance Metrics

- **95% User Coverage**: Successfully generated recommendations for 95% of active users
- **~30% CTR**: Click-through rate on recommended courses
- **~15% Completion Lift**: Increased course completion rates
- **<100ms Latency**: Real-time recommendation serving

### Business Impact

- Enhanced learner engagement through personalized discovery
- Reduced time-to-skill by surfacing relevant learning paths
- Enabled data-driven content strategy for course providers
- Supported multi-tenant SaaS model with white-label capabilities

### Technical Achievements

- **Scalability**: Processed 2.8M interactions with sub-linear scaling
- **Model Performance**:
  - ALS RMSE: 0.82
  - Precision@10: 0.28
  - Recall@10: 0.35
- **System Reliability**: 99.9% uptime with automated failover
- **Deployment Flexibility**: Successfully deployed on AWS, Azure, and GCP

## Technical Stack

**Big Data & ML**
- PySpark (MLlib for ALS, DataFrame API for ETL)
- Databricks (unified analytics platform)
- Scikit-learn (content-based filtering)
- HuggingFace Transformers (sentence embeddings)

**Cloud & Infrastructure**
- Azure (primary deployment)
- AWS & GCP (multi-cloud support)
- Docker (containerization)
- Apache Airflow (orchestration)

**Model Operations**
- MLflow (experiment tracking, model registry)
- FastAPI (recommendation API)
- Redis (caching layer)
- PostgreSQL (metadata storage)

**NLP & Content Analysis**
- spaCy (text preprocessing)
- TF-IDF (feature extraction)
- Sentence-BERT (semantic embeddings)

## Key Learnings

1. **Hybrid > Single Algorithm**: Combining collaborative and content-based approaches significantly improved coverage and quality

2. **Cold-Start Strategy Critical**: Dedicated cold-start handling was essential for user experience in multi-tenant environment

3. **Content Quality Matters**: Deduplication and content normalization improved recommendation relevance by ~20%

4. **Monitor Diversity**: Preventing "filter bubble" required explicit diversity constraints in recommendation ranking

5. **A/B Testing Essential**: Continuous experimentation revealed user preferences varied significantly across tenant domains

## Future Enhancements

- **Deep Learning Models**: Experiment with neural collaborative filtering and attention mechanisms
- **Real-Time Learning**: Incorporate online learning for immediate personalization
- **Contextual Recommendations**: Add time-of-day, device, and session context
- **Explainability**: Provide transparent explanations for recommendations
- **Multi-Modal Content**: Incorporate video previews, instructor profiles

## Code Repository

*Note: Code is proprietary to Cresteo. Representative examples shown above.*

---

**Project Duration**: 8 months (ongoing maintenance)
**Team Size**: 3 data scientists, 2 ML engineers, 1 data engineer
**My Role**: Lead Data Scientist - Architecture design, model development, deployment strategy

[← Back to Projects](/projects)
