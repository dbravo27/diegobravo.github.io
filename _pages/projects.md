---
layout: page
title: Projects
permalink: /projects/
---

A selection of impactful machine learning and AI projects demonstrating expertise across recommender systems, optimization, NLP, computer vision, and generative AI.

---

## Enterprise-Scale AI Solutions

<div class="project-grid" markdown="1">

{% for project in site.projects %}
<div class="project-card" markdown="1">

### [{{ project.title }}]({{ site.baseurl }}{{ project.url }})

{{ project.excerpt }}

**Tech Stack:** {{ project.tech_stack }}

[Read more →]({{ site.baseurl }}{{ project.url }})

</div>
{% endfor %}

</div>

---

## Project Categories

### Recommender Systems
- **Multi-Tenant LMS Recommender** - Hybrid recommendation pipeline serving 267K users with 95% coverage
- **Retail Assortment Optimization** - $10M revenue boost serving 300K stores

### Pricing & Optimization
- **ML Pricing Engine** - 97% reduction in quote turnaround time for printing industry
- **Real Estate Price Forecasting** - 20% increase in sales conversions

### Generative AI & NLP
- **AI-Powered Sales Agent** - 20+ page financial reports with interactive chat capabilities
- **Customer Sentiment Analysis** - BERT-based pipeline for e-commerce feedback
- **Legal Document Processing** - 70% efficiency boost in FinTech loan approvals

### Forecasting & Analytics
- **Inventory Optimization** - 5% stockout reduction, 10% waste reduction using Prophet/XGBoost
- **Logistics Real-Time Optimization** - 40% operational efficiency improvement

### Computer Vision
- **Fashion Image Classification** - 99% accuracy CNN model for retail applications
- **In-Store Customer Analytics** - Computer vision for mall behavior tracking

### MLOps & Infrastructure
- **MLOps Transformation** - Level 0 to Level 1 advancement, $120K annual cost savings
- **AI Courses Platform** - 30+ courses for 400+ data scientists

---

## Impact Metrics

<div class="metrics-grid" markdown="1">

**$10M+**
Revenue Generated

**97%**
Quote Time Reduction

**70%**
Efficiency Boost

**95%**
User Coverage

**99%**
Model Accuracy

**40%**
Operations Improvement

</div>

---

## Research Publications

In addition to industry projects, I maintain an active research program with **11 published papers** in top-tier mathematics journals:

- **Linear Algebra and its Applications** (3 papers)
- **Journal of Algebraic Combinatorics**
- **Journal of Algebra**
- **Proceedings of the American Mathematical Society**
- **Rocky Mountain Journal of Mathematics**

Topics include graph theory, spectral analysis, homological algebra, and representation theory.

[View all publications on CV page →]({{ site.baseurl }}/cv/#publications)

---

## Open Source & Community

I'm committed to contributing to the data science community through:

- **Technical Leadership**: Led teams of 30+ data scientists
- **Education**: Created 30+ Python and Data Science courses
- **Mentorship**: Advised 1 PhD and 2 Master's theses
- **Speaking**: Presented at 20+ international conferences

---

## Technologies Used Across Projects

**Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Prophet, Optuna, AutoML

**Deep Learning:** PyTorch, TensorFlow, Keras, Transformers, BERT, OpenCV, YOLO

**NLP:** spaCy, HuggingFace, BERT, Named Entity Recognition, Topic Modeling

**Generative AI:** Azure OpenAI, OpenAI, Gemini API, LangChain, RAG systems

**Big Data:** PySpark, Databricks, Apache Airflow, DVC

**Cloud:** Azure, AWS, GCP

**Deployment:** Docker, FastAPI, GitHub Actions, Streamlit, Gradio, Azure Web Apps

---

*Interested in collaborating or learning more about any of these projects? [Get in touch!](mailto:dbravo27@gmail.com)*

<style>
.project-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
  margin: 2rem 0;
}

.project-card {
  border: 1px solid #ddd;
  padding: 1.5rem;
  border-radius: 8px;
  background: #f9f9f9;
  transition: transform 0.2s, box-shadow 0.2s;
}

.project-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.project-card h3 {
  color: #0066cc;
  margin-top: 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
  text-align: center;
}

.metrics-grid strong {
  display: block;
  font-size: 2rem;
  color: #0066cc;
  margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
  .project-grid {
    grid-template-columns: 1fr;
  }
}
</style>
