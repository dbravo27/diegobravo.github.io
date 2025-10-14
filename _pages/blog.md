---
layout: page
title: Blog
permalink: /blog/
---

# Blog

Insights on Machine Learning, AI, Data Science, and Mathematical Computing

---

<div class="blog-intro" markdown="1">

Welcome to my technical blog where I share insights from over 7 years of experience building production ML systems, conducting research, and leading data science teams. Topics include:

- **Machine Learning Engineering**: Production systems, MLOps, scalability
- **Deep Learning & NLP**: Transformers, BERT, generative AI
- **Mathematical Foundations**: Linear algebra, optimization, statistical theory
- **Cloud & Infrastructure**: Azure, AWS, Databricks
- **Leadership & Strategy**: Team management, project delivery, AI strategy

</div>

---

## Recent Posts

<div class="posts-list" markdown="1">

{% for post in site.posts %}
<article class="post-preview" markdown="1">

## [{{ post.title }}]({{ post.url }})

<div class="post-meta">
{{ post.date | date: "%B %d, %Y" }} {% if post.reading_time %}• {{ post.reading_time }} min read{% endif %}
</div>

{{ post.excerpt }}

<div class="post-tags">
{% for tag in post.tags %}
<span class="tag">{{ tag }}</span>
{% endfor %}
</div>

[Read more →]({{ post.url }})

</article>

<hr>

{% endfor %}

</div>

---

## Topics

<div class="topics-grid" markdown="1">

### 🤖 Machine Learning
Articles on production ML systems, model optimization, and best practices

### 🧠 Deep Learning
Neural networks, transformers, computer vision, and NLP techniques

### 📊 Data Engineering
Big data pipelines, cloud architecture, and data infrastructure

### 📈 Business Impact
Case studies showing measurable ROI from AI/ML initiatives

### 🔬 Research
Bridging academic research with practical applications

### 💡 Leadership
Technical team management and AI strategy

</div>

---

## Subscribe

*Stay updated with new posts on ML engineering, AI research, and data science leadership.*

[RSS Feed](/feed.xml)

---

## Categories

- [All Posts](/blog/)
- [Machine Learning](/blog/category/machine-learning/)
- [Deep Learning](/blog/category/deep-learning/)
- [MLOps](/blog/category/mlops/)
- [Data Engineering](/blog/category/data-engineering/)
- [Research](/blog/category/research/)
- [Case Studies](/blog/category/case-studies/)

---

<style>
.blog-intro {
  background: #f5f5f5;
  padding: 2rem;
  border-left: 4px solid #0066cc;
  margin: 2rem 0;
}

.posts-list {
  margin: 2rem 0;
}

.post-preview {
  margin: 2rem 0;
}

.post-preview h2 {
  margin-bottom: 0.5rem;
}

.post-meta {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

.post-tags {
  margin: 1rem 0;
}

.tag {
  display: inline-block;
  background: #e8f4f8;
  color: #0066cc;
  padding: 0.25rem 0.75rem;
  border-radius: 3px;
  font-size: 0.85rem;
  margin-right: 0.5rem;
}

.topics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.topics-grid h3 {
  color: #0066cc;
  border-bottom: 2px solid #0066cc;
  padding-bottom: 0.5rem;
}

@media (max-width: 768px) {
  .topics-grid {
    grid-template-columns: 1fr;
  }
}
</style>
