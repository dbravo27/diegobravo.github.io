---
layout: post
title: "Building Production ML Systems: 5 Hard-Earned Lessons"
date: 2025-09-15
author: Dr. Diego Bravo
tags: [machine-learning, mlops, production, engineering, best-practices]
reading_time: 12
excerpt: After deploying dozens of ML systems serving millions of users, here are the most important lessons I've learned about building production-grade machine learning systems that actually deliver business value.
---

# Building Production ML Systems: 5 Hard-Earned Lessons

After seven years of building and deploying machine learning systems—from recommender systems serving millions of users to pricing optimization engines generating millions in revenue—I've learned that **getting a model to work in production is fundamentally different from getting it to work in a notebook**.

In this post, I'll share five hard-earned lessons that separate proof-of-concept ML from production-grade systems that deliver sustained business value.

---

## Lesson 1: Feature Engineering Matters More Than Model Architecture

### The Myth of the Perfect Algorithm

Early in my career, I spent weeks fine-tuning model architectures, trying XGBoost vs. LightGBM vs. CatBoost, obsessing over hyperparameters. I'd eke out a 0.5% improvement in validation accuracy and feel accomplished.

Then I'd deploy the model and realize the business impact was... minimal.

### What Actually Works

The breakthrough came when I shifted focus from **model selection** to **feature engineering**. In a recent pricing optimization project for a top US printing company:

- **Initial model** (basic features): XGBoost with MAPE of 8.2%
- **Same model with engineered features**: MAPE of 6.2%

That's a **24% error reduction** from features alone, with zero changes to the algorithm.

### Key Feature Engineering Strategies

**1. Domain-Specific Features**

Work closely with domain experts to create features that encode business logic:

```python
def engineer_domain_features(quote_data, customer_history):
    """
    Features informed by 20+ years of printing industry expertise
    """
    features = {}

    # Job complexity (from veteran estimators)
    features['complexity_score'] = (
        quote_data['num_colors'] *
        quote_data['num_sides'] *
        substrate_difficulty_map[quote_data['substrate']]
    )

    # Customer value (from sales team insights)
    features['customer_clv'] = customer_history['total_revenue'] * (
        1 + customer_history['avg_margin']
    )

    # Market timing (from operations)
    features['capacity_pressure'] = (
        current_backlog / shop_capacity *
        seasonal_demand_factors[quote_data['month']]
    )

    return features
```

**2. Interaction Features**

Don't just use raw features—capture how they interact:

```python
# Revenue growth × Market cap: Different implications for small vs large companies
features['growth_cap_interaction'] = (
    df['revenue_growth_rate'] * np.log(df['market_cap'])
)

# Customer frequency × Recency: Recent + frequent = high value
features['rfm_interaction'] = (
    df['order_frequency'] * np.exp(-df['days_since_last_order'] / 365)
)
```

**3. Temporal Features**

Time-based patterns are often the strongest signals:

```python
# Cyclical encoding for seasonality
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Trend and momentum
df['revenue_3m_trend'] = df.groupby('company_id')['revenue'].rolling(3).mean()
df['revenue_acceleration'] = df['revenue_3m_trend'].diff()
```

### The 80/20 Rule

In my experience: **80% of model performance comes from 20% of the features** (usually the carefully engineered ones), while the other 80% of features contribute only 20% of performance.

**Actionable takeaway**: Spend 3x more time on feature engineering than on model selection. Your future self will thank you.

---

## Lesson 2: Data Quality Trumps Data Quantity

### The "More Data" Fallacy

A common refrain in ML is "we need more data." But I've seen systems with millions of rows perform worse than systems with thousands of high-quality examples.

### Real-World Example: Recommender System

In a recommender system project for a learning management platform (2.8M interactions, 267K users):

**Initial approach**:
- Used all 2.8M interactions
- Many were "scroll past" views (<3 seconds)
- Model learned noise, not preferences
- Precision@10: 0.19

**Refined approach**:
- Filtered to meaningful interactions (>30 seconds engagement, completions, ratings)
- Reduced to 800K high-quality interactions
- Model learned true preferences
- Precision@10: 0.28 (**47% improvement**)

### Data Quality Checklist

Before training, ask:

1. **Representation**: Does the training data match production data distribution?
   ```python
   # Check for distribution shift
   from scipy.stats import ks_2samp

   for feature in features:
       statistic, pvalue = ks_2samp(train[feature], prod[feature])
       if pvalue < 0.05:
           print(f"Warning: {feature} distributions differ significantly")
   ```

2. **Labeling**: Are labels accurate and consistent?
   - In a loan approval NLP project, we found 15% label errors
   - Re-labeling improved F1 score from 0.82 to 0.90

3. **Freshness**: How quickly does data become stale?
   - Market data: Hours
   - Customer preferences: Weeks
   - Demographics: Months

4. **Bias**: Are important subgroups underrepresented?
   ```python
   # Check for class imbalance and subgroup representation
   print(df.groupby(['target', 'sensitive_attribute']).size())
   ```

### The Cost of Bad Data

In a $10M assortment optimization project for AB InBev:

- **Bad data cost**: Initial deployment used stale pricing data → $50K in suboptimal recommendations in first week
- **Good data value**: After implementing real-time data pipeline → $10M profit in 6 months

**Actionable takeaway**: Invest in data infrastructure and quality checks *before* scaling your model. Garbage in, garbage out—at scale.

---

## Lesson 3: Monitor Behavior, Not Just Metrics

### The Metric Illusion

Most ML monitoring focuses on model metrics: accuracy, RMSE, AUC. But I've seen models with perfect validation metrics fail catastrophically in production.

### What Actually Breaks in Production

Models fail in ways that don't show up in offline metrics:

**Case Study: Pricing Engine**

After deploying our pricing optimization model:

- ✅ **MAPE remained stable**: 6.2% (same as validation)
- ✅ **Latency was fine**: <100ms
- ❌ **Prices were wrong**: Recommending $5,000 for jobs that should be $50,000

**What happened?**

A rare combination of features (custom die-cut + metallic ink + rush delivery) never appeared in training data. Model extrapolated poorly.

**Offline metrics didn't catch this** because the combination was too rare (<0.01% of validation data).

### Behavioral Monitoring

Instead of just tracking metrics, monitor **model behavior**:

**1. Input Distribution Monitoring**

```python
from alibi_detect import KSDrift

# Detect distribution shift in real-time
drift_detector = KSDrift(
    reference_data=train_features,
    p_val=0.05
)

# In production
for batch in production_stream:
    drift_result = drift_detector.predict(batch)
    if drift_result['data']['is_drift']:
        alert_team(f"Feature drift detected: {drift_result['data']['distance']}")
```

**2. Prediction Range Monitoring**

```python
# Track if predictions stay in expected ranges
class PredictionMonitor:
    def __init__(self, historical_predictions):
        self.p01 = np.percentile(historical_predictions, 1)
        self.p99 = np.percentile(historical_predictions, 99)

    def check_prediction(self, pred):
        if pred < self.p01 or pred > self.p99:
            alert_team(f"Prediction {pred} outside historical range")
```

**3. Business Metric Monitoring**

The ultimate test: **is the model helping the business?**

```python
# A/B test monitoring
class BusinessMetricMonitor:
    def __init__(self):
        self.control_metrics = []
        self.treatment_metrics = []

    def log_outcome(self, group, converted, revenue):
        if group == 'control':
            self.control_metrics.append({'converted': converted, 'revenue': revenue})
        else:
            self.treatment_metrics.append({'converted': converted, 'revenue': revenue})

    def check_health(self):
        # Is ML version performing better?
        control_cvr = np.mean([m['converted'] for m in self.control_metrics])
        treatment_cvr = np.mean([m['converted'] for m in self.treatment_metrics])

        if treatment_cvr < control_cvr * 0.95:  # 5% degradation
            alert_team("Model underperforming baseline")
```

### Monitoring in Practice

At Cencosud (South America's largest retailer), we transformed MLOps from Level 0 to Level 1:

**Before**:
- Checked model accuracy weekly
- Discovered failures days after they occurred
- Lost ~$10K during each failure

**After**:
- Real-time behavioral monitoring
- Automated alerts for drift, anomalies, business metrics
- Caught and fixed issues in <1 hour
- Saved ~$120K annually

**Actionable takeaway**: Build monitoring into your deployment pipeline from day one. Monitor inputs, outputs, and business impact—not just model metrics.

---

## Lesson 4: Simplicity Scales, Complexity Breaks

### The Complexity Trap

As data scientists, we're trained to build sophisticated models. Neural networks! Transformers! Ensemble stacking!

But in production, **complexity is a liability**:

- More moving parts = more failure modes
- Harder to debug
- Slower to iterate
- Difficult to explain to stakeholders

### Start Simple, Add Complexity Only When Justified

**Example: Fashion Image Classification**

For a retail fashion project, I was tempted to build a state-of-the-art vision transformer:

**Complex approach (initial instinct)**:
- Vision Transformer (ViT) with 86M parameters
- Training time: 2 days on 4 GPUs
- Inference: 200ms per image
- Accuracy: 99.1%
- Deployment: Requires GPU inference (expensive)

**Simple approach (what we actually built)**:
- ResNet-50 (25M parameters) with transfer learning
- Training time: 4 hours on single GPU
- Inference: 20ms per image
- Accuracy: 99.0%
- Deployment: CPU inference (cheap)

**Result**: Saved $5K/month in infrastructure costs for 0.1% accuracy difference.

### When to Add Complexity

Add complexity only when:

1. **Simple approaches plateau**: Tried linear → tree-based → ensemble?
2. **Business value justifies it**: Will 1% improvement generate >$X revenue?
3. **You can maintain it**: Can your team debug and update the complex model?

### The "Explain It to Your CEO" Test

If you can't explain your model to a non-technical executive in 2 minutes, it's probably too complex for production.

**Examples**:

✅ **Good**: "We predict prices using historical winning quotes similar to the current one"

❌ **Bad**: "We use a gradient-boosted ensemble with CatBoost, LightGBM, and XGBoost stacked via ridge regression with feature engineering from a variational autoencoder"

### Architecture: Keep It Modular

```python
# Bad: Monolithic model
class ComplexModel:
    def predict(self, data):
        # 500 lines of feature engineering
        # 100 lines of preprocessing
        # Multiple models
        # Ensembling logic
        # Post-processing
        return result

# Good: Modular pipeline
class ModelPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = XGBoostModel()  # Simple, proven
        self.postprocessor = Postprocessor()

    def predict(self, data):
        data = self.preprocessor.transform(data)
        features = self.feature_engineer.transform(data)
        prediction = self.model.predict(features)
        return self.postprocessor.transform(prediction)
```

**Benefits**:
- Each component can be tested independently
- Easy to swap out pieces
- Clear debugging path

**Actionable takeaway**: Use the simplest model that meets business requirements. Complexity is a cost you pay in maintenance, debugging, and iteration speed.

---

## Lesson 5: The Model is Only 20% of the System

### The ML System Iceberg

When you deploy ML to production, the model is just the tip of the iceberg. The bulk of the work is infrastructure:

```
          [Model]  ← 20% of work
    ===============
    |              |
    |  Data Pipes  | ← 80% of work
    | Monitoring   |
    | Serving      |
    | Retraining   |
    | Versioning   |
    | Testing      |
    |______________|
```

### Real-World Example: Logistics Optimization

At Cresteo, we built a real-time optimization system for truck logistics:

**Time breakdown**:
- **Model development**: 3 weeks (20%)
- **Infrastructure**: 3 months (80%)
  - Data pipeline from GPS sensors
  - Real-time feature computation
  - Low-latency serving (<50ms)
  - A/B testing framework
  - Continuous retraining pipeline
  - Monitoring dashboards

### The 7 Components of Production ML

**1. Data Pipeline**
```python
# Daily ETL pipeline
@airflow_dag(schedule_interval='@daily')
def training_data_pipeline():
    raw_data = extract_from_sources()
    cleaned = clean_and_validate(raw_data)
    features = engineer_features(cleaned)
    store_to_warehouse(features)
```

**2. Feature Store**
```python
# Consistent features for training and serving
from feast import FeatureStore

fs = FeatureStore(repo_path=".")
features = fs.get_online_features(
    features=['user_ltv', 'product_popularity'],
    entity_rows=[{"user_id": 123}]
)
```

**3. Model Training Pipeline**
```python
# Automated retraining
@airflow_dag(schedule_interval='@weekly')
def train_and_deploy():
    data = load_training_data()
    model = train_model(data)

    if model.performance > current_model.performance:
        deploy_model(model)
        archive_old_model()
```

**4. Serving Infrastructure**
```python
# Low-latency serving
@app.post("/predict")
async def predict(request: PredictionRequest):
    features = await feature_store.get_online_features(request.id)
    prediction = model.predict(features)
    log_prediction(request, prediction)
    return prediction
```

**5. A/B Testing Framework**
```python
# Always validate with experiments
class ABTestingFramework:
    def get_model_version(self, user_id):
        if hash(user_id) % 100 < 10:  # 10% traffic
            return "new_model"
        else:
            return "current_model"
```

**6. Monitoring & Alerting**
```python
# Track everything
class ProductionMonitor:
    def log_prediction(self, input, output, latency):
        # Log to monitoring system
        metrics.log('prediction_latency', latency)
        metrics.log('prediction_value', output)

        # Check for anomalies
        if latency > SLA_THRESHOLD:
            alert_team('High latency detected')
```

**7. Model Registry**
```python
# Version control for models
import mlflow

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```

### The Business Impact

**MLOps at Cencosud**:

By investing in infrastructure, we:
- Reduced cloud costs by 15% (~$120K annually)
- Cut deployment time from 2 weeks to 1 day
- Increased experimentation velocity by 3x
- Improved model performance through faster iteration

**Actionable takeaway**: Budget 80% of project time for infrastructure. The model is important, but the system around it is what delivers sustained business value.

---

## Conclusion: From Notebooks to Production

The journey from Jupyter notebook to production ML system requires a fundamental shift in mindset:

| **Research Mindset**           | **Production Mindset**              |
|--------------------------------|-------------------------------------|
| Maximize accuracy              | Maximize business value             |
| Focus on algorithms            | Focus on features                   |
| More data is better            | Better data is better               |
| Track model metrics            | Track business metrics              |
| Build complex models           | Build maintainable systems          |
| Model is the deliverable       | System is the deliverable           |

### Your Production ML Checklist

Before deploying your next model, ask:

- [ ] Have I invested enough in feature engineering?
- [ ] Is my training data high-quality and representative?
- [ ] Do I have behavioral monitoring, not just metric monitoring?
- [ ] Is this the simplest model that meets business requirements?
- [ ] Do I have the infrastructure to maintain this system?

### Final Thoughts

After deploying dozens of ML systems—some wildly successful (generating millions in revenue), others spectacular failures (costing thousands before we caught them)—I've learned that **production ML is fundamentally an engineering discipline**.

The math and algorithms are important, but they're table stakes. What separates successful ML systems from failed POCs is:

1. **Thoughtful feature engineering** informed by domain expertise
2. **Ruthless focus on data quality** over data quantity
3. **Comprehensive monitoring** of behavior and business impact
4. **Disciplined simplicity** in model design
5. **Investment in infrastructure** to support the full ML lifecycle

Build systems, not just models. Your business stakeholders (and your future self) will thank you.

---

**About the Author**: Dr. Diego Bravo is a Senior Data Scientist and AI Leader with 7+ years of experience building production ML systems. He holds a PhD in Mathematics and has led teams deploying systems serving millions of users and generating millions in revenue. Currently at Cresteo, he specializes in recommender systems, pricing optimization, and generative AI.

**Interested in discussing production ML systems?** [Connect on LinkedIn](https://linkedin.com/in/diegobravoguerrero) or [email me](mailto:dbravo27@gmail.com).

---

*If you found this helpful, check out my other posts on [MLOps best practices](/blog/category/mlops/), [feature engineering](/blog/category/machine-learning/), and [AI strategy](/blog/category/case-studies/).*
