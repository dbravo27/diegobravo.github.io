---
layout: project
title: ML-Powered Pricing Optimization Engine
tech_stack: XGBoost, Optuna, Python, Azure, FastAPI, Docker
date: 2024-06-10
company: Cresteo (Client: Top U.S. Printing Company)
tags: [machine-learning, optimization, pricing, xgboost]
excerpt: ML platform reducing quote turnaround from 24h to <2h (97% on-time), delivering 12% profit lift through intelligent pricing recommendations for commercial printing.
---

# ML-Powered Pricing Optimization Engine

## Project Overview

Designed and deployed a machine learning platform that transformed the pricing process for one of the top commercial printing companies in the United States. The system uses advanced ML algorithms to recommend optimal prices in real-time, reducing quote turnaround time from 24 hours to under 2 hours while delivering an estimated 12% profit lift.

## Business Challenge

Commercial printing is a highly competitive, price-sensitive industry where quoting speed and accuracy directly impact win rates. The client faced several challenges:

### Pain Points

- **Slow Turnaround**: Manual pricing process took 24+ hours, losing deals to faster competitors
- **Inconsistent Pricing**: Different estimators produced varying quotes for similar jobs
- **Suboptimal Margins**: Conservative pricing left money on the table; aggressive pricing lost deals
- **Limited Market Insight**: Difficulty incorporating competitive intelligence and market trends
- **Complex Variables**: 50+ features affecting price (substrate, quantity, finishing, delivery, etc.)

### Business Requirements

- Reduce quote turnaround to <2 hours
- Maintain or improve profit margins
- Increase quote acceptance rate
- Provide confidence intervals and explanation for recommendations
- Integrate seamlessly with existing ERP system

## Technical Solution

### Data Strategy

**Historical Data Analysis (5 years)**
- 150K+ completed quotes (won and lost)
- 50+ features per quote
- Win/loss outcomes and actual margins
- Customer historical purchase patterns
- Seasonal and market trend indicators

**Feature Engineering**

Developed 120+ engineered features across multiple categories:

```python
class PricingFeatureEngine:
    """
    Advanced feature engineering for pricing optimization
    """

    def engineer_features(self, quote_data):
        """Generate comprehensive feature set for pricing model"""

        features = {}

        # 1. Product complexity features
        features.update(self._compute_complexity_score(quote_data))

        # 2. Customer value features
        features.update(self._compute_customer_features(quote_data))

        # 3. Market condition features
        features.update(self._compute_market_features(quote_data))

        # 4. Operational capacity features
        features.update(self._compute_capacity_features(quote_data))

        # 5. Competition features
        features.update(self._compute_competition_features(quote_data))

        return features

    def _compute_complexity_score(self, quote):
        """Job complexity affects production cost and risk"""
        return {
            'complexity_score': self._calculate_complexity(quote),
            'finish_difficulty': self._finish_complexity(quote),
            'substrate_premium': self._substrate_cost_multiplier(quote),
            'color_complexity': quote['num_colors'] * quote['num_sides'],
            'custom_work_flag': int(quote['has_custom_requirements'])
        }

    def _compute_customer_features(self, quote):
        """Customer history influences pricing strategy"""
        customer_id = quote['customer_id']
        history = self.customer_db.get_history(customer_id)

        return {
            'customer_lifetime_value': history['total_revenue'],
            'customer_frequency': history['order_count'],
            'avg_margin_with_customer': history['avg_margin'],
            'payment_reliability_score': history['payment_score'],
            'price_sensitivity': self._estimate_elasticity(history),
            'days_since_last_order': (datetime.now() - history['last_order']).days
        }

    def _compute_market_features(self, quote):
        """Market conditions affect competitive dynamics"""
        return {
            'seasonal_demand_factor': self._get_seasonal_index(quote['quote_date']),
            'competitive_intensity': self._get_market_pressure(quote['region']),
            'paper_cost_index': self._get_commodity_index('paper', quote['quote_date']),
            'regional_price_index': self._get_regional_index(quote['region'])
        }
```

### ML Model Architecture

**Ensemble Approach**: Combined multiple XGBoost models for robust predictions

```python
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

class PricingEnsemble:
    """
    Ensemble of XGBoost models for pricing prediction
    """

    def __init__(self):
        self.price_model = None      # Predicts optimal price
        self.win_prob_model = None   # Predicts win probability
        self.margin_model = None     # Predicts expected margin

    def train(self, X, y_price, y_win, y_margin):
        """Train ensemble of models"""

        # Hyperparameter optimization with Optuna
        price_params = self._optimize_hyperparameters(
            X, y_price, objective='reg:squarederror'
        )

        win_params = self._optimize_hyperparameters(
            X, y_win, objective='binary:logistic'
        )

        margin_params = self._optimize_hyperparameters(
            X, y_margin, objective='reg:squarederror'
        )

        # Train final models
        self.price_model = xgb.XGBRegressor(**price_params)
        self.price_model.fit(X, y_price)

        self.win_prob_model = xgb.XGBClassifier(**win_params)
        self.win_prob_model.fit(X, y_win)

        self.margin_model = xgb.XGBRegressor(**margin_params)
        self.margin_model.fit(X, y_margin)

    def _optimize_hyperparameters(self, X, y, objective):
        """Bayesian hyperparameter optimization using Optuna"""

        def objective_function(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'objective': objective
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if objective == 'binary:logistic':
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, preds)
                else:
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = -mean_absolute_percentage_error(y_val, preds)

                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_function, n_trials=100)

        return study.best_params

    def recommend_price(self, quote_features):
        """Generate pricing recommendation with confidence"""

        # Base price prediction
        base_price = self.price_model.predict(quote_features)[0]

        # Win probability at different price points
        price_elasticity = self._compute_price_elasticity(
            quote_features, base_price
        )

        # Expected margin at different price points
        expected_margins = self._compute_expected_margins(
            quote_features, base_price
        )

        # Optimize for expected value (win_prob * margin)
        optimal_price = self._maximize_expected_value(
            base_price, price_elasticity, expected_margins
        )

        # Confidence interval using quantile regression
        confidence_interval = self._compute_confidence_interval(
            quote_features, optimal_price
        )

        return {
            'recommended_price': optimal_price,
            'base_price': base_price,
            'win_probability': self._predict_win_prob(quote_features, optimal_price),
            'expected_margin': self._predict_margin(quote_features, optimal_price),
            'confidence_interval': confidence_interval,
            'price_range': (confidence_interval[0], confidence_interval[1])
        }
```

### Optimization Strategy

**Multi-Objective Optimization**: Balanced three competing objectives

```python
def maximize_expected_value(base_price, features):
    """
    Optimize price to maximize expected value
    EV = Win_Probability(price) × Margin(price)
    """

    def objective(price_multiplier):
        # Adjust price
        candidate_price = base_price * price_multiplier

        # Update features with new price
        features_at_price = update_price_features(features, candidate_price)

        # Predict outcomes
        win_prob = win_prob_model.predict_proba(features_at_price)[:, 1][0]
        margin = margin_model.predict(features_at_price)[0]

        # Expected value
        ev = win_prob * margin

        # Add penalty for extreme prices
        price_deviation_penalty = abs(price_multiplier - 1.0) * 0.1

        return -(ev - price_deviation_penalty)  # Negative for minimization

    # Optimize within reasonable bounds
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        objective,
        bounds=(0.85, 1.25),  # Allow ±15% price adjustment
        method='bounded'
    )

    optimal_multiplier = result.x
    optimal_price = base_price * optimal_multiplier

    return optimal_price
```

### Model Interpretability

**SHAP Values for Explainability**

```python
import shap

class PricingExplainer:
    """Explain pricing recommendations using SHAP"""

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, quote_features):
        """Generate explanation for pricing decision"""

        shap_values = self.explainer.shap_values(quote_features)

        # Top features driving the price
        feature_importance = pd.DataFrame({
            'feature': quote_features.columns,
            'impact': shap_values[0],
            'value': quote_features.iloc[0].values
        }).sort_values('impact', key=abs, ascending=False)

        explanation = {
            'top_price_drivers': feature_importance.head(10).to_dict('records'),
            'base_value': self.explainer.expected_value,
            'predicted_value': self.model.predict(quote_features)[0]
        }

        return explanation
```

### Deployment Architecture

**Real-Time API Service**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI(title="Pricing Optimization API")

class QuoteRequest(BaseModel):
    customer_id: str
    product_specs: dict
    quantity: int
    delivery_date: str
    region: str

class PriceRecommendation(BaseModel):
    recommended_price: float
    confidence_interval: tuple
    win_probability: float
    expected_margin: float
    explanation: dict

@app.post("/recommend-price", response_model=PriceRecommendation)
async def recommend_price(quote: QuoteRequest):
    """
    Generate optimal price recommendation for quote
    """
    try:
        # Feature engineering
        features = feature_engine.engineer_features(quote.dict())

        # Model prediction
        recommendation = pricing_model.recommend_price(features)

        # Explanation
        explanation = explainer.explain_prediction(features)

        # Log for monitoring
        log_prediction(quote, recommendation)

        return PriceRecommendation(
            **recommendation,
            explanation=explanation
        )

    except Exception as e:
        logging.error(f"Pricing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Pricing service error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": MODEL_VERSION}
```

## Results & Impact

### Business Metrics

**Speed**
- ✅ **97% On-Time**: Quotes delivered in <2 hours (from 24h)
- ✅ **87% Faster**: Average turnaround time reduced by 87%

**Profitability**
- ✅ **12% Profit Lift**: Estimated margin improvement
- ✅ **~$2M Annual**: Projected additional profit

**Productivity**
- ✅ **20% Efficiency Gain**: Estimators handle 20% more quotes
- ✅ **Reduced Errors**: 40% fewer pricing errors

**Customer Experience**
- ✅ **Higher Win Rate**: 8% increase in quote acceptance
- ✅ **Client Satisfaction**: NPS score improved by 15 points

### Model Performance

**Predictive Accuracy**
- Price MAPE: 6.2%
- Win Probability AUC: 0.84
- Margin MAPE: 8.1%

**System Reliability**
- 99.7% uptime
- <100ms API response time
- Zero data loss incidents

## Technical Stack

**ML & Data Science**
- XGBoost (gradient boosting)
- Optuna (hyperparameter optimization)
- SHAP (model interpretability)
- Scikit-learn (preprocessing, validation)
- Pandas, NumPy (data manipulation)

**Deployment & Infrastructure**
- FastAPI (API framework)
- Docker (containerization)
- Azure App Service (hosting)
- Azure Database (PostgreSQL)
- Redis (caching)
- GitHub Actions (CI/CD)

**Monitoring & MLOps**
- MLflow (experiment tracking, model registry)
- Azure Application Insights (monitoring)
- Custom dashboards (Plotly, Streamlit)

## Key Learnings

1. **Feature Engineering > Model Complexity**: Well-engineered features had larger impact than model architecture choices

2. **Explainability Critical**: Sales team adoption required transparent explanations, not just accurate predictions

3. **Continuous Learning**: Market dynamics shift; implemented monthly retraining pipeline

4. **Confidence Matters**: Providing confidence intervals allowed sales to make informed adjustments

5. **A/B Testing Validation**: Shadow mode A/B testing for 2 months validated model before full deployment

## Future Enhancements

- **Dynamic Pricing**: Real-time price adjustments based on capacity utilization
- **Competitive Intelligence**: Integrate competitor pricing data
- **Customer Segmentation**: Personalized pricing strategies by customer segment
- **Multi-Product Bundles**: Optimize pricing for bundled offerings
- **Reinforcement Learning**: Explore RL for adaptive pricing strategies

---

**Project Duration**: 6 months (development) + ongoing optimization
**Team**: 2 data scientists, 1 ML engineer, 1 software engineer
**My Role**: Lead Data Scientist - Full project ownership from conception to deployment

[← Back to Projects]({{ site.baseurl }}/projects/)
