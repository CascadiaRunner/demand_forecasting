# Supply Chain Demand Forecasting Portfolio Project

**Author:** Christopher Garland  
**Contact:** Christopher.Garland@gmail.com  
**LinkedIn:** [linkedin.com/in/christopher-garland-08216222](https://linkedin.com/in/christopher-garland-08216222)

---

## Overview

This portfolio project demonstrates practical demand forecasting capabilities for supply chain applications. Using realistic retail sales data with seasonal patterns, I built and compared three forecasting models to show how machine learning can reduce inventory costs and improve operational efficiency.

## Business Context

Accurate demand forecasting is critical for supply chain operations:
- **Overstocking** ties up capital and incurs holding costs
- **Understocking** leads to lost sales and customer dissatisfaction
- Poor forecasts create inefficiencies throughout the supply chain

This project quantifies the business impact of improved forecasting accuracy.

## Dataset

**Synthetic retail sales data** spanning 3 years (2022-2024):
- 1,096 daily observations
- Realistic patterns including:
  - Long-term growth trend
  - Weekly seasonality (higher weekend demand)
  - Annual seasonality (holiday peaks)
  - Monthly cycles
  - Random variation

The data mirrors real-world demand patterns without exposing proprietary business information.

## Methodology

### Models Developed

1. **Baseline: 30-Day Moving Average**
   - Simple, interpretable benchmark
   - Uses historical average to predict future demand
   
2. **Linear Regression with Seasonal Features**
   - Incorporates time-based features (day of week, month, quarter)
   - Uses rolling averages and standard deviations
   - Captures linear relationships

3. **Random Forest Regressor**
   - Non-linear ensemble model
   - Automatically captures complex interactions
   - More sophisticated feature importance analysis

### Feature Engineering

Key features included:
- Day of week, month, quarter
- Weekend indicator
- 7-day and 30-day rolling averages
- 7-day rolling standard deviation (volatility measure)

## Results

### Model Performance

| Model | MAE (units) | MAPE | R² Score |
|-------|-------------|------|----------|
| Moving Average (Baseline) | 77.7 | 5.76% | - |
| Linear Regression | 60.4 | 4.47% | 0.643 |
| Random Forest | 62.7 | 4.62% | 0.636 |

**Key Finding:** Both ML models improved forecast accuracy by ~20% vs. the baseline.

### Business Impact

Assuming realistic supply chain costs:
- Average unit cost: $50
- Annual holding cost: 20% of unit cost
- Stockout penalty: 3x unit cost

**Estimated Annual Cost Savings: $91,874**

This represents a 7% reduction in inventory-related costs by switching from the baseline to the optimized Random Forest model.

### Breakdown:
- **Baseline Model:** $1,311,982 in holding + stockout costs
- **Optimized Model:** $1,220,109 in holding + stockout costs
- **Savings:** $91,874 annually

## Visualizations

The project includes three visualization sets:

1. **Forecast Performance Charts**
   - Detailed 90-day view comparing all models
   - Full test period with confidence bands
   
2. **Model Comparison Dashboard**
   - MAE, MAPE, R² comparisons
   - Cost savings visualization
   
3. **Error Analysis**
   - Forecast error distribution
   - Actual vs. predicted scatter plot

## Key Takeaways

✓ **Time-series forecasting expertise** - Built and evaluated multiple forecasting approaches  
✓ **Feature engineering** - Created meaningful predictors from temporal data  
✓ **Model selection** - Systematically compared models using appropriate metrics  
✓ **Business impact quantification** - Translated accuracy improvements into dollar savings  
✓ **Clear communication** - Presented technical results in business-friendly terms

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models
- **matplotlib & seaborn** - Visualization

## Files Included

- `demand_forecasting_portfolio.py` - Complete analysis script
- `forecast_performance.png` - Model comparison visualizations
- `model_comparison.png` - Metrics and cost analysis
- `error_analysis.png` - Forecast error distribution
- `README.md` - This document

## How to Run

```bash
python3 demand_forecasting_portfolio.py
```

This will:
1. Generate synthetic demand data
2. Train and evaluate all three models
3. Calculate business impact metrics
4. Create visualization outputs

## Applicability

This forecasting approach is applicable to:
- E-commerce demand planning
- Retail inventory management
- Manufacturing production scheduling
- Distribution center operations
- Any time-series forecasting challenge with seasonal patterns

## Contact

Interested in discussing how these techniques could apply to your supply chain challenges?

**Christopher Garland**  
Email: Christopher.Garland@gmail.com  
LinkedIn: [linkedin.com/in/christopher-garland-08216222](https://linkedin.com/in/christopher-garland-08216222)
