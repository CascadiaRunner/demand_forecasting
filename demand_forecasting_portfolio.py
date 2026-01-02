"""
Supply Chain Demand Forecasting Portfolio Project
Author: Christopher Garland
Purpose: Demonstrate time-series forecasting capabilities for supply chain applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("SUPPLY CHAIN DEMAND FORECASTING - PORTFOLIO PROJECT")
print("Christopher Garland - Data Scientist")
print("=" * 70)
print("\n")

# Generate realistic retail sales data with seasonality and trend
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
n_days = len(date_range)

# Create base demand with trend
base_demand = 1000 + np.linspace(0, 500, n_days)

# Add weekly seasonality (higher on weekends)
day_of_week = pd.Series(date_range.dayofweek)
weekly_pattern = 50 * np.sin(2 * np.pi * day_of_week / 7)

# Add yearly seasonality (holiday peaks)
day_of_year = pd.Series(date_range.dayofyear)
yearly_pattern = 200 * np.sin(2 * np.pi * day_of_year / 365.25)

# Add monthly seasonality
monthly_pattern = 100 * np.sin(2 * np.pi * day_of_year / 30)

# Add random noise
noise = np.random.normal(0, 50, n_days)

# Combine all components
demand = base_demand + weekly_pattern + yearly_pattern + monthly_pattern + noise
demand = np.maximum(demand, 100)  # Floor at 100 units

# Create dataframe
df = pd.DataFrame({
    'date': date_range,
    'demand': demand
})

print("Dataset Overview:")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total Days: {len(df)}")
print(f"Average Daily Demand: {df['demand'].mean():.0f} units")
print(f"Min Demand: {df['demand'].min():.0f} units")
print(f"Max Demand: {df['demand'].max():.0f} units")
print("\n")

# Add time-based features
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Add rolling averages (lagged features to avoid lookahead bias)
df['rolling_7d_avg'] = df['demand'].rolling(window=7, min_periods=1).mean().shift(1)
df['rolling_30d_avg'] = df['demand'].rolling(window=30, min_periods=1).mean().shift(1)
df['rolling_7d_std'] = df['demand'].rolling(window=7, min_periods=1).std().shift(1)

# Fill NaN values
df = df.fillna(method='bfill')

# Split into train/test (80/20 split)
train_size = int(len(df) * 0.8)
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

print("Train/Test Split:")
print(f"Training Period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"Testing Period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
print(f"Training Days: {len(train_df)}")
print(f"Testing Days: {len(test_df)}")
print("\n")

# Model 1: Simple Moving Average Baseline
print("=" * 70)
print("MODEL 1: 30-Day Moving Average (Baseline)")
print("=" * 70)
test_df['ma_forecast'] = test_df['rolling_30d_avg']
ma_mae = mean_absolute_error(test_df['demand'], test_df['ma_forecast'])
ma_rmse = np.sqrt(mean_squared_error(test_df['demand'], test_df['ma_forecast']))
ma_mape = np.mean(np.abs((test_df['demand'] - test_df['ma_forecast']) / test_df['demand'])) * 100

print(f"Mean Absolute Error (MAE): {ma_mae:.2f} units")
print(f"Root Mean Squared Error (RMSE): {ma_rmse:.2f} units")
print(f"Mean Absolute Percentage Error (MAPE): {ma_mape:.2f}%")
print("\n")

# Model 2: Linear Regression with Seasonal Features
print("=" * 70)
print("MODEL 2: Multi-Feature Linear Regression")
print("=" * 70)

from sklearn.linear_model import LinearRegression

features = ['day_of_week', 'month', 'quarter', 'is_weekend', 
            'rolling_7d_avg', 'rolling_30d_avg', 'rolling_7d_std']

X_train = train_df[features]
y_train = train_df['demand']
X_test = test_df[features]
y_test = test_df['demand']

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

test_df['lr_forecast'] = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, test_df['lr_forecast'])
lr_rmse = np.sqrt(mean_squared_error(y_test, test_df['lr_forecast']))
lr_mape = np.mean(np.abs((y_test - test_df['lr_forecast']) / y_test)) * 100
lr_r2 = r2_score(y_test, test_df['lr_forecast'])

print(f"Mean Absolute Error (MAE): {lr_mae:.2f} units")
print(f"Root Mean Squared Error (RMSE): {lr_rmse:.2f} units")
print(f"Mean Absolute Percentage Error (MAPE): {lr_mape:.2f}%")
print(f"R² Score: {lr_r2:.4f}")
print("\n")

print("Feature Importance (Coefficients):")
for feature, coef in zip(features, lr_model.coef_):
    print(f"  {feature}: {coef:.2f}")
print("\n")

# Model 3: Random Forest (more sophisticated)
print("=" * 70)
print("MODEL 3: Random Forest Regressor")
print("=" * 70)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

test_df['rf_forecast'] = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, test_df['rf_forecast'])
rf_rmse = np.sqrt(mean_squared_error(y_test, test_df['rf_forecast']))
rf_mape = np.mean(np.abs((y_test - test_df['rf_forecast']) / y_test)) * 100
rf_r2 = r2_score(y_test, test_df['rf_forecast'])

print(f"Mean Absolute Error (MAE): {rf_mae:.2f} units")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f} units")
print(f"Mean Absolute Percentage Error (MAPE): {rf_mape:.2f}%")
print(f"R² Score: {rf_r2:.4f}")
print("\n")

print("Feature Importance:")
for feature, importance in sorted(zip(features, rf_model.feature_importances_), 
                                   key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.4f}")
print("\n")

# Model Comparison
print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
comparison_df = pd.DataFrame({
    'Model': ['Moving Average', 'Linear Regression', 'Random Forest'],
    'MAE': [ma_mae, lr_mae, rf_mae],
    'RMSE': [ma_rmse, lr_rmse, rf_rmse],
    'MAPE (%)': [ma_mape, lr_mape, rf_mape],
    'R²': [np.nan, lr_r2, rf_r2]
})
print(comparison_df.to_string(index=False))
print("\n")

improvement_vs_baseline = ((ma_mae - rf_mae) / ma_mae) * 100
print(f"Random Forest improved forecast accuracy by {improvement_vs_baseline:.1f}% vs. baseline")
print("\n")

# Business Impact Calculation
print("=" * 70)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 70)

avg_unit_cost = 50  # Assume $50 per unit
holding_cost_pct = 0.20  # 20% annual holding cost
stockout_cost_multiplier = 3  # Stockouts cost 3x the unit cost

# Calculate costs for baseline vs. optimized model
baseline_overstock = np.maximum(test_df['ma_forecast'] - test_df['demand'], 0).sum()
baseline_understock = np.maximum(test_df['demand'] - test_df['ma_forecast'], 0).sum()

rf_overstock = np.maximum(test_df['rf_forecast'] - test_df['demand'], 0).sum()
rf_understock = np.maximum(test_df['demand'] - test_df['rf_forecast'], 0).sum()

baseline_cost = (baseline_overstock * avg_unit_cost * holding_cost_pct + 
                 baseline_understock * avg_unit_cost * stockout_cost_multiplier)
rf_cost = (rf_overstock * avg_unit_cost * holding_cost_pct + 
           rf_understock * avg_unit_cost * stockout_cost_multiplier)

annual_savings = baseline_cost - rf_cost

print(f"Assumptions:")
print(f"  - Average unit cost: ${avg_unit_cost}")
print(f"  - Annual holding cost: {holding_cost_pct*100:.0f}% of unit cost")
print(f"  - Stockout penalty: {stockout_cost_multiplier}x unit cost")
print("\n")

print(f"Baseline Model (Moving Average):")
print(f"  - Overstock units: {baseline_overstock:,.0f}")
print(f"  - Understock units: {baseline_understock:,.0f}")
print(f"  - Total estimated cost: ${baseline_cost:,.0f}")
print("\n")

print(f"Optimized Model (Random Forest):")
print(f"  - Overstock units: {rf_overstock:,.0f}")
print(f"  - Understock units: {rf_understock:,.0f}")
print(f"  - Total estimated cost: ${rf_cost:,.0f}")
print("\n")

print(f"ANNUAL COST SAVINGS: ${annual_savings:,.0f}")
print(f"ROI on forecasting improvement: {(annual_savings/baseline_cost)*100:.1f}%")
print("\n")

# Create visualizations
print("Generating visualizations...")

# Figure 1: Actual vs Forecasted Demand
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Last 90 days detailed view
last_90_days = test_df.tail(90)
axes[0].plot(last_90_days['date'], last_90_days['demand'], 
             label='Actual Demand', linewidth=2, color='black', alpha=0.7)
axes[0].plot(last_90_days['date'], last_90_days['ma_forecast'], 
             label='Moving Avg Forecast', linewidth=1.5, linestyle='--', alpha=0.7)
axes[0].plot(last_90_days['date'], last_90_days['lr_forecast'], 
             label='Linear Reg Forecast', linewidth=1.5, linestyle='--', alpha=0.7)
axes[0].plot(last_90_days['date'], last_90_days['rf_forecast'], 
             label='Random Forest Forecast', linewidth=2, alpha=0.8)
axes[0].set_title('Demand Forecasting Performance - Last 90 Days (Detailed View)', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Units', fontsize=12)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Full test period
axes[1].plot(test_df['date'], test_df['demand'], 
             label='Actual Demand', linewidth=2, color='black', alpha=0.7)
axes[1].plot(test_df['date'], test_df['rf_forecast'], 
             label='Random Forest Forecast', linewidth=2, alpha=0.8)
axes[1].fill_between(test_df['date'], 
                      test_df['rf_forecast'] - rf_rmse, 
                      test_df['rf_forecast'] + rf_rmse,
                      alpha=0.2, label='±1 RMSE Confidence Band')
axes[1].set_title('Full Test Period Forecast Performance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Units', fontsize=12)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/forecast_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: forecast_performance.png")

# Figure 2: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MAE Comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['MAE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('MAE (units)', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['MAE']):
    axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# MAPE Comparison
axes[0, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 1].set_title('Mean Absolute Percentage Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAPE (%)', fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=15)
for i, v in enumerate(comparison_df['MAPE (%)']):
    axes[0, 1].text(i, v + 0.05, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')

# R² Comparison
r2_data = comparison_df[comparison_df['R²'].notna()]
axes[1, 0].bar(r2_data['Model'], r2_data['R²'], color=['#4ECDC4', '#45B7D1'])
axes[1, 0].set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('R² Score', fontsize=10)
axes[1, 0].set_ylim([0, 1])
axes[1, 0].tick_params(axis='x', rotation=15)
for i, (model, r2) in enumerate(zip(r2_data['Model'], r2_data['R²'])):
    axes[1, 0].text(i, r2 + 0.02, f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')

# Cost Savings
cost_data = pd.DataFrame({
    'Model': ['Baseline\n(Moving Avg)', 'Optimized\n(Random Forest)'],
    'Cost': [baseline_cost, rf_cost]
})
bars = axes[1, 1].bar(cost_data['Model'], cost_data['Cost'], color=['#FF6B6B', '#45B7D1'])
axes[1, 1].set_title(f'Estimated Annual Cost (Savings: ${annual_savings:,.0f})', 
                      fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Cost ($)', fontsize=10)
for i, v in enumerate(cost_data['Cost']):
    axes[1, 1].text(i, v + 1000, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('./outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")

# Figure 3: Forecast Error Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of errors
rf_errors = test_df['demand'] - test_df['rf_forecast']
axes[0].hist(rf_errors, bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0].axvline(rf_errors.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Mean Error: {rf_errors.mean():.1f}')
axes[0].set_title('Forecast Error Distribution (Random Forest)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Error (Actual - Forecast)', fontsize=10)
axes[0].set_ylabel('Frequency', fontsize=10)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter plot: Actual vs Predicted
axes[1].scatter(test_df['demand'], test_df['rf_forecast'], alpha=0.5, s=20, color='#45B7D1')
axes[1].plot([test_df['demand'].min(), test_df['demand'].max()], 
             [test_df['demand'].min(), test_df['demand'].max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title('Actual vs Predicted Demand', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual Demand (units)', fontsize=10)
axes[1].set_ylabel('Predicted Demand (units)', fontsize=10)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_analysis.png")

print("\n" + "=" * 70)
print("PORTFOLIO PROJECT COMPLETE")
print("=" * 70)
print("\nKey Takeaways:")
print("  ✓ Developed multiple forecasting models with increasing sophistication")
print("  ✓ Random Forest achieved best performance with MAPE of {:.2f}%".format(rf_mape))
print("  ✓ Improved forecast accuracy by {:.1f}% vs baseline".format(improvement_vs_baseline))
print("  ✓ Estimated annual cost savings: ${:,.0f}".format(annual_savings))
print("\nThis demonstrates:")
print("  • Time-series forecasting expertise")
print("  • Feature engineering for demand prediction")
print("  • Model comparison and selection")
print("  • Business impact quantification")
print("  • Clear visualization and communication")
