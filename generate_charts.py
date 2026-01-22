"""
Generate static chart images for README.md documentation.
"""

import os
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import PortfolioOptimizer

# Create images directory
os.makedirs('images', exist_ok=True)

# Configuration
TICKERS = ['SPY', 'QQQ', 'GLD', 'VNQ', 'BTC-USD']
START_DATE = '2020-06-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.05
MAX_WEIGHTS = {'BTC-USD': 0.25}

print("Initializing Portfolio Optimizer...")
optimizer = PortfolioOptimizer(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    risk_free_rate=RISK_FREE_RATE,
    max_weights=MAX_WEIGHTS
)

# Fetch and process data
print("Fetching data...")
prices = optimizer.fetch_data()
returns = optimizer.calculate_returns()
mean_returns, cov_matrix = optimizer.get_metrics()

# ============================================================================
# Chart 1: Normalized Price Series
# ============================================================================
print("Generating Chart 1: Normalized Price Series...")
normalized_prices = prices / prices.iloc[0] * 100

fig1 = px.line(
    normalized_prices,
    title='<b>Asset Performance: Normalized Price Series</b><br><sup>Initial Investment = $100 (June 2020 - December 2024)</sup>',
    labels={'value': 'Normalized Price ($)', 'Date': 'Date', 'variable': 'Asset'}
)
fig1.update_layout(
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x unified'
)
fig1.write_image('images/price_performance.png', width=1200, height=600, scale=2)
print("  ✓ Saved: images/price_performance.png")

# ============================================================================
# Chart 2: Correlation Heatmap
# ============================================================================
print("Generating Chart 2: Correlation Heatmap...")
corr_matrix = returns.corr()

fig2 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu_r',
    zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2),
    texttemplate='%{text}',
    textfont=dict(size=14, color='white'),
    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
))
fig2.update_layout(
    title='<b>Asset Correlation Matrix</b><br><sup>Cross-asset correlations for diversification analysis</sup>',
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    xaxis=dict(side='bottom'),
    yaxis=dict(autorange='reversed')
)
fig2.write_image('images/correlation_heatmap.png', width=800, height=700, scale=2)
print("  ✓ Saved: images/correlation_heatmap.png")

# ============================================================================
# Chart 3: Monte Carlo Simulation
# ============================================================================
print("Generating Chart 3: Monte Carlo Simulation (10,000 portfolios)...")
mc_results = optimizer.monte_carlo_simulation(n_portfolios=10000, seed=42)

fig3 = px.scatter(
    mc_results,
    x='Volatility',
    y='Return',
    color='Sharpe',
    color_continuous_scale='Viridis',
    title='<b>Monte Carlo Portfolio Simulation</b><br><sup>10,000 random portfolio allocations with Max Sharpe portfolio highlighted</sup>',
    labels={'Return': 'Expected Return (%)', 'Volatility': 'Risk (Volatility %)', 'Sharpe': 'Sharpe Ratio'}
)

# Add optimal portfolio marker
msr = optimizer.optimize_sharpe()
fig3.add_trace(go.Scatter(
    x=[msr['volatility'] * 100],
    y=[msr['return'] * 100],
    mode='markers',
    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
    name='Max Sharpe Portfolio',
    hovertemplate=f"<b>Max Sharpe Portfolio</b><br>Return: {msr['return']*100:.2f}%<br>Volatility: {msr['volatility']*100:.2f}%<br>Sharpe: {msr['sharpe']:.4f}<extra></extra>"
))

fig3.update_layout(
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    showlegend=True
)
fig3.write_image('images/monte_carlo.png', width=1200, height=700, scale=2)
print("  ✓ Saved: images/monte_carlo.png")

# ============================================================================
# Chart 4: Efficient Frontier
# ============================================================================
print("Generating Chart 4: Efficient Frontier...")
ef = optimizer.get_efficient_frontier(n_points=100)

fig4 = go.Figure()

# Monte Carlo cloud
fig4.add_trace(go.Scatter(
    x=mc_results['Volatility'],
    y=mc_results['Return'],
    mode='markers',
    marker=dict(size=3, color=mc_results['Sharpe'], colorscale='Viridis', opacity=0.5),
    name='Random Portfolios'
))

# Efficient Frontier
fig4.add_trace(go.Scatter(
    x=ef['Volatility'] * 100,
    y=ef['Return'] * 100,
    mode='lines',
    line=dict(color='#FF6B6B', width=4),
    name='Efficient Frontier'
))

# Max Sharpe
fig4.add_trace(go.Scatter(
    x=[msr['volatility'] * 100],
    y=[msr['return'] * 100],
    mode='markers',
    marker=dict(size=18, color='#FFD93D', symbol='star', line=dict(width=2, color='white')),
    name=f"Max Sharpe (SR: {msr['sharpe']:.3f})"
))

# Min Volatility
min_vol = optimizer.optimize_min_volatility()
fig4.add_trace(go.Scatter(
    x=[min_vol['volatility'] * 100],
    y=[min_vol['return'] * 100],
    mode='markers',
    marker=dict(size=16, color='#6BCB77', symbol='diamond', line=dict(width=2, color='white')),
    name=f"Min Volatility (SR: {min_vol['sharpe']:.3f})"
))

fig4.update_layout(
    title='<b>Efficient Frontier Analysis</b><br><sup>Optimal risk-return trade-off with key portfolios marked</sup>',
    xaxis_title='Annualized Volatility (%)',
    yaxis_title='Annualized Return (%)',
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig4.write_image('images/efficient_frontier.png', width=1200, height=700, scale=2)
print("  ✓ Saved: images/efficient_frontier.png")

# ============================================================================
# Chart 5: Portfolio Weight Comparison
# ============================================================================
print("Generating Chart 5: Portfolio Weight Comparison...")

# Equal weight
ew_weights = {t: 1/len(TICKERS) for t in TICKERS}

# Optimized weights
opt_weights = msr['weights']

weights_df = pd.DataFrame({
    'Asset': list(ew_weights.keys()),
    'Equal Weight': [v * 100 for v in ew_weights.values()],
    'Max Sharpe': [opt_weights[t] * 100 for t in ew_weights.keys()]
})

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    name='Equal Weight',
    x=weights_df['Asset'],
    y=weights_df['Equal Weight'],
    marker_color='#636EFA'
))
fig5.add_trace(go.Bar(
    name='Max Sharpe Optimized',
    x=weights_df['Asset'],
    y=weights_df['Max Sharpe'],
    marker_color='#EF553B'
))

fig5.update_layout(
    title='<b>Portfolio Allocation Comparison</b><br><sup>Equal Weight vs. Optimized Max Sharpe Portfolio</sup>',
    xaxis_title='Asset',
    yaxis_title='Weight (%)',
    barmode='group',
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig5.write_image('images/weight_comparison.png', width=1000, height=600, scale=2)
print("  ✓ Saved: images/weight_comparison.png")

# ============================================================================
# Chart 6: Cumulative Returns Comparison
# ============================================================================
print("Generating Chart 6: Cumulative Returns...")

# Calculate portfolio returns
ew_weights_arr = np.array([ew_weights[t] for t in returns.columns])
opt_weights_arr = np.array([opt_weights[t] for t in returns.columns])

ew_returns = (returns * ew_weights_arr).sum(axis=1)
opt_returns = (returns * opt_weights_arr).sum(axis=1)

# Cumulative returns
ew_cum = (1 + ew_returns).cumprod() * 100
opt_cum = (1 + opt_returns).cumprod() * 100

cum_df = pd.DataFrame({
    'Date': ew_cum.index,
    'Equal Weight': ew_cum.values,
    'Max Sharpe Optimized': opt_cum.values
}).melt(id_vars='Date', var_name='Strategy', value_name='Value')

fig6 = px.line(
    cum_df,
    x='Date',
    y='Value',
    color='Strategy',
    title='<b>Cumulative Portfolio Performance</b><br><sup>$100 Initial Investment (June 2020 - December 2024)</sup>',
    labels={'Value': 'Portfolio Value ($)', 'Date': 'Date'}
)
fig6.update_layout(
    template='plotly_dark',
    font=dict(family='Arial', size=14),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
fig6.write_image('images/cumulative_returns.png', width=1200, height=600, scale=2)
print("  ✓ Saved: images/cumulative_returns.png")

# ============================================================================
# Chart 7: KPI Dashboard
# ============================================================================
print("Generating Chart 7: KPI Dashboard...")

# Calculate metrics
ew_vol = np.sqrt(np.dot(ew_weights_arr.T, np.dot(cov_matrix.values, ew_weights_arr)))
ew_ret = np.dot(ew_weights_arr, mean_returns.values)
ew_sharpe = (ew_ret - RISK_FREE_RATE) / ew_vol

metrics = {
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
    'Equal Weight': [f'{ew_ret*100:.2f}%', f'{ew_vol*100:.2f}%', f'{ew_sharpe:.3f}'],
    'Max Sharpe': [f"{msr['return']*100:.2f}%", f"{msr['volatility']*100:.2f}%", f"{msr['sharpe']:.4f}"]
}
metrics_df = pd.DataFrame(metrics)

fig7 = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Metric</b>', '<b>Equal Weight</b>', '<b>Max Sharpe</b>'],
        fill_color='#2E4057',
        align='center',
        font=dict(color='white', size=16)
    ),
    cells=dict(
        values=[metrics_df['Metric'], metrics_df['Equal Weight'], metrics_df['Max Sharpe']],
        fill_color=[['#1A1A2E', '#16213E'] * 2],
        align='center',
        font=dict(color='white', size=14),
        height=40
    )
)])
fig7.update_layout(
    title='<b>Key Performance Indicators (KPIs)</b>',
    template='plotly_dark',
    font=dict(family='Arial', size=14)
)
fig7.write_image('images/kpi_dashboard.png', width=800, height=300, scale=2)
print("  ✓ Saved: images/kpi_dashboard.png")

print("\n" + "="*60)
print("All charts generated successfully!")
print("="*60)
print("\nImages saved to ./images/ directory:")
for f in os.listdir('images'):
    if f.endswith('.png'):
        size = os.path.getsize(f'images/{f}') / 1024
        print(f"  • {f} ({size:.1f} KB)")
