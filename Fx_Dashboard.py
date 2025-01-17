import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objs as go
from Fx_SourceCode_Adapt.Fx_data_module import get_fx_pairs_data, FirstTwoMoments, MaxSharpeRatio,  MomentumStrategy, MeanReversionStrategy, VolatilityStrategy, CorrelationStrategy
from Fx_SourceCode_Adapt.Fx_broker import Backtest, StopLoss, Broker, Position
import logging
import random
import string

# Remove yfinance error logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Map the symbols in the dictionary to securities
sp_dict = {
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY",
    "GBPUSD=X": "GBPUSD",
    "AUDUSD=X": "AUDUSD",
    "USDCAD=X": "USDCAD",
    "NZDUSD=X": "NZDUSD",
    "USDCHF=X": "USDCHF",
    "USDMXN=X": "USDMXN",
    "USDSEK=X": "USDSEK",
    "USDKRW=X": "USDKRW",
    "USDTRY=X": "USDTRY",
    "USDZAR=X": "USDZAR",
    "USDBRL=X": "USDBRL",
    "USDCNY=X": "USDCNY",
    "USDINR=X": "USDINR",
    "USDRUB=X": "USDRUB",
    "USDTHB=X": "USDTHB",
    "USDSGD=X": "USDSGD",
    "USDPLN=X": "USDPLN"
}
universe_options = [{"label": security, "value": symbol} for symbol, security in sp_dict.items()]

# Create Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("FX Strategies Backtester", style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Date Selectors (Initial and Final Date)
    html.Div([
        html.Div([
            html.Label("Initial Date:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dcc.DatePickerSingle(
                id='initial-date',
                date='2019-01-01',
                display_format='YYYY-MM-DD',
                style={'width': '250px'}
            )
        ], style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center', 'marginRight': '20px'}),

        html.Div([
            html.Label("Final Date:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dcc.DatePickerSingle(
                id='final-date',
                date='2020-01-01',
                display_format='YYYY-MM-DD',
                style={'width': '250px'}
            )
        ], style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # FX Strategy Dropdown
    html.Div([
        html.Label("FX Strategy:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='information-class',
            options=[
                {'label': 'FirstTwoMoments', 'value': 'first_two_moments'},
                {'label': 'MomentumStrategy', 'value': 'momentum_strategy'},
                {'label': 'MeanReversionStrategy', 'value': 'mean_reversion_strategy'},
                {'label': 'VolatilityStrategy', 'value': 'volatility_strategy'},
                {'label': 'MaxSharpeRatio', 'value': 'max_sharpe_ratio'},
                {'label': 'CorrelationStrategy', 'value': 'correlation_strategy'}
            ],
            value='first_two_moments',
            clearable=False,
            style={'width': '300px', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # FX Pairs Selector
    html.Div([
        html.Label("Select FX Pairs:", style={'fontSize': '18px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='universe-dropdown',
            options=universe_options,
            value=[],
            multi=True,
            placeholder='Pick one or more FX pairs',
            style={'width': '400px', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Run Backtest Button
    html.Button("Run Backtest", id='run-button', n_clicks=0, 
                style={'display': 'block', 'margin': '0 auto', 'marginTop': '20px', 'marginBottom': '20px'}),

    # Portfolio Summary Table and Graphs
    html.Div([
        html.H3("Final Portfolio Summary", style={'fontSize': '20px', 'fontWeight': 'bold'}),
        dash_table.DataTable(id='portfolio-summary-table', style_table={'overflowX': 'auto'})
    ], style={'marginTop': '40px', 'textAlign': 'center'}),

    dcc.Graph(id='correlation-graph', style={'marginTop': '40px'}),

    dcc.Graph(id='portfolio-value-graph', style={'marginTop': '40px'}),

    # Statistics Table (Daily Returns) placed at the bottom
    html.Div([
        html.H3("Statistics (Daily Returns)", style={'fontSize': '20px', 'fontWeight': 'bold'}),
        dash_table.DataTable(
            id='stats-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ], style={'marginTop': '40px', 'textAlign': 'center'})
], style={'backgroundColor': '#f4f4f4', 'padding': '20px'})

@app.callback(
    [
        Output('portfolio-summary-table', 'data'),
        Output('portfolio-summary-table', 'columns'),
        Output('correlation-graph', 'figure'),
        Output('portfolio-value-graph', 'figure'),
        Output('stats-table', 'data'),
        Output('stats-table', 'columns')
    ],
    Input('run-button', 'n_clicks'),
    State('initial-date', 'date'),
    State('final-date', 'date'),
    State('information-class', 'value'),
    State('universe-dropdown', 'value')
)
def run_backtest(n_clicks, init_date_str, final_date_str, information_class_str, selected_symbols):
    if n_clicks == 0:
        return [], [], go.Figure(), go.Figure(), [], []

    init_date = datetime.strptime(init_date_str, "%Y-%m-%d")
    final_date = datetime.strptime(final_date_str, "%Y-%m-%d")

    # Map information class to your strategy
    information_class = {
        'first_two_moments': FirstTwoMoments,
        'momentum_strategy': MomentumStrategy,
        'mean_reversion_strategy': MeanReversionStrategy,
        'volatility_strategy': VolatilityStrategy,
        'max_sharpe_ratio': MaxSharpeRatio,
        'correlation_strategy': CorrelationStrategy
    }.get(information_class_str, FirstTwoMoments)

    # -------------------------
    # 1) BACKTEST AND BROKER
    # -------------------------
    backtest = Backtest(
        initial_date=init_date,
        final_date=final_date,
        information_class=information_class,
        risk_model=StopLoss,
        name_blockchain='backtest_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5)),
        verbose=False
    )
    backtest.universe = selected_symbols

    backtest.run_backtest()
    broker = Broker(cash=1_000_000, verbose=False)

    # Transaction log from the backtest
    transaction_log = backtest.broker.get_transaction_log()

    # We'll track daily portfolio value
    portfolio_values = []
    last_prices = {}
    rebalance_dates = []

    current_date = init_date
    while current_date <= final_date:
        # Skip weekends explicitly
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Filter transactions for current day
        daily_transactions = transaction_log[transaction_log['Date'] == current_date]
        for _, transaction in daily_transactions.iterrows():
            ticker = transaction['Ticker']
            action = transaction['Action']
            quantity = transaction['Quantity']
            price = transaction['Price']

            if action == 'BUY':
                broker.buy(ticker, quantity, price, current_date)
            elif action == 'SELL':
                broker.sell(ticker, quantity, price, current_date)

        # Check for rebalancing
        if backtest.rebalance_flag().time_to_rebalance(current_date):
            rebalance_dates.append(current_date)

        # Get current market prices using the adapted get_fx_pairs_data function
        market_prices = {}
        for ticker in broker.positions.keys():
            try:
                data = get_fx_pairs_data(
                    [ticker], 
                    current_date.strftime('%Y-%m-%d'), 
                    (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if not data.empty:
                    market_prices[ticker] = data.iloc[-1]['Close']
                    last_prices[ticker] = market_prices[ticker]
                else:
                    market_prices[ticker] = last_prices.get(ticker, 0)
            except Exception:
                market_prices[ticker] = last_prices.get(ticker, 0)

        # Calculate portfolio value for the current day
        try:
            portfolio_value = broker.get_portfolio_value(market_prices)
        except Exception:
            portfolio_value = portfolio_values[-1][1] if portfolio_values else 1_000_000

        portfolio_values.append((current_date, portfolio_value))

        current_date += timedelta(days=1)

    # -------------------------
    # 2) PREPARE SUMMARY TABLE
    # -------------------------
    final_market_prices = {
        ticker: last_prices.get(ticker, 0) for ticker in broker.positions.keys()
    }
    summary_data = []
    for ticker, position in broker.positions.items():
        summary_data.append({
            "Ticker": ticker,
            "Shares": position.quantity,
            "Last Price": final_market_prices[ticker],
            "Value": position.quantity * final_market_prices[ticker]
        })

    total_value = sum(item["Value"] for item in summary_data) + broker.cash
    summary_data.append({
        "Ticker": "Total",
        "Shares": "-",
        "Last Price": "-",
        "Value": total_value
    })

    summary_columns = [
        {"name": "Ticker", "id": "Ticker"},
        {"name": "Shares", "id": "Shares"},
        {"name": "Last Price", "id": "Last Price"},
        {"name": "Value", "id": "Value"}
    ]

    # -------------------------
    # 3) CORRELATION GRAPH
    # -------------------------
    # Build a DataFrame of daily closes for each selected symbol so we can compute daily returns and correlation.
    price_df = pd.DataFrame()
    for symbol in selected_symbols:
        data = get_fx_pairs_data([symbol], init_date_str, final_date_str)
        if not data.empty:
            # Align on date index to handle merges properly
            data = data[['Date','Close']].set_index('Date')
            data.index = pd.to_datetime(data.index)
            price_df[symbol] = data['Close']

    # Drop any days that are all NaN
    price_df.dropna(how='all', inplace=True)

    # Compute daily returns and then plot correlation heatmap
    ret_df = price_df.pct_change().dropna()
    if not ret_df.empty:
        corr_matrix = ret_df.corr()
        corr_fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlGn',
                zmin=-1,
                zmax=1
            )
        )
        corr_fig.update_layout(title="Correlation Heatmap of Daily Returns")
    else:
        corr_fig = go.Figure()

    # -------------------------
    # 4) PORTFOLIO VALUE GRAPH
    # -------------------------
    df_portfolio = pd.DataFrame(portfolio_values, columns=['Date', 'Portfolio Value'])
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(
        go.Scatter(
            x=df_portfolio['Date'], 
            y=df_portfolio['Portfolio Value'], 
            mode='lines', 
            name='Portfolio Value'
        )
    )
    fig_portfolio.add_trace(
        go.Scatter(
            x=df_portfolio['Date'],
            y=[1_000_000] * len(df_portfolio),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='1M Threshold'
        )
    )

    # Mark rebalancing points
    for reb_date in rebalance_dates:
        marker_day = reb_date - timedelta(days=1)
        val = df_portfolio.loc[df_portfolio['Date'] == marker_day, 'Portfolio Value']
        if not val.empty:
            fig_portfolio.add_trace(
                go.Scatter(
                    x=[marker_day],
                    y=[val.values[0]],
                    mode='markers',
                    marker=dict(color='blue', size=10, symbol='circle'),
                    showlegend=False
                )
            )
    fig_portfolio.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value"
    )

    # -------------------------
    # 5) STATISTICS TABLE
    # -------------------------
    stats_data = []
    if not ret_df.empty:
        for symbol in ret_df.columns:
            avg_return = ret_df[symbol].mean() * 100
            std_return = ret_df[symbol].std() * 100
            stats_data.append({
                'Ticker': symbol,
                'Average Return (%)': round(avg_return, 2),
                'Std Dev (%)': round(std_return, 2)
            })
    
    # Entire portfolio stats
    if len(df_portfolio) > 1:
        df_portfolio['Daily Return'] = df_portfolio['Portfolio Value'].pct_change()
        daily_returns = df_portfolio['Daily Return'].dropna()
        
        if not daily_returns.empty:
            avg_port = daily_returns.mean() * 100
            std_port = daily_returns.std() * 100
            stats_data.append({
                'Ticker': 'Entire Portfolio',
                'Average Return (%)': round(avg_port, 2),
                'Std Dev (%)': round(std_port, 2)
            })

    stats_columns = [
        {'name': 'Ticker', 'id': 'Ticker'},
        {'name': 'Average Return (%)', 'id': 'Average Return (%)'},
        {'name': 'Std Dev (%)', 'id': 'Std Dev (%)'}
    ]

    return (
        summary_data,             
        summary_columns,         
        corr_fig,                 
        fig_portfolio,            
        stats_data,               
        stats_columns            
    )

if __name__ == '__main__':
    app.run_server(debug=True)