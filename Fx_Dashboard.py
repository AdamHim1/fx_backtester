import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objs as go
from Fx_SourceCode_Adapt.Fx_data_module import get_fx_pairs_data, FirstTwoMoments, MaxSharpeRatio, MomentumStrategy, MeanReversionStrategy, VolatilityStrategy
from Fx_SourceCode_Adapt.Fx_broker import Backtest, StopLoss, Broker, Position
import logging
import random
import string
import numpy as np

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
    html.H1("FX Strategies Backtester", style={'textAlign': 'center', 'marginBottom': '20px', 'color': 'white'}),

    # Date Selectors (Initial and Final Date)
    html.Div([
        html.Div([
            html.Label("Initial Date:", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': 'white'}),
            dcc.DatePickerSingle(
                id='initial-date',
                date='2019-01-01',
                display_format='YYYY-MM-DD',
                style={'width': '250px'}
            )
        ], style={'display': 'inline-block', 'width': '48%', 'textAlign': 'center', 'marginRight': '20px'}),

        html.Div([
            html.Label("Final Date:", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': 'white'}),
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
        html.Label("FX Strategy:", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': 'white'}),
        dcc.Dropdown(
            id='information-class',
            options=[
                {'label': 'FirstTwoMoments', 'value': 'first_two_moments'},
                {'label': 'MomentumStrategy', 'value': 'momentum_strategy'},
                {'label': 'MeanReversionStrategy', 'value': 'mean_reversion_strategy'},
                {'label': 'VolatilityStrategy', 'value': 'volatility_strategy'},
                {'label': 'MaxSharpeRatio', 'value': 'max_sharpe_ratio'}
            ],
            value='first_two_moments',
            clearable=False,
            style={'width': '300px', 'margin': '0 auto', 'backgroundColor': '#444', 'color': 'black'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # FX Pairs Selector
    html.Div([
        html.Label("Select FX Pairs:", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': 'white'}),
        dcc.Dropdown(
            id='universe-dropdown',
            options=universe_options,
            value=[],
            multi=True,
            placeholder='Pick one or more FX pairs',
            style={'width': '400px', 'margin': '0 auto', 'backgroundColor': '#444', 'color': 'black'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Run Backtest Button
    html.Button("Run Backtest", id='run-button', n_clicks=0, 
                style={'display': 'block', 'margin': '0 auto', 'marginTop': '20px', 'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'fontSize': '16px'}),

    # Currency Performance Statistics Table and Graphs
    html.Div([
        html.H3("Final Portfolio's Currencies Statistics", style={'fontSize': '20px', 'fontWeight': 'bold', 'color': 'white'}),
        dash_table.DataTable(id='stats-table', style_table={'overflowX': 'auto', 'backgroundColor': '#333'},
                             style_cell={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#333'})
    ], style={'marginTop': '40px', 'textAlign': 'center'}),

    dcc.Graph(id='portfolio-value-graph', style={'marginTop': '40px'}),

], style={'backgroundColor': '#222222', 'padding': '20px'})

@app.callback(
    [
        Output('stats-table', 'data'),
        Output('stats-table', 'columns'),
        Output('portfolio-value-graph', 'figure')
    ],
    Input('run-button', 'n_clicks'),
    State('initial-date', 'date'),
    State('final-date', 'date'),
    State('information-class', 'value'),
    State('universe-dropdown', 'value')
)
def run_backtest(n_clicks, init_date_str, final_date_str, information_class_str, selected_symbols):
    if n_clicks == 0:
        return [], [], go.Figure()

    init_date = datetime.strptime(init_date_str, "%Y-%m-%d")
    final_date = datetime.strptime(final_date_str, "%Y-%m-%d")

    # Map information class to your strategy
    information_class = {
        'first_two_moments': FirstTwoMoments,
        'momentum_strategy': MomentumStrategy,
        'mean_reversion_strategy': MeanReversionStrategy,
        'volatility_strategy': VolatilityStrategy,
        'max_sharpe_ratio': MaxSharpeRatio
    }.get(information_class_str, FirstTwoMoments)

   # Performing the Backtest
   
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

    # Compute and Display Portfolio's Performance 
    df_portfolio = pd.DataFrame(portfolio_values, columns=['Date', 'Portfolio Perf'])
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(
        go.Scatter(
            x=df_portfolio['Date'], 
            y=df_portfolio['Portfolio Perf'], 
            mode='lines', 
            name='Portfolio Perf'
        )
    )
    fig_portfolio.update_layout(
        title="Portfolio's Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Perf",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Prepare Currency Performance Data for Statistics Table
    final_market_prices = {
        ticker: last_prices.get(ticker, 0) for ticker in broker.positions.keys()
    }
    summary_data = []
    for ticker, position in broker.positions.items():
        avg_return = 0
        std_return = 0
        try:
            data = get_fx_pairs_data([ticker], init_date_str, final_date_str)
            if not data.empty:
                daily_returns = data['Close'].pct_change().dropna()
                avg_return = daily_returns.mean() * 100
                std_return = daily_returns.std() * 100
                
        except Exception:
            pass
        
        summary_data.append({
            "Ticker": ticker,
            "Annualized Return (%)": round(avg_return, 2),
            "Annualized Std Dev (%)": round(std_return, 2),
            "Last Price": final_market_prices[ticker],
            "Perf": position.quantity * final_market_prices[ticker]
        })

    # Sort the currencies by annualized return
    summary_data = sorted(summary_data, key=lambda x: x['Annualized Return (%)'], reverse=True)

    summary_columns = [
        {"name": "Ticker", "id": "Ticker"},
        {"name": "Annualized Return (%)", "id": "Annualized Return (%)"},
        {"name": "Annualized Std Dev (%)", "id": "Annualized Std Dev (%)"},
        {"name": "Last Price", "id": "Last Price"},
        {"name": "Value", "id": "Perf"}
    ]

    return summary_data, summary_columns, fig_portfolio

if __name__ == '__main__':
    app.run_server(debug=True)
