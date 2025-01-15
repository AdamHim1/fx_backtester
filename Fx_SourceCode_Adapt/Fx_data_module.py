import yfinance as yf
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Updated Constants for FX Universe, including EM currencies
#---------------------------------------------------------

# List of major FX pairs and some EM currencies 
UNIVERSE_FX = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",  
    "USDTRY=X", "USDMXN=X", "USDBRL=X", "USDINR=X", "USDSGD=X", "USDZAR=X", "USDCLP=X",  
    "USDPLN=X", "USDHUF=X", "USDRUB=X", "USDTHB=X", "USDKRW=X", "USDIDR=X", "USDCNY=X"  
]

#---------------------------------------------------------
# Functions for FX Data Retrieval
#---------------------------------------------------------

def get_fx_data(ticker, start_date, end_date):
    """get_fx_data retrieves historical data on prices for a given FX pair"""
    try:
        fx_pair = yf.Ticker(ticker)
        data = fx_pair.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        # Add ticker to the dataframe for identification
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Could not retrieve data for {ticker}: {e}")
        return pd.DataFrame()

def get_fx_pairs_data(tickers, start_date, end_date):
    """get_fx_pairs_data retrieves historical data for a list of FX pairs"""
    dfs = []
    for ticker in tickers:
        df = get_fx_data(ticker, start_date, end_date)
        if not df.empty:
            dfs.append(df)
    # Concatenate all dataframes into one
    if dfs:
        data = pd.concat(dfs)
        return data
    else:
        logging.warning("No data available for the provided FX tickers.")
        return pd.DataFrame()

#---------------------------------------------------------
# Classes
#---------------------------------------------------------

@dataclass
class DataModule:
    data: pd.DataFrame

@dataclass
class Information:
    s: timedelta = timedelta(days=360)  # Time window (rolling window)
    data_module: DataModule = None  # Data module containing the FX data
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t: datetime):
        """Slice the data to get the rolling window for backtest"""
        data = self.data_module.data
        s = self.s
        data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
        return data

    def get_prices(self, t: datetime):
        """Get the prices of FX pairs at time t"""
        data = self.slice_data(t)
        prices = data.groupby(self.company_column)[self.adj_close_column].last()
        prices = prices.to_dict()  # Return as dictionary
        return prices

    def compute_information(self, t: datetime):
        """Calculate the necessary information (e.g., returns, volatility)"""
        pass

    def compute_portfolio(self, t: datetime, information_set: dict):
        """Compute portfolio weights"""
        pass

@dataclass
class FirstTwoMoments(Information):
    def compute_portfolio(self, t: datetime, information_set):
        """Compute the portfolio based on the expected returns and covariance matrix"""
        mu = information_set['expected_return']
        Sigma = information_set['covariance_matrix']
        gamma = 1  # Risk aversion parameter
        n = len(mu)
        # Objective function for optimization
        obj = lambda x: -x.dot(mu) + gamma / 2 * x.dot(Sigma).dot(x)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0.0, 1.0)] * n  # Allow only long positions
        x0 = np.ones(n) / n  # Equal weight initialization
        res = minimize(obj, x0, constraints=cons, bounds=bounds)

        portfolio = {k: None for k in information_set['companies']}
        if res.success:
            for i, company in enumerate(information_set['companies']):
                portfolio[company] = res.x[i]
        else:
            logging.warning("Optimization did not converge.")
            portfolio = {k: 1 / len(information_set['companies']) for k in information_set['companies']}
        
        return portfolio

    def compute_information(self, t: datetime):
        """Compute the information set including expected returns and covariance"""
        data = self.slice_data(t)
        information_set = {}

        # Calculate returns per currency pair
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Compute covariance matrix
        data_pivot = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data_pivot = data_pivot.dropna(axis=0)
        covariance_matrix = data_pivot.cov().to_numpy()
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set
