#%%
import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------

UNIVERSE_SEC = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",  
    "USDTRY=X", "USDMXN=X", "USDBRL=X", "USDINR=X", "USDSGD=X", "USDZAR=X", "USDCLP=X",  
    "USDPLN=X", "USDHUF=X", "USDRUB=X", "USDTHB=X", "USDKRW=X", "USDIDR=X", "USDCNY=X"  ]

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

# function that retrieves historical data on prices for a given stock
def get_fx_data(ticker, start_date, end_date):
    """get_fx_data retrieves historical data on FX prices for a given FX pair.

    Args:
        ticker (str): The FX pair ticker (e.g., 'EURUSD=X').
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data.

    Example:
        df = get_fx_data('EURUSD=X', '2000-01-01', '2020-12-31')
    """
    try:
        fx_pair = yf.Ticker(ticker)
        data = fx_pair.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        
        if data.empty:
            logging.warning(f"No data found for {ticker}.")
            return pd.DataFrame()

        # If necessary, adjust the timezone or handle timezone issues here
        data['ticker'] = ticker
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        logging.warning(f"Could not retrieve data for {ticker}: {e}")
        return pd.DataFrame()


def get_fx_pairs_data(tickers, start_date, end_date):
    """get_fx_pairs_data retrieves historical data on FX prices for a list of FX pairs."""
    dfs = []
    for ticker in tickers:
        df = get_fx_data(ticker, start_date, end_date)
        if not df.empty:
            dfs.append(df)
    
    if dfs:
        # Concatenate all dataframes and reset index
        combined_data = pd.concat(dfs)
        combined_data.reset_index(drop=True, inplace=True)
        return combined_data
    else:
        logging.warning("No FX data available.")
        return pd.DataFrame()


#---------------------------------------------------------
# Classes 
#---------------------------------------------------------

# Class that represents the data used in the backtest. 
@dataclass
class DataModule:
    data: pd.DataFrame

# Interface for the information set 
@dataclass
class Information:
    s: timedelta = timedelta(days=360) # Time step (rolling window)
    data_module: DataModule = None # Data module
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t : datetime):

        # Get the data module 
        data = self.data_module.data
        # Get the time step 
        s = self.s

        # Convert both `t` and the data column to timezone-aware, if needed
        if t.tzinfo is not None:
            # If `t` is timezone-aware, make sure data is also timezone-aware
            data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(t.tzinfo.zone, ambiguous='NaT', nonexistent='NaT')
        else:
            # If `t` is timezone-naive, ensure the data is timezone-naive as well
            data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(None)
        
        # Get the data only between t-s and t
        data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
        return data

    def get_prices(self, t : datetime):
        # gets the prices at which the portfolio will be rebalanced at time t 
        data = self.slice_data(t)
        
        # get the last price for each company
        prices = data.groupby(self.company_column)[self.adj_close_column].last()
        # to dict, ticker as key price as value 
        prices = prices.to_dict()
        return prices

    def compute_information(self, t : datetime):  
        pass

    def compute_portfolio(self, t : datetime,  information_set : dict):
        pass
     
@dataclass
class FirstTwoMoments(Information):
    def compute_portfolio(self, t:datetime, information_set):
        try:
            mu = information_set['expected_return']
            Sigma = information_set['covariance_matrix']
            gamma = 1 # risk aversion parameter
            n = len(mu)
            # objective function
            obj = lambda x: -x.dot(mu) + gamma/2 * x.dot(Sigma).dot(x)
            # constraints
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # bounds, allow short selling, +- inf 
            bounds = [(0.0, 1.0)] * n
            # initial guess, equal weights
            x0 = np.ones(n) / n
            # minimize
            res = minimize(obj, x0, constraints=cons, bounds=bounds)

            # prepare dictionary 
            portfolio = {k: None for k in information_set['companies']}

            # if converged update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio
        except Exception as e:
            # if something goes wrong return an equal weight portfolio but let the user know 
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t : datetime):
        # Get the data module 
        data = self.slice_data(t)
        # the information set will be a dictionary with the data
        information_set = {}

        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # expected return per company
        data['return'] =  data.groupby(self.company_column)[self.adj_close_column].pct_change() #.mean()
        
        # expected return by company 
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

        # 1. pivot the data
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        # drop missing values
        data = data.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data.cov()
        # convert to numpy matrix 
        covariance_matrix = covariance_matrix.to_numpy()
        # add to the information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        return information_set

