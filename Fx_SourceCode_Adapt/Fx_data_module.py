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
    
@dataclass
class MaxSharpeRatio(Information):
    def compute_portfolio(self, t: datetime, info_set: dict):
        try:
            expected_returns = info_set['expected_return']
            covariance_matrix = info_set['covariance_matrix']
            num_assets = len(expected_returns)

            # Max Sharpe ratio objective function
            def negative_sharpe_ratio(weights):
                epsilon = 0.0000000000001  
                numerator = weights.dot(expected_returns)
                denominator = np.sqrt(weights.dot(covariance_matrix).dot(weights)) + epsilon
                return -numerator / denominator  # Negate for minimization

            # Constraints: sum of weights equals 1 
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
            bounds = [(0.0, 1.0)] * num_assets

            # Initial guess: equal weights for each currency pair
            initial_guess = np.ones(num_assets) / num_assets

            # Run the optimization
            result = minimize(negative_sharpe_ratio,
                              initial_guess,
                              constraints=constraints,
                              bounds=bounds,
                              method='SLSQP')

            # Prepare portfolio allocation dictionary
            portfolio_allocation = {pair: None for pair in info_set['currency_pairs']}

            # If optimization is successful, update portfolio allocation
            if result.success:
                for i, currency_pair in enumerate(info_set['currency_pairs']):
                    portfolio_allocation[currency_pair] = result.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio_allocation

        except Exception as e:
            # Return an equal weight portfolio in case of Error
            return {pair: 1/len(info_set['currency_pairs']) for pair in info_set['currency_pairs']}
        
    def compute_information(self, t: datetime):
        # Get the data for the given time period
        data = self.slice_data(t)
        info_set = {}

        # Sort data by currency pair and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Calculate returns for each currency pair
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Expected return by currency pair
        info_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Calculate covariance matrix for the currency pairs

        # 1. Pivot the data
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        # Drop missing values
        data_pivot = data_pivot.dropna(axis=0)
        # 2. Compute the covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # Add to information set
        info_set['covariance_matrix'] = covariance_matrix
        info_set['currency_pairs'] = data_pivot.columns.to_numpy()

        return info_set
    
@dataclass
class MomentumStrategy(Information):
    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            # Get the expected returns (momentum indicator) from the information set
            expected_returns = information_set['expected_return']
            
            # Sort the currencies by the highest 30-day return 
            sorted_currencies = expected_returns.argsort()[::-1]  
            top_currencies = sorted_currencies[:len(sorted_currencies)//2] 

            # Initialize portfolio with all weights set to zero
            portfolio = {company: 0 for company in information_set['companies']}
            
            # Allocate equal weight to the top half of the currencies
            weight = 1 / len(top_currencies)
            for i in top_currencies:
                portfolio[information_set['companies'][i]] = weight

            return portfolio

        except Exception as e:
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t: datetime):
        # Get the data for the current time step
        data = self.slice_data(t)
        
        # Initialize the information set to store the computed values
        information_set = {}

        # Sort data by company and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Compute the daily return for each currency
        data.loc[:, 'return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Calculate the expected return for each currency (momentum indicator: 30-day return)
        information_set['expected_return'] = data.groupby(self.company_column)['return'].apply(lambda x: x.tail(30).mean()).to_numpy()

        # Pivot the data to compute the covariance matrix
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        
        # Drop missing values
        data_pivot = data_pivot.dropna(axis=0)

        # Compute the covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # Add the computed covariance matrix and company names to the information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set

@dataclass
class MeanReversionStrategy(Information):
    window: int = 30  

    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            # Get expected returns and compute z-scores
            expected_returns = information_set['expected_return']
            rolling_mean = np.mean(expected_returns)
            rolling_std = np.std(expected_returns)

            # Z-score (deviation from mean)
            z_scores = (expected_returns - rolling_mean) / rolling_std

            # Create portfolio: Buy if z-score is negative (mean below), Sell if positive
            portfolio = {company: 0 for company in information_set['companies']}
            long_currencies = np.where(z_scores < -1)[0]  
            short_currencies = np.where(z_scores > 1)[0]  

            # Allocate portfolio
            weight_long = 1 / len(long_currencies) if len(long_currencies) > 0 else 0
            weight_short = -1 / len(short_currencies) if len(short_currencies) > 0 else 0

            for i in long_currencies:
                portfolio[information_set['companies'][i]] = weight_long
            for i in short_currencies:
                portfolio[information_set['companies'][i]] = weight_short

            return portfolio

        except Exception as e:
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Compute returns
        data.loc[:, 'return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Expected returns for mean reversion: use rolling mean and std deviation
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Covariance matrix
        data_pivot = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data_pivot = data_pivot.dropna(axis=0)
        covariance_matrix = data_pivot.cov().to_numpy()

        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set

@dataclass
class VolatilityStrategy(Information):
    def compute_portfolio(self, t: datetime, information_set: dict):
        """Construct portfolio based on volatility strategy (top 5 long, bottom 5 short)."""
        try:
            volatilities = information_set['volatility']
            sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)

            # Select top 5 volatile currencies for long and bottom 5 for short
            long_positions = sorted_volatilities[:5]
            short_positions = sorted_volatilities[-5:]

            portfolio = {}
            for currency, _ in long_positions:
                portfolio[currency] = 1 / 10  # Long position for top 5

            for currency, _ in short_positions:
                portfolio[currency] = -1 / 10  # Short position for bottom 5

            return portfolio

        except Exception as e:
            logging.warning(f"Error computing Volatility strategy portfolio: {e}")
            return {}

    def compute_information(self, t: datetime):
        """Compute volatility information set."""
        data = self.slice_data(t)
        information_set = {}

        # Sort data by currency and time
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Compute returns
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Compute volatility (30-day rolling standard deviation of returns)
        data['volatility'] = data.groupby(self.company_column)['return'].rolling(window=30).std().reset_index(level=0, drop=True)

        # Collect the average volatility for each currency
        information_set['volatility'] = data.groupby(self.company_column)['volatility'].mean().to_dict()

        return information_set

@dataclass
class CorrelationStrategy(Information):
    lookback_window: int = 30  # Lookback period to calculate correlation

    def compute_portfolio(self, t: datetime, information_set: dict):
        try:
            # Get the expected returns (as a proxy for momentum) and covariance (for correlation)
            returns = information_set['expected_return']
            covariance_matrix = information_set['covariance_matrix']
            
            # Calculate correlation matrix from covariance matrix
            correlation_matrix = np.corrcoef(covariance_matrix)

            # Create an empty portfolio dictionary
            portfolio = {currency: 0 for currency in information_set['companies']}

            # Initialize long and short lists
            long_currencies = []
            short_currencies = []
            num_currencies = len(correlation_matrix)

            # Loop through pairs to identify which to go long and which to short
            for i in range(num_currencies):
                for j in range(i + 1, num_currencies):
                    # Long for negatively correlated pairs
                    if correlation_matrix[i, j] < -0.5:  # threshold for low correlation
                        long_currencies.append(i)
                        long_currencies.append(j)
                    # Short for highly positively correlated pairs
                    elif correlation_matrix[i, j] > 0.5:  # threshold for high correlation
                        short_currencies.append(i)
                        short_currencies.append(j)

            # Calculate the weight for long and short positions
            total_positions = len(long_currencies) + len(short_currencies)
            long_weight = 1 / total_positions if total_positions > 0 else 0
            short_weight = -1 / total_positions if total_positions > 0 else 0

            # Assign weights to long and short positions
            for idx in long_currencies:
                portfolio[information_set['companies'][idx]] = long_weight
            for idx in short_currencies:
                portfolio[information_set['companies'][idx]] = short_weight

            # Log if no positions were allocated
            if sum(portfolio.values()) == 0:
                logging.warning("No portfolio allocation assigned.")

            return portfolio

        except Exception as e:
            logging.warning(f"Error computing Correlation strategy portfolio: {e}")
            # Return equal weights if there's an error
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}

        # Sort data by currency and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # Compute returns for each currency pair
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()

        # Calculate expected return for each currency pair
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Pivot data to calculate covariance matrix (for correlation)
        data_pivot = data.pivot(index=self.time_column, 
                                columns=self.company_column, 
                                values=self.adj_close_column)
        
        # Drop missing values
        data_pivot = data_pivot.dropna(axis=0)

        # Calculate covariance matrix
        covariance_matrix = data_pivot.cov().to_numpy()

        # Add covariance matrix and companies to information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data_pivot.columns.to_numpy()

        return information_set


